"""Build the single table for the five-state relationship model.

Input  : raw_data/Eastwick/data and raw_data/HCMST/raw.
Output : dataset/relationship_five_state_dataset.csv  (one row = one pair x one wave)

Five states:
  NR  No relationship   - never related / no tie / broke up / divorced / separated / partner died
  NP  No romantic potential  - acquaintance or friend, no potential
  RP  Romantic potential     - acquaintance or friend with potential (incl. casual dating)
  CR  Committed relationship - serious dating / main romantic partner, incl. unmarried cohabitation
  M   Married
A row whose original answer cannot be mapped is kept with state = "missing".
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RE = ROOT / "raw_data/Eastwick/data"
RH = ROOT / "raw_data/HCMST/raw"
STATES = ["NR", "NP", "RP", "CR", "M"]
WAVE_DAYS_EW = 21.0          # Eastwick waves ~3 weeks apart (median 21 d in raw dates)
SEED = 20261019

# ------------------------------------------------------------------ helpers
def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) else np.nan

def clean_neg(s):
    """Stata survey codes: -1 refused, -2 not asked -> NaN."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s >= 0)

def yrmo_to_years(v):
    n = num(v)
    if not np.isfinite(n):
        return np.nan
    yr, mo = divmod(int(n), 100)
    return yr + (mo - 0.5) / 12 if 1900 < yr < 2100 and 1 <= mo <= 12 else np.nan

def yrmo_to_date(v):
    n = num(v)
    if not np.isfinite(n):
        return ""
    yr, mo = divmod(int(n), 100)
    return f"{yr:04d}-{mo:02d}" if 1900 < yr < 2100 and 1 <= mo <= 12 else ""

# ------------------------------------------------------------------ Eastwick
EW_MAP = {1: ("NR", "R_Status=1 no relationship of any sort"),
          2: ("NP", "R_Status=2 acquaintance, no romantic potential"),
          4: ("NP", "R_Status=4 friend, no romantic potential"),
          3: ("RP", "R_Status=3 acquaintance, romantic potential"),
          5: ("RP", "R_Status=5 friend, romantic potential"),
          6: ("RP", "R_Status=6 dating casually -> potential, not yet confirmed"),
          7: ("CR", "R_Status=7 dating seriously")}
EW_RAW_LABEL = {1: "no relationship", 2: "acquaintance no potential", 3: "acquaintance potential",
                4: "friend no potential", 5: "friend potential", 6: "dating casually",
                7: "dating seriously", 8: "skipped"}

def skip_codes():
    led = pd.read_csv(OUT / "meta/variable_ledger.csv")
    out = {}
    for v, opt in zip(led.variable, led.response_options.fillna("")):
        m = re.findall(r"(\d+)\s*=\s*Skip", str(opt), flags=re.I)
        if m:
            out[v] = [int(x) for x in m]
    return out

def build_eastwick():
    st = pd.read_excel(RE / "IDswithRelStatus.xlsx")
    st.columns = ["ID", "Relationship_Number", "Target", "Wave", "R_Status"]
    st["ID"] = st.ID.astype(int)
    feats = json.loads((OUT / "meta/candidate_features.json").read_text())
    person_vars = feats["participant_baseline"]
    target_vars = feats["target_entry"]
    sas = pd.read_sas(RE / "SamPredictors.sas7bdat")
    sas["Target"] = sas.Target.map(lambda x: x.decode() if isinstance(x, bytes) else str(x))
    for c in ["ID", "Relationship_Number", "Wave"]:
        sas[c] = sas[c].astype(int)
    keep = ["ID", "Relationship_Number", "Target", "Wave", "RomIntTimeVarying"] + person_vars + target_vars
    keep = list(dict.fromkeys(keep))
    df = st.merge(sas[keep], on=["ID", "Relationship_Number", "Target", "Wave"], how="left",
                  validate="one_to_one", indicator=True)
    assert (df._merge == "both").all(), df._merge.value_counts()
    df = df.drop(columns="_merge")
    # entry questionnaire: partner age / ethnicity, participant family background
    ent = pd.read_csv(RE / "full initiation data for osf.csv", low_memory=False,
                      usecols=["Target", "WaveEnter", "Pwave", "START", "R_AGE", "R_ETH", "R_RACE",
                               "In_ETH", "In_RACE", "In_INCOME", "In_MEDU", "In_FEDU"])
    ent["Target"] = ent.Target.astype(str)
    assert ent.Target.is_unique
    df = df.merge(ent.drop(columns=["Pwave", "START", "In_MEDU"]), on="Target", how="left", validate="many_to_one")
    # actual wave start dates known for participant-waves where a target entered
    ent["pid"] = ent.Pwave.astype(str).str.extract(r"^(\d+)v")[0].astype(float)
    ent["d"] = pd.to_datetime(ent.START, errors="coerce")
    dates = ent.dropna(subset=["pid", "d"]).groupby(["pid", "WaveEnter"]).d.min()
    # skip codes -> NaN
    sk = skip_codes()
    for c in df.columns:
        if c in sk:
            df[c] = pd.to_numeric(df[c], errors="coerce").where(lambda s: ~s.isin(sk[c]))
    df["R_AGE"] = pd.to_numeric(df.R_AGE, errors="coerce").where(lambda s: s.between(15, 100))

    rows = []
    first_wave = df.groupby("Target").Wave.transform("min")
    for i, r in df.iterrows():
        code = num(r.R_Status)
        if np.isfinite(code) and int(code) in EW_MAP:
            state, note = EW_MAP[int(code)]
            miss = ""
        else:
            state, note = "missing", "R_Status blank/skip in raw file"
            miss = "no_status_answer_this_wave"
        pid = int(r.ID)
        d = dates.get((float(pid), int(r.Wave)))
        t = (r.Wave - first_wave[i]) * WAVE_DAYS_EW / 365.25
        age = num(r.S_AGE) + (r.Wave - 1) * WAVE_DAYS_EW / 365.25
        row = dict(source="Eastwick", person_id=f"EW:{pid}", pair_id=f"EW:{r.Target}",
                   wave=int(r.Wave), obs_date=d.strftime("%Y-%m-%d") if d is not None and pd.notna(d) else "",
                   time_years=t, time_basis="nominal 21 days per wave since pair first observed",
                   raw_status=("" if not np.isfinite(code) else str(int(code))),
                   raw_status_label=EW_RAW_LABEL.get(int(code), "") if np.isfinite(code) else "blank",
                   state=state, mapping_note=note, missing_reason=miss,
                   nr_type="no_tie" if state == "NR" else "", partner_deceased=0, cohabiting=np.nan,
                   resp_age=age, resp_gender={1: "male", 2: "female"}.get(int(num(r.S_GENDER)), "") if np.isfinite(num(r.S_GENDER)) else "",
                   partner_age=num(r.R_AGE),
                   partner_gender={1: "female", 2: "male"}.get(int(num(r.S_GENDER)), "") if np.isfinite(num(r.S_GENDER)) else "",
                   same_sex=0, known_years=np.nan, romantic_years=np.nan, cohab_years=np.nan,
                   married_years=np.nan, resp_edu_years=np.nan, partner_edu_years=np.nan,
                   hh_income_band=np.nan,
                   romantic_interest_self=num(r.RomIntTimeVarying),
                   partner_interest_perceived=num(r.R_RLSHP1_2),
                   rel_quality_1to5=np.nan)
        for c in person_vars:
            row[f"ew_{c}"] = num(r[c])
        for c in target_vars:
            row[f"ew_{c}"] = num(r[c])
        for c in ["R_ETH", "R_RACE", "In_ETH", "In_RACE", "In_FEDU"]:
            row[f"ew_{c}"] = num(r[c])
        row["ew_In_INCOME_family_of_origin"] = num(r.In_INCOME)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["partner_age_minus_resp"] = out.partner_age - out.resp_age
    rep = dict(raw_rows=len(st), participants=st.ID.nunique(), pairs=st.Target.nunique())
    return out, rep

# ------------------------------------------------------------------ HCMST
def yn(v, yes=1, no=2):
    n = num(v)
    return True if n == yes else False if n == no else None

def h_state(w, r, married_before):
    """Map one HCMST wave, then reconcile with the official derived broke_up flag."""
    res = _h_state(w, r, married_before)
    if w == 1:
        return res
    bu = num(r.get("w6_broke_up" if w == 6 else f"w{w}_broke_up"))
    state, nrt, dec, cohab, raw, note = res
    if bu == 1 and state in ("M", "CR"):
        return "NR", "ended_per_official_flag", 0, None, raw, "answered still together but official broke_up flag (uses text answers) = broke up"
    if bu == 0 and state == "NR" and dec != 1:
        return "missing", "", np.nan, None, raw, "conflicting raw answers (official flag says still together)"
    return res

def _h_state(w, r, married_before):
    """Return (state, nr_type, deceased, cohab, raw_string, note)."""
    if w == 1:
        raw = f"qflag={num(r.qflag):.0f};married={num(r.married):.0f};q19={num(r.q19):.0f};partner_deceased={num(r.partner_deceased):.0f}"
        if num(r.partner_deceased) == 1:
            return "NR", "partner_deceased", 1, None, raw, "W1 text answers imply reported partner deceased"
        c = yn(r.q19)
        if num(r.married) == 1:
            return "M", "", 0, c, raw, "W1 married to partner"
        return "CR", "", 0, c, raw, "W1 unmarried main romantic partner (cohabiting or not) -> committed relationship"
    if w == 6:
        g = lambda k: r.get(k)
        raw = ";".join(f"{k}={num(g(k)):.0f}" for k in ["w6_still_mar", "w6_relationship_end_mar", "w6_still_rel",
                       "w6_got_married", "w6_relationship_end_nonmar", "w6_live_together", "w6_live_together_C", "w6_broke_up"]
                       if np.isfinite(num(g(k))))
        q1, q3, q5, q7, q9 = yn(g("w6_still_mar")), num(g("w6_relationship_end_mar")), yn(g("w6_still_rel")), yn(g("w6_got_married")), num(g("w6_relationship_end_nonmar"))
        cm, cu = yn(g("w6_live_together")), yn(g("w6_live_together_C"))
        bu = num(g("w6_broke_up"))
    else:
        p = f"w{w}_"
        g = lambda k: r.get(p + k)
        no5 = 0 if w == 4 else 2
        raw = ";".join(f"{p}{k}={num(g(k)):.0f}" for k in ["q1", "q2", "q3", "q5", "q6", "q7", "q9", "broke_up"]
                       if np.isfinite(num(g(k))))
        q1, q3 = yn(g("q1")), num(g("q3"))
        q5, q7, q9 = yn(g("q5"), 1, no5), yn(g("q7")), num(g("q9"))
        cm, cu = yn(g("q2")), yn(g("q6"), 1, no5)
        bu = num(g("broke_up"))
        if w == 2 and num(r.get("w2_deceased")) == 1:
            raw += ";w2_deceased=1"
            return "NR", "partner_deceased", 1, None, raw, "W2 text implies partner deceased"
    dead_codes_q9 = (2, 3) if w == 4 else (2,)
    if q3 == 3 or q9 in dead_codes_q9 or bu == 2:
        return "NR", "partner_deceased", 1, None, raw, "partner passed away"
    if q1 is True:
        return "M", "", 0, cm, raw, "still married to partner"
    if q1 is False:
        t = {1: "divorced", 2: "separated"}.get(q3, "marriage_ended")
        return "NR", t, 0, None, raw, f"marriage ended ({t}) -> relationship terminated"
    if q5 is True:
        if q7 is True:
            return "M", "", 0, cu, raw, "married partner since last wave"
        return "CR", "", 0, cu, raw, "still in romantic relationship, unmarried"
    if q5 is False:
        t = "broke_up" if q9 == 1 else "ended_other"
        return "NR", t, 0, None, raw, f"romantic relationship ended ({t}) -> relationship terminated"
    if bu == 1:
        return "NR", "broke_up", 0, None, raw, "broke_up flag (question items blank)"
    return "missing", "", np.nan, None, raw, ""

def build_hcmst():
    b = pd.read_stata(RH / "HCMST_ver_3.04.dta", convert_categoricals=False)
    w4 = pd.read_stata(RH / "wave_4_supplement_v1_2.dta", convert_categoricals=False)
    w5 = pd.read_stata(RH / "HCMST_wave_5_supplement_ver_1.dta", convert_categoricals=False)
    w6 = pd.read_stata(RH / "HCMST_wave_6_public_v1.dta", convert_categoricals=False)
    for d in (b, w4, w5, w6):
        assert d.caseid_new.is_unique
    b = b.merge(w4, on="caseid_new", how="left", validate="1:1").merge(w5, on="caseid_new", how="left", validate="1:1")
    b = b.merge(w6, on="caseid_new", how="left", validate="1:1", indicator="in_w6")
    n_all = len(b)
    q = b.loc[b.qflag == 1].copy()          # respondents with a partner = a pair exists
    date_cols = {1: "HCMST_main_interview_yrmo", 2: "w2_HCMST_interview_fin_yrmo", 3: "w3_HCMST_interview_fin_yrmo",
                 4: "w4_HCMST_interview_fin_yrmo", 5: "w5_HCMST_interview_fin_yrmo", 6: "w6_HCMST_yrmo"}
    inc_cols = {1: "ppincimp", 2: "pp2_ppincimp", 3: "pp3_ppincimp", 4: "pp4_ppincimp", 5: "pp5_ppincimp", 6: "pp6_ppincimp"}
    base_vars = ["ppeduc", "ppeducat", "q10", "ppincimp", "ppethm", "respondent_race", "partner_race", "ppwork",
                 "ppmarit", "ppreg4", "ppmsacat", "children_in_hh", "q23", "q25", "q26", "q27", "q28", "q32_internet",
                 "q24_met_online", "relationship_quality", "same_sex_couple", "age_difference",
                 "respondent_religion_at_16", "partner_religion_at_16", "papglb_status", "q21a", "q21b", "q21c", "q21d"]
    rows = []
    for _, r in q.iterrows():
        cid = int(r.caseid_new)
        t0 = yrmo_to_years(r[date_cols[1]])
        age0, page0 = num(clean_neg(pd.Series([r.ppage]))[0]), num(clean_neg(pd.Series([r.q9]))[0])
        met0, rom0, coh0 = num(r.how_long_ago_first_met), num(r.how_long_relationship), num(r.how_long_ago_first_cohab)
        mar_age = num(r.q21d) if num(r.married) == 1 and num(r.q21d) > 0 else np.nan
        pg = {1: "male", 2: "female"}.get(int(num(r.q4)), "") if np.isfinite(num(r.q4)) else ""
        rg = {1: "male", 2: "female"}.get(int(num(r.ppgender)), "")
        prev = None; ended = False; married_before = num(r.married) == 1
        for w in range(1, 7):
            if w == 6 and r.in_w6 != "both":
                state, nrt, dec, cohab, raw, note = "missing", "", np.nan, None, "", ""
            else:
                state, nrt, dec, cohab, raw, note = h_state(w, r, married_before)
            ty = yrmo_to_years(r.get(date_cols[w]))
            e = ty - t0 if np.isfinite(ty) and np.isfinite(t0) else np.nan
            miss = ""
            if state == "missing":
                miss = "not_followed_after_relationship_ended" if ended else "no_response_or_not_in_this_wave"
                note = note or "no usable answer in raw file for this wave"
                if note.startswith("conflicting"):
                    miss = "conflicting_raw_answers"
            together = state in ("CR", "M")
            # marriage start for new marriages (W4-W6 give year/month)
            if state == "M" and not np.isfinite(mar_age):
                yk = {4: ("w4_mar_year", "w4_mar_month"), 5: ("w5_mar_year", "w5_mar_month"), 6: ("w6_martiming_Year", "w6_martiming_Month")}.get(w)
                if yk and np.isfinite(num(r.get(yk[0]))) and num(r.get(yk[0])) > 1900 and np.isfinite(t0):
                    mo = num(r.get(yk[1])); mo = mo if 1 <= mo <= 12 else 6.5
                    mar_age = age0 + (num(r.get(yk[0])) + (mo - .5) / 12 - t0) if np.isfinite(age0) else np.nan
            row = dict(source="HCMST", person_id=f"HC:{cid}", pair_id=f"HC:{cid}", wave=w,
                       obs_date=yrmo_to_date(r.get(date_cols[w])), time_years=e,
                       time_basis="interview month minus W1 interview month",
                       raw_status=raw,
                       raw_status_label="", state=state, mapping_note=note, missing_reason=miss,
                       nr_type=nrt, partner_deceased=dec,
                       cohabiting=(1 if cohab is True else 0 if cohab is False else np.nan),
                       resp_age=age0 + (e if np.isfinite(e) else 0) if w == 1 or np.isfinite(e) else np.nan,
                       resp_gender=rg, partner_age=page0 + (e if np.isfinite(e) else 0) if w == 1 or np.isfinite(e) else np.nan,
                       partner_gender=pg, same_sex=num(r.same_sex_couple),
                       known_years=(met0 + e) if np.isfinite(met0) and (w == 1 or np.isfinite(e)) else np.nan,
                       romantic_years=(rom0 + (e if w > 1 else 0)) if together and np.isfinite(rom0) and (w == 1 or np.isfinite(e)) else np.nan,
                       cohab_years=(coh0 + (e if w > 1 else 0)) if cohab is True and np.isfinite(coh0) and (w == 1 or np.isfinite(e)) else np.nan,
                       married_years=np.nan,
                       resp_edu_years=num(r.respondent_yrsed), partner_edu_years=num(r.partner_yrsed),
                       hh_income_band=min(num(clean_neg(pd.Series([r.get(inc_cols[w])]))[0]), 19.0),  # W6 splits >=$175k into 19-21
                       romantic_interest_self=np.nan, partner_interest_perceived=np.nan,
                       rel_quality_1to5=(num(r.relationship_quality) if w == 1 else num(r.relationship_quality_w4) if w == 4 else np.nan))
            if state == "M" and np.isfinite(mar_age) and np.isfinite(row["resp_age"]) and row["resp_age"] >= mar_age:
                row["married_years"] = row["resp_age"] - mar_age
            for c in base_vars:
                row[f"hc_{c}"] = num(clean_neg(pd.Series([r.get(c)]))[0])
            rows.append(row)
            if state == "NR":
                ended = True
            elif state in ("CR", "M", "NP", "RP"):
                ended = False
            prev = state
    out = pd.DataFrame(rows)
    out["partner_age_minus_resp"] = out.partner_age - out.resp_age
    rep = dict(raw_respondents=n_all, respondents_with_partner=len(q), respondents_without_partner_excluded=int(n_all - len(q)),
               w6_rows=int((b.in_w6 == "both").sum()))
    return out, rep

def main():
    e, er = build_eastwick()
    h, hr = build_hcmst()
    df = pd.concat([e, h], ignore_index=True, sort=False)
    df = df.sort_values(["source", "pair_id", "wave"]).reset_index(drop=True)
    # one split over all people from both sources pooled together (person-level).
    persons = df.drop_duplicates("person_id")[["person_id", "source"]].sort_values("person_id")
    rng = np.random.default_rng(SEED)
    persons = persons.iloc[rng.permutation(len(persons))].reset_index(drop=True)
    # interleave by source so each source keeps ~20% test (sources are not split separately)
    persons["k"] = persons.groupby("source").cumcount()
    persons["n"] = persons.groupby("source").person_id.transform("size")
    persons["split"] = np.where(persons.k < np.round(0.2 * persons.n), "test", "train")
    tr = persons.split == "train"
    persons.loc[tr, "cv_fold"] = np.arange(tr.sum()) % 5
    df = df.merge(persons[["person_id", "split", "cv_fold"]], on="person_id", how="left", validate="many_to_one")
    for s in STATES:
        df[s] = np.where(df.state == "missing", np.nan, (df.state == s).astype(float))
    df["state_missing"] = (df.state == "missing").astype(int)
    grp = df.groupby("pair_id", sort=False)
    df["prev_state"] = grp.state.shift(1).fillna("")
    df["next_state"] = grp.state.shift(-1).fillna("")
    df["next_wave"] = grp.wave.shift(-1)
    df["interval_years_to_next"] = grp.time_years.shift(-1) - df.time_years
    df.insert(0, "row_id", np.arange(1, len(df) + 1))
    core = ["row_id", "source", "person_id", "pair_id", "wave", "obs_date", "time_years", "time_basis", "split", "cv_fold",
            "raw_status", "raw_status_label", "state", *STATES, "state_missing", "missing_reason", "mapping_note",
            "nr_type", "partner_deceased", "cohabiting", "prev_state", "next_state", "next_wave", "interval_years_to_next",
            "resp_age", "resp_gender", "partner_age", "partner_gender", "partner_age_minus_resp", "same_sex",
            "known_years", "romantic_years", "cohab_years", "married_years", "resp_edu_years", "partner_edu_years",
            "hh_income_band", "romantic_interest_self", "partner_interest_perceived", "rel_quality_1to5"]
    rest = [c for c in df.columns if c not in core]
    df = df[core + sorted([c for c in rest if c.startswith("ew_")]) + sorted([c for c in rest if c.startswith("hc_")])]
    assert df.pair_id.notna().all() and not df.duplicated(["pair_id", "wave"]).any()
    assert set(df.state) <= set(STATES) | {"missing"}
    path = OUT / "relationship_five_state_dataset.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    summary = dict(eastwick=er, hcmst=hr, rows=len(df), columns=df.shape[1],
                   by_source_state=df.groupby(["source", "state"]).size().unstack(fill_value=0).to_dict(orient="index"),
                   by_source_split_persons=df.groupby(["source", "split"]).person_id.nunique().unstack().to_dict(orient="index"),
                   cv_fold_persons=df.drop_duplicates("person_id").groupby(["cv_fold", "source"]).size().unstack().to_dict(orient="index"),
                   missing_reasons=df.loc[df.state == "missing"].groupby(["source", "missing_reason"]).size().to_dict())
    summary["missing_reasons"] = {"/".join(k): int(v) for k, v in summary["missing_reasons"].items()}
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))

if __name__ == "__main__":
    main()
