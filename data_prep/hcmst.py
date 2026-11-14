# HCMST 2009 to 2017 and HCMST 2017 to 2022: wave-by-wave state coding and the person by wave panels.
import numpy as np

def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) else np.nan

def yn(v, yes=1, no=2):
    n = num(v)
    return True if n == yes else False if n == no else None

def _h_state(w, r, married_before):

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

def h_state(w, r, married_before):

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

def yrmo_to_years(v):
    n = num(v)
    if not np.isfinite(n):
        return np.nan
    yr, mo = divmod(int(n), 100)
    return yr + (mo - 0.5) / 12 if 1900 < yr < 2100 and 1 <= mo <= 12 else np.nan

def step_hcmst2009():

    from pathlib import Path
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]

    RH = ROOT / "raw_data/HCMST/raw"
    CACHE = ROOT / "cache"
    b = pd.read_stata(RH / "HCMST_ver_3.04.dta", convert_categoricals=False)
    w4 = pd.read_stata(RH / "wave_4_supplement_v1_2.dta", convert_categoricals=False)
    w5 = pd.read_stata(RH / "HCMST_wave_5_supplement_ver_1.dta", convert_categoricals=False)
    w6 = pd.read_stata(RH / "HCMST_wave_6_public_v1.dta", convert_categoricals=False)
    b = b.merge(w4, on="caseid_new", how="left", validate="1:1").merge(w5, on="caseid_new", how="left", validate="1:1")
    b = b.merge(w6, on="caseid_new", how="left", validate="1:1", indicator="in_w6")
    lab = pd.read_stata(RH / "HCMST_ver_3.04.dta")
    assert (lab.caseid_new.to_numpy() == pd.read_stata(RH / "HCMST_ver_3.04.dta", convert_categoricals=False).caseid_new.to_numpy()).all()
    lab = lab.set_index("caseid_new")
    q = b[b.qflag == 1]
    assert len(q) == 3009
    DATE = {1: "HCMST_main_interview_yrmo", 2: "w2_HCMST_interview_fin_yrmo", 3: "w3_HCMST_interview_fin_yrmo",
            4: "w4_HCMST_interview_fin_yrmo", 5: "w5_HCMST_interview_fin_yrmo", 6: "w6_HCMST_yrmo"}

    def inc_mid(v):
        v = v.astype(str)
        lo_hi = v.str.replace(",", "").str.extract(r"\$(\d+)\s*to\s*\$(\d+)").astype(float)
        mid = lo_hi.mean(axis=1)
        mid[v.str.lower() == "less than $5,000"] = 2500
        more = v.str.extract(r"\$([\d,]+) or more")[0].str.replace(",", "").astype(float)
        mid[more.notna()] = more[more.notna()] * 1.25
        return mid

    QUAL = {"very poor": 1, "poor": 2, "fair": 3, "good": 4, "excellent": 5}
    L = lab.loc[q.caseid_new]
    fixed = pd.DataFrame({
        "female": (L.ppgender.astype(str) == "female").astype(float).to_numpy(),
        "black": (L.ppethm.astype(str) == "black, non-hispanic").astype(float).to_numpy(),
        "hispanic": (L.ppethm.astype(str) == "hispanic").astype(float).to_numpy(),
        "edu_years": pd.to_numeric(L.respondent_yrsed, errors="coerce").to_numpy(),
        "same_sex": (L.same_sex_couple.astype(str) == "same-sex couple").astype(float).to_numpy(),
        "met_online": np.where(L.q24_met_online.isna(), np.nan, (L.q24_met_online.astype(str) == "met online").astype(float)),
        "weight": pd.to_numeric(L.weight2, errors="coerce").to_numpy(),
    }, index=q.caseid_new.astype(int).to_numpy())
    inc1 = pd.Series(inc_mid(L.ppincimp).to_numpy(), index=fixed.index)
    qual1 = pd.Series(L.relationship_quality.astype(str).map(QUAL).to_numpy(), index=fixed.index)

    rows = []
    for _, r in q.iterrows():
        cid = int(r.caseid_new)
        t0 = yrmo_to_years(r[DATE[1]])
        age0 = num(r.ppage)
        dur0 = num(r.how_long_relationship)
        for w in range(1, 7):
            if w == 6 and r.in_w6 != "both":
                continue
            state, nrt, dec, cohab, raw, note = h_state(w, r, num(r.married) == 1)
            if state == "missing":
                continue
            ty = yrmo_to_years(r.get(DATE[w]))
            if not np.isfinite(ty):
                continue
            if state == "M":
                st = "M"
            elif state == "CR":
                st = "C" if cohab is True else "D" if cohab is False else "U"
            elif state == "NR":
                st = "E"
            else:
                raise ValueError(state)
            e = ty - t0
            rows.append(dict(caseid=cid, wave=w, cmonth=int(round((ty - 1980) * 12 + 0.5)), state=st,
                             same_partner=1.0 if (w > 1 and st != "E") else (0.0 if st == "E" else np.nan),
                             age=age0 + e, income=inc1[cid] if w == 1 else np.nan,
                             rel_quality=qual1[cid] if w == 1 else np.nan,
                             rel_dur_yrs=dur0 + e if (st != "E" and np.isfinite(dur0)) else np.nan,
                             end_type=nrt if st == "E" else ""))
            if st == "E":
                break
    p = pd.DataFrame(rows).join(fixed, on="caseid")

    dead1 = p[(p.wave == 1) & (p.state == "E")].caseid
    p = p[~p.caseid.isin(dead1)].reset_index(drop=True)
    assert p.caseid.nunique() == 2981
    bad = p.age - p.rel_dur_yrs < 12
    p.loc[bad, "rel_dur_yrs"] = np.nan
    p["cohort"] = "HCMST2009"
    p.to_csv(CACHE / "hcmst2009_panel.csv", index=False)
    print(p.shape, "invalid durations set missing:", int(bad.sum()))
    print(pd.crosstab(p.wave, p.state).to_string())
    print(p.end_type.value_counts().to_dict())

def step_hcmst2017():

    from pathlib import Path
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    RAW = ROOT / "raw_data/HCMST2017/raw/hcmst2017_2022_v2.2.dta"
    CACHE = ROOT / "cache"
    d = pd.read_stata(RAW)
    assert len(d) == 3510 and d.caseid_new.is_unique
    s = lambda c: d[c].astype(str).where(d[c].notna())

    def cm(ts):
        ts = pd.to_datetime(ts)
        return (ts.dt.year - 1980) * 12 + ts.dt.month

    W1 = {"married": "M", "in unmarried cohab partnership": "C", "in unmarried noncohab partnership": "D", "unpartnered": "S"}
    st1 = s("w1_partnership_status_cohab").map(W1)
    st2 = s("w2_partner_type_cohab").map(W1)
    pt3, live3 = s("w3_partner_type"), s("w3_live_w_partner")
    st3 = pd.Series(np.select([pt3 == "married", (pt3 == "in unmarried partnership") & (live3 == "yes"),
                               (pt3 == "in unmarried partnership") & (live3 == "no"), pt3 == "unpartnered"],
                              ["M", "C", "D", "S"], ""), index=d.index).replace("", np.nan)
    newmar3 = d.w3_when_married_yr.notna()
    assert (st3[newmar3].isin(["C", "D"])).all()
    st3[newmar3] = "M"

    sec2, sec3 = s("w2_section"), s("w3_section")
    same2 = pd.Series(np.nan, index=d.index)
    same2[sec2.str.startswith(("A,", "C,"), na=False)] = 1
    same2[sec2.str.startswith(("B,", "no longer"), na=False) & st2.isin(["D", "C", "M"])] = 0
    same3 = pd.Series(np.nan, index=d.index)
    same3[sec3.str.startswith(("A,", "C,"), na=False)] = 1
    same3[sec3.str.startswith(("B,", "D,"), na=False) & st3.isin(["D", "C", "M"])] = 0

    def num(c):
        return pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)

    age1 = pd.to_numeric(d.w1_ppage.astype(str), errors="coerce")
    EDU = {"No formal education": 0, "1st, 2nd, 3rd, or 4th grade": 3, "5th or 6th grade": 6, "7th or 8th grade": 8,
           "9th grade": 9, "10th grade": 10, "11th grade": 11, "12th grade NO DIPLOMA": 12,
           "HIGH SCHOOL GRADUATE - high school DIPLOMA or the equivalent (GED)": 12, "Some college, no degree": 13,
           "Associate degree": 14, "Bachelors degree": 16, "Masters degree": 18, "Professional or Doctorate degree": 20}
    edu = s("w1_ppeduc").map(EDU)
    assert edu.notna().sum() == s("w1_ppeduc").notna().sum(), set(s("w1_ppeduc").dropna()) - set(EDU)

    def inc_mid(col):
        v = s(col)
        lo_hi = v.str.replace(",", "").str.extract(r"\$(\d+)\s*to\s*\$(\d+)").astype(float)
        mid = lo_hi.mean(axis=1)
        mid[v == "Less than $5,000"] = 2500
        mid[v.str.contains("or more", na=False)] = 250000 * 1.25
        return mid

    QUAL = {"Very Poor": 1, "Poor": 2, "Fair": 3, "Good": 4, "Excellent": 5}
    eth = s("w1_ppethm")
    fixed = pd.DataFrame({
        "female": (s("w1_ppgender") == "Female").astype(float),
        "black": (eth == "Black, Non-Hispanic").astype(float),
        "hispanic": (eth == "Hispanic").astype(float),
        "edu_years": edu,
        "same_sex": (s("w1_same_sex_couple") == "same_sex_couple").astype(float).where(s("w1_same_sex_couple").notna()),
        "met_online": (s("w1_q24_met_online") == "yes").astype(float).where(s("w1_q24_met_online").notna()),
        "weight": num("w1_weight_combo"),
    })
    rows = []
    dates = {1: pd.Series((2017 - 1980) * 12 + 7, index=d.index), 2: cm(d.w2_month_of_survey), 3: cm(d.w3_month_interview)}
    states = {1: st1, 2: st2, 3: st3}
    same = {1: pd.Series(np.nan, index=d.index), 2: same2, 3: same3}
    inc = {1: inc_mid("w1_ppincimp"), 2: inc_mid("w2_ppincimp"), 3: inc_mid("w3_ppincimp")}
    qual = {1: s("w1_q34").map(QUAL), 2: s("w2_rel_qual_combo").map(QUAL), 3: s("w3_rel_qual").map(QUAL)}
    rel_start = {1: num("w1_relationship_start_calmonth"), 2: pd.Series(np.nan, index=d.index), 3: num("w3_month_rel_started")}
    reldur = {1: num("w1_relate_duration_in2017_years"), 2: num("w2_relationship_duration"), 3: num("w3_relationship_duration_yrs")}
    for w in (1, 2, 3):
        ok = states[w].notna() & dates[w].notna()
        for i in np.where(ok)[0]:
            rows.append(dict(caseid=int(d.caseid_new.iat[i]), wave=w, cmonth=int(dates[w].iat[i]), state=states[w].iat[i],
                             same_partner=same[w].iat[i], age=age1.iat[i] + (dates[w].iat[i] - dates[1].iat[i]) / 12,
                             income=inc[w].iat[i], rel_quality=qual[w].iat[i] if states[w].iat[i] != "S" else np.nan,
                             rel_dur_yrs=reldur[w].iat[i] if states[w].iat[i] != "S" else np.nan))
    p = pd.DataFrame(rows).join(fixed.set_index(d.caseid_new.astype(int)), on="caseid")

    bad = (p.rel_dur_yrs < 0) | (p.age - p.rel_dur_yrs < 12)
    p.loc[bad, "rel_dur_yrs"] = np.nan

    p = p.sort_values(["caseid", "wave"]).reset_index(drop=True)
    prev = p.groupby("caseid").state.shift(1)
    p.loc[prev.isna() | (prev == "S"), "same_partner"] = np.nan
    p["cohort"] = "HCMST2017"
    p.to_csv(CACHE / "hcmst2017_panel.csv", index=False)
    print(p.shape)
    print(pd.crosstab(p.wave, p.state).to_string())
    g = p.sort_values(["caseid", "wave"]).groupby("caseid")
    p = p.sort_values(["caseid", "wave"])
    p["prev"] = g.state.shift(1)
    t = p[p.prev.notna()]
    print(pd.crosstab([t.prev], [t.state, t.same_partner.fillna(-1)]).to_string())
    print(p[["age", "income", "rel_quality", "rel_dur_yrs", "edu_years"]].describe().T.round(2).to_string())

if __name__ == "__main__":
    import sys
    for s in (step_hcmst2009, step_hcmst2017):
        if len(sys.argv) == 1 or s.__name__ in sys.argv[1:]:
            s()
