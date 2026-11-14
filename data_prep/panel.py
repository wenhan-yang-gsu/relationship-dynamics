# Harmonize NLSY97, HCMST 2009 and HCMST 2017 into one panel and add covariates, households and the split.
def step_harmonize():

    from pathlib import Path
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    CACHE = ROOT / "cache"
    MONTH = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                         "September", "October", "November", "December"], 1)}

    n = pd.read_csv(CACHE / "nlsy97_panel_cov.csv")
    n = n[n.state.notna()].copy()
    n["dur"] = np.where(n.state == "D", n.dur_dating_yrs, np.where(n.state.isin(["C", "M"]), n.dur_state_yrs, np.nan))
    x = pd.read_csv(CACHE / "nlsy97_extract.csv", usecols=lambda c: c.startswith(("PUBID", "CV_INCOME_FAMILY", "CV_INCOME_GROSS_YR")))
    x = x.set_index("PUBID|1997")
    inc = np.full(len(n), np.nan)
    for y in n.year.unique():
        for name in ("CV_INCOME_FAMILY", "CV_INCOME_GROSS_YR"):
            c = f"{name}|{y}"
            if c in x:
                m = (n.year == y).to_numpy()
                v = x[c].reindex(n.pubid[m]).to_numpy().astype(float)
                inc[m] = np.where(np.isnan(inc[m]) & (v >= 0), v, inc[m])
    N = pd.DataFrame(dict(dataset="NLSY97", cohort="NLSY97", pid="N" + n.pubid.astype(str), wave=n["round"],
                          t_years=(n.cmonth - 0.5) / 12, state=n.state, same_partner=n.same_partner,
                          dur_state_yrs=n.dur, age=n.age, female=n.female, black=n.black, hispanic=n.hispanic,
                          edu_years=n.edu_years, log_income=np.log1p(inc), kids_any=(n.kids > 0).astype(float).where(n.kids.notna()),
                          rel_quality=np.nan, weight=n.weight_panel))

    h9 = pd.read_csv(CACHE / "hcmst2009_panel.csv").sort_values(["caseid", "wave"])
    raw9 = pd.read_stata(ROOT / "raw_data/HCMST/raw/HCMST_ver_3.04.dta", convert_categoricals=False).set_index("caseid_new")
    def age_at(col):
        v = pd.to_numeric(raw9[col], errors="coerce")
        return v.where(v > 0)
    a0 = pd.to_numeric(raw9.ppage, errors="coerce")
    d9 = {"D": a0 - age_at("q21b"), "C": a0 - age_at("q21c"), "M": a0 - age_at("q21d")}

    h7 = pd.read_csv(CACHE / "hcmst2017_panel.csv").sort_values(["caseid", "wave"])
    raw7 = pd.read_stata(ROOT / "raw_data/HCMST2017/raw/hcmst2017_2022_v2.2.dta").set_index("caseid_new")
    def ym(ycol, mcol):
        y = pd.to_numeric(raw7[ycol], errors="coerce")
        m = raw7[mcol]
        m = pd.to_numeric(m, errors="coerce") if m.dtype.kind in "fi" else m.astype(str).map(MONTH)
        return y + (m.fillna(6.5) - 0.5) / 12
    ev7 = {1: {"D": ym("w1_q21b_year", "w1_q21b_month"), "C": ym("w1_q21c_year", "w1_q21c_month"), "M": ym("w1_q21d_year", "w1_q21d_month")},
           2: {"D": ym("w2_q21b_year", "w2_q21b_month"), "C": ym("w2_q21c_year", "w2_q21c_month"), "M": ym("w2_q21d_year", "w2_q21d_month")},
           3: {"D": ym("w3_Q21B_year", "w3_Q21B_month"), "C": ym("w3_Q21C_year", "w3_Q21C_month"), "M": ym("w3_Q21D_year", "w3_Q21D_month")}}
    moved7 = {2: pd.to_numeric(raw7.get("w2_move_in_yr"), errors="coerce"), 3: pd.to_numeric(raw7.get("w3_move_in_yr"), errors="coerce")}
    mar7 = {3: ym("w3_when_married_yr", "w3_when_married_month")}

    def hc_durations(h, cohort):
        out = np.full(len(h), np.nan)
        t = (h.cmonth.to_numpy() - 0.5) / 12 + 1980
        prev = {}
        for k, (cid, w, st, same) in enumerate(zip(h.caseid, h.wave, h.state, h.same_partner)):
            val = np.nan
            if st in ("D", "C", "M"):
                if w == 1:
                    if cohort == "HCMST2009":
                        val = d9[st].get(cid, np.nan)
                    else:
                        val = t[k] - ev7[1][st].get(cid, np.nan)
                elif cid in prev and same == 1:
                    pst, pdur, pt = prev[cid]
                    dt = t[k] - pt
                    if st == pst:
                        val = pdur + dt if np.isfinite(pdur) else np.nan
                    else:
                        ev = np.nan
                        if cohort == "HCMST2017" and st == "C" and w in moved7:
                            ev = moved7[w].get(cid, np.nan) + 0.5
                        if cohort == "HCMST2017" and st == "M" and w in mar7:
                            ev = mar7[w].get(cid, np.nan)
                        val = t[k] - ev if np.isfinite(ev) and pt <= ev <= t[k] else dt / 2
                elif cohort == "HCMST2017" and w in (2, 3):
                    val = t[k] - ev7[w][st].get(cid, np.nan)
            if np.isfinite(val) and val < 0:
                val = np.nan
            out[k] = val
            prev[cid] = (st, val, t[k])
        return out

    H = []
    for h, cohort in ((h9, "HCMST2009"), (h7, "HCMST2017")):
        h = h.reset_index(drop=True)
        dur = hc_durations(h, cohort)
        dur = np.where(h.age.to_numpy() - dur < 12, np.nan, dur)
        H.append(pd.DataFrame(dict(dataset="HCMST", cohort=cohort, pid=("H9_" if cohort == "HCMST2009" else "H7_") + h.caseid.astype(str),
                                   wave=h.wave, t_years=(h.cmonth - 0.5) / 12, state=h.state, same_partner=h.same_partner,
                                   dur_state_yrs=dur, age=h.age, female=h.female, black=h.black, hispanic=h.hispanic,
                                   edu_years=h.edu_years, log_income=np.log1p(h.income), kids_any=np.nan,
                                   rel_quality=h.rel_quality, weight=h.weight)))
    A = pd.concat([N] + H, ignore_index=True).sort_values(["pid", "t_years"]).reset_index(drop=True)

    for c in ("log_income", "rel_quality", "edu_years"):
        A[c] = A.groupby("pid")[c].ffill()
    A.loc[A.state.isin(["S", "N", "E"]), "rel_quality"] = np.nan
    A.to_csv(CACHE / "panel_all.csv", index=False)
    print(A.groupby("cohort").agg(persons=("pid", "nunique"), rows=("pid", "size")).to_string())
    print(pd.crosstab(A.cohort, A.state).to_string())
    print(A.groupby(["cohort", "state"]).dur_state_yrs.agg(["count", "median", "max"]).round(2).loc[lambda d: d["count"] > 0].to_string())
    miss = A.groupby("cohort")[["age", "female", "black", "hispanic", "edu_years", "log_income", "rel_quality", "weight"]].apply(lambda d: d.isna().mean().round(3))
    print(miss.to_string())

def step_augment():

    from pathlib import Path
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    CACHE = ROOT / "cache"
    SEED = 20261110
    A = pd.read_csv(CACHE / "panel_all.csv")
    n = pd.read_csv(CACHE / "nlsy97_panel_cov.csv")
    x = pd.read_csv(CACHE / "nlsy97_extract.csv",
                    usecols=lambda c: c.startswith(("PUBID", "YSAQ2-298D18", "YSAQ2-298D20", "YSAQ2-298D21",
                                                    "YSAQ2-292A2", "YSAQ2-292A4", "YSAQ2-292A5", "MAR_STATUS_"))).set_index("PUBID|1997")
    design = pd.read_csv(CACHE / "nlsy97_design.csv").set_index("pubid")

    isN = A.cohort == "NLSY97"
    A["pubid"] = np.where(isN, A.pid.str[1:].where(isN, "0"), "0").astype(int)
    key = n.set_index(["pubid", "round"])
    traits = ["afqt", "mom_edu", "both_parents12", "parent_relig", "tipi_extra", "tipi_agree", "tipi_consc",
              "tipi_stable", "tipi_open"]
    idx = pd.MultiIndex.from_arrays([A.pubid[isN], A.wave[isN]])
    for c in traits + ["n_unions_before", "kids", "year", "cmonth"]:
        A.loc[isN, c] = key[c].reindex(idx).to_numpy()
    A.loc[isN, "prior_union"] = (A.loc[isN, "n_unions_before"] > 0).astype(float).where(A.loc[isN, "n_unions_before"].notna())

    def feel(row_year, pub, state):
        out = [np.nan] * 3
        if 2002 <= row_year <= 2005 and state == "D":
            cols = [f"YSAQ2-298D18|{row_year}", f"YSAQ2-298D20|{row_year}", f"YSAQ2-298D21|{row_year}"]
        elif row_year == 2008 and state in ("C", "M"):
            cols = ["YSAQ2-292A2|2008", "YSAQ2-292A4|2008", "YSAQ2-292A5|2008"]
        else:
            return out
        v = [x.at[pub, c] for c in cols]
        return [float(a) if a >= 0 else np.nan for a in v]

    F = np.array([feel(int(y), p, s) if not np.isnan(y) else [np.nan] * 3
                  for y, p, s in zip(A.loc[isN, "year"], A.loc[isN, "pubid"], A.loc[isN, "state"])])
    A.loc[isN, ["rel_close", "rel_conflict", "rel_future"]] = F

    ms_cols = [c for c in x if c.startswith("MAR_STATUS_")]
    months = np.array([(int(c[11:15]) - 1980) * 12 + int(c[16:18]) for c in ms_cols])
    MS = x[ms_cols]
    A = A.sort_values(["pid", "t_years"]).reset_index(drop=True)
    isN = A.cohort == "NLSY97"
    g = A.groupby("pid")
    A["prev_state"] = g.state.shift(1)
    A["prev_cmonth"] = g.cmonth.shift(1)
    LEVEL = {"S": 0, "N": 0, "D": 1, "U": 1.5, "C": 2, "M": 3, "E": -1}
    ended = A.prev_state.isin(["D", "C", "M", "U"]) & (
        A.state.isin(["S", "N", "E"]) | (A.state.map(LEVEL) < A.prev_state.map(LEVEL)) | (A.same_partner == 0))
    A["end_tag"] = ""
    sel = isN & ended & (A.prev_state == "M")
    for i in np.where(sel)[0]:
        row = MS.loc[A.pubid.iat[i]].to_numpy()
        m = (months > A.prev_cmonth.iat[i]) & (months <= A.cmonth.iat[i])
        A.at[i, "end_tag"] = "X" if (row[m] == 5).any() else "B"

    raw9 = pd.read_stata(ROOT / "raw_data/HCMST/raw/HCMST_ver_3.04.dta").set_index("caseid_new")
    raw7 = pd.read_stata(ROOT / "raw_data/HCMST2017/raw/hcmst2017_2022_v2.2.dta").set_index("caseid_new")
    h9 = pd.read_csv(CACHE / "hcmst2009_panel.csv").set_index(["caseid", "wave"])
    is9, is7 = A.cohort == "HCMST2009", A.cohort == "HCMST2017"
    cid = A.pid.str.split("_").str[1].where(~isN, "0").astype(int)

    et = h9.end_type.reindex(pd.MultiIndex.from_arrays([cid[is9], A.wave[is9]])).to_numpy()
    A.loc[is9 & (A.state == "E"), "end_tag"] = np.where(et[(A.state[is9] == "E").to_numpy()] == "partner_deceased", "X", "B")
    TIMES = {"once (this is my first marriage)": 1, "twice": 2, "three times": 3, "four or more times": 4,
             "never married": 0, "once": 1}
    q17a = raw9.q17a.astype(str).map(TIMES); q17b = raw9.q17b.astype(str).map(TIMES)
    prior9 = pd.Series(np.where(q17a.notna(), (q17a > 1).astype(float), np.where(q17b.notna(), (q17b >= 1).astype(float), np.nan)), index=raw9.index)
    A.loc[is9, "prior_union"] = prior9.reindex(cid[is9]).to_numpy()
    kids9 = (pd.to_numeric(raw9.children_in_hh, errors="coerce") > 0).astype(float)
    A.loc[is9 & (A.wave == 1), "kids_any"] = kids9.reindex(cid[is9 & (A.wave == 1)]).to_numpy()

    for w, col in ((2, "w2_relationship_end"), (3, "w3_relationship_end_combo")):
        m = is7 & (A.wave == w) & ended
        died = raw7[col].astype(str).reindex(cid[m]).to_numpy() == "Partner died"
        A.loc[m, "end_tag"] = np.where(died, "X", "B")
        ppt = sum(pd.to_numeric(raw7[f"w{w}_{c}"], errors="coerce") for c in ("PPT01", "PPT25", "PPT612", "PPT1317"))
        A.loc[is7 & (A.wave == w), "kids_any"] = (ppt > 0).astype(float).reindex(cid[is7 & (A.wave == w)]).to_numpy()
    ppt1 = sum(pd.to_numeric(raw7[f"w1_{c}"].astype(str), errors="coerce") for c in ("PPT01", "PPT25", "PPT612", "PPT1317"))
    A.loc[is7 & (A.wave == 1), "kids_any"] = (ppt1 > 0).astype(float).reindex(cid[is7 & (A.wave == 1)]).to_numpy()
    tm = pd.to_numeric(raw7.w1_q17, errors="coerce")
    cur_m = raw7.w1_partnership_status_cohab.astype(str) == "married"
    prior7 = pd.Series(np.where(tm.isna(), np.nan, np.where(cur_m, (tm > 1).astype(float), (tm >= 1).astype(float))), index=raw7.index)
    A.loc[is7, "prior_union"] = prior7.reindex(cid[is7]).to_numpy()
    for c in ("kids_any", "prior_union"):
        A[c] = A.groupby("pid")[c].ffill()

    A["household"] = np.where(isN, "NH" + A.pubid.map(design.sidcode).astype("Int64").astype(str), A.pid)
    rng = np.random.default_rng(SEED)
    assign = {}
    for src, d in A.groupby("cohort"):
        hh = np.array(sorted(d.household.unique()))
        u = rng.random(len(hh))
        assign.update(dict(zip(hh, np.where(u < 0.70, "train", np.where(u < 0.85, "valid", "test")))))
    A["split"] = A.household.map(assign)
    A = A.drop(columns=["prev_state", "prev_cmonth"])
    A.to_csv(CACHE / "panel.csv", index=False)

    pd.set_option("display.width", 200)
    print(A.groupby(["cohort", "split"]).agg(persons=("pid", "nunique"), households=("household", "nunique"), rows=("pid", "size")).to_string())
    print(pd.crosstab(A.cohort, A.end_tag.replace("", "none")).to_string())
    cols = ["prior_union", "kids_any", "rel_close", "rel_conflict", "rel_future", "afqt", "tipi_extra"]
    print(A.groupby("cohort")[cols].apply(lambda d: d.notna().mean().round(3)).to_string())
    print(A[["rel_close", "rel_conflict", "rel_future"]].describe().round(2).to_string())
    assert A.groupby("pid").split.nunique().max() == 1 and A.groupby("household").split.nunique().max() == 1

if __name__ == "__main__":
    import sys
    steps = [step_harmonize, step_augment]
    for s in steps:
        if len(sys.argv) == 1 or s.__name__ in sys.argv[1:]:
            s()
