# NLSY97: variable index, extraction, codebook checks, the relationship-state panel and the covariates.
def step_nlsy_index():

    from pathlib import Path
    import pandas as pd

    RAW = Path(__file__).resolve().parents[1] / "raw_data/NLSY97/raw"
    OUT = Path(__file__).resolve().parents[1] / "cache/nlsy97_index.csv"

    lines = (RAW / "nlsy97_all_1997-2023.sdf").read_text(encoding="utf-8", errors="replace").splitlines()
    h = next(i for i, ln in enumerate(lines) if ln.startswith("Number") and "Question Name" in ln)
    hdr = lines[h]
    c_year, c_desc, c_q = hdr.index("Year"), hdr.index("Variable Description"), hdr.index("Question Name")
    rows = []
    for ln in lines[h + 2:]:
        if not ln.strip():
            continue
        rows.append(dict(ref=ln[:c_year].strip(), year=ln[c_year:c_desc].strip(),
                         desc=ln[c_desc:c_q].strip(), qname=ln[c_q:].strip()))
    ix = pd.DataFrame(rows)
    ix["col"] = ix.ref.str.replace(".", "", regex=False)
    ix.to_csv(OUT, index=False)
    print(len(ix), "variables;", ix.year.value_counts().sort_index().to_dict())

def step_nlsy_extract():

    from pathlib import Path
    import re
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    RAW = ROOT / "raw_data/NLSY97/raw/nlsy97_all_1997-2023.csv"
    CACHE = ROOT / "cache"
    ix = pd.read_csv(CACHE / "nlsy97_index.csv", dtype=str)

    DATING_YEARS = [str(y) for y in range(2002, 2012)]

    EXACT = [
        "PUBID", "KEY!SEX", "KEY!BDATE_M", "KEY!BDATE_Y", "KEY!RACE_ETHNICITY", "CV_SAMPLE_TYPE",
        "CV_INTERVIEW_CMONTH", "CV_INTERVIEW_DATE_M", "CV_INTERVIEW_DATE_Y",
        "CV_INTERVIEW_DATE~M", "CV_INTERVIEW_DATE~Y", "RNI",
        "CV_HGC_EVER_EDT", "CV_HIGHEST_DEGREE_EVER_EDT", "CV_ENROLLSTAT",
        "CV_INCOME_FAMILY", "CV_INCOME_GROSS_YR", "CV_HH_POV_RATIO", "CV_HH_SIZE",
        "CV_BIO_CHILD_HH", "CV_BIO_CHILD_NR", "CV_BIO_CHILD_HH_U18", "CV_BIO_CHILD_NR_U18",
        "CV_URBAN-RURAL", "CV_CENSUS_REGION", "CV_MARSTAT_COLLAPSED",
        "CV_HGC_BIO_MOM", "CV_HGC_BIO_DAD", "CV_HGC_RES_MOM", "CV_HGC_RES_DAD",
        "CV_YTH_REL_HH_AGE_12", "FP_ADHRISKI", "FP_PPRELIG",
        "ASVAB_MATH_VERBAL_SCORE_PCT", "SAMPLING_WEIGHT_CC", "SAMPLING_PANEL_WEIGHT",
    ]
    PATTERNS = [
        r"^MAR_STATUS_\d{4}\.\d{2}$", r"^MAR_COHABITATION_\d{4}\.\d{2}$", r"^MAR_PARTNER_LINK_\d{4}\.\d{2}$",
        r"^YTEL-TIPIA~0000\d\d$",
    ]
    DATING = [
        r"^YSAQ2-298D1$", r"^YSAQ2-298D4~[MY]$", r"^YSAQ2-298D5A$", r"^YSAQ2-298D6$", r"^YSAQ2-298D7~[MY]$",
        r"^YSAQ2-298D5B$", r"^YSAQ2-298D5C$",
        r"^YSAQ2-298D18$", r"^YSAQ2-298D19$", r"^YSAQ2-298D20$", r"^YSAQ2-298D21$",
        r"^YSAQ2-701$", r"^YSAQ2-707$", r"^YSAQ2-709~[MY]$", r"^YSAQ2-711$", r"^YSAQ2-713$", r"^YSAQ2-715$",
        r"^YSAQ2-291A$", r"^YSAQ2-294$", r"^YSAQ2-298A1$",
        r"^YSAQ2-292A[2-5]$",
    ]

    sel = ix.qname.isin(EXACT)
    for p in PATTERNS:
        sel |= ix.qname.str.match(p, na=False)
    dat = pd.Series(False, index=ix.index)
    for p in DATING:
        dat |= ix.qname.str.match(p, na=False)
    sel |= dat & ix.year.isin(DATING_YEARS)
    vars_ = ix[sel].copy()
    vars_["name"] = vars_.qname + "|" + vars_.year
    assert vars_.name.is_unique, vars_.name[vars_.name.duplicated()].tolist()

    with open(RAW) as f:
        header = f.readline().rstrip("\n").replace('"', "").split(",")
        assert header == ix.col.tolist(), "csv header does not match the index"
        pos = [header.index(c) for c in vars_.col]
        rows = []
        for ln in f:
            v = ln.rstrip("\n").split(",")
            assert len(v) == len(header)
            rows.append([v[i] for i in pos])

    out = pd.DataFrame(rows, columns=vars_.name.tolist()).apply(pd.to_numeric)
    assert len(out) == 8984 and out["PUBID|1997"].is_unique
    out.to_csv(CACHE / "nlsy97_extract.csv", index=False)
    vars_[["ref", "qname", "year", "desc", "name"]].to_csv(CACHE / "nlsy97_extract_vars.csv", index=False)
    print(out.shape)
    print(vars_.assign(fam=vars_.qname.str.replace(r"[_.~]\d.*$|\d{4}\.\d{2}$", "", regex=True)).groupby("fam").size().to_string())

def step_nlsy_verify():

    from pathlib import Path
    import re
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    CDB = ROOT / "raw_data/NLSY97/raw/nlsy97_all_1997-2023.cdb"
    CACHE = ROOT / "cache"
    x = pd.read_csv(CACHE / "nlsy97_extract.csv")
    v = pd.read_csv(CACHE / "nlsy97_extract_vars.csv", dtype=str)
    want = dict(zip(v.ref, v.name))

    blocks, cur, buf = {}, None, []
    head = re.compile(r"^([A-Z]\d{5}\.\d{2})\s+\[")
    with open(CDB, encoding="latin-1") as f:
        for ln in f:
            m = head.match(ln)
            if m:
                if cur in want:
                    blocks[cur] = "".join(buf)
                cur, buf = m.group(1), []
            elif cur in want:
                buf.append(ln)
        if cur in want:
            blocks[cur] = "".join(buf)

    def num(pat, txt):
        m = re.search(pat, txt)
        return int(m.group(1)) if m else 0

    bad = []
    for ref, name in want.items():
        t = blocks[ref]
        exp = {-1: num(r"Refusal\(-1\)\s+(\d+)", t), -2: num(r"Don't Know\(-2\)\s+(\d+)", t),
               -3: num(r"Invalid Skip\(-3\)\s+(\d+)", t), -4: num(r"VALID SKIP\(-4\)\s+(\d+)", t),
               -5: num(r"NON-INTERVIEW\(-5\)\s+(\d+)", t)}
        tot = num(r"TOTAL =+>\s+(\d+)", t)
        col = x[name]
        got = {k: int((col == k).sum()) for k in exp}
        got_tot = int((~col.isin([-4, -5])).sum())
        if got != exp or got_tot != tot:
            bad.append((name, exp, got, tot, got_tot))
    print(f"checked {len(want)} variables; mismatches: {len(bad)}")
    for b in bad[:20]:
        print(b)

def step_nlsy_build():

    from pathlib import Path
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    CACHE = ROOT / "cache"
    x = pd.read_csv(CACHE / "nlsy97_extract.csv")
    YEARS = [1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
             2013, 2015, 2017, 2019, 2021, 2023]
    DATING_YEARS = range(2002, 2012)

    ms = [c for c in x if c.startswith("MAR_STATUS_")]
    mc = [c for c in x if c.startswith("MAR_COHABITATION_")]
    cmonth = lambda c: (int(c.split("_")[-1][:4]) - 1980) * 12 + int(c.split("_")[-1][5:7])
    months = np.array([cmonth(c) for c in ms])
    assert (np.diff(months) == 1).all()
    S = x[ms].to_numpy()
    C = pd.DataFrame(x[mc].to_numpy(), columns=[cmonth(c) for c in mc]).reindex(columns=months).to_numpy()

    partner = (C >= 101) & (C <= 111)
    spouse = (C >= 201) & (C <= 210)
    alone = np.isnan(C) | (C == -4)
    arr = np.full(S.shape, "", dtype="<U1")
    arr[(S == 2) & partner] = "C"
    arr[(S == 2) & ~partner] = "M"
    arr[S == 1] = "C"
    other = np.isin(S, [0, 3, 4, 5])
    arr[other & (partner | spouse)] = "C"

    arr[np.isin(S, [3, 4]) & spouse] = "M"
    arr[other & alone] = "N"
    arr[(S < 0) | (C == -3)] = ""

    pid = np.where(partner | spouse, C, np.nan)

    ml = [c for c in x if c.startswith("MAR_PARTNER_LINK_")]
    LK = pd.DataFrame(x[ml].to_numpy(), columns=[cmonth(c) for c in ml]).reindex(columns=months).to_numpy()
    LK = np.where(LK > 0, LK, np.nan)

    np.savez_compressed(CACHE / "nlsy97_monthly.npz", pubid=x["PUBID|1997"].to_numpy(), months=months,
                        state=arr, partner=pid)

    rows = []
    for r, y in enumerate(YEARS, 1):
        cm = x[f"CV_INTERVIEW_CMONTH|{y}"].to_numpy()
        for i in np.where(cm > 0)[0]:
            j = cm[i] - months[0]
            a = arr[i, j] if 0 <= j < len(months) else ""
            st = a
            dat = np.nan
            if y in DATING_YEARS and a == "N":
                q707 = x[f"YSAQ2-707|{y}"].iat[i] if f"YSAQ2-707|{y}" in x else -4
                d1, d5 = x[f"YSAQ2-298D1|{y}"].iat[i], x[f"YSAQ2-298D5A|{y}"].iat[i]
                if q707 == 1 or d5 == 1:
                    st, dat = "D", 1.0
                elif (d1 == 0 or d5 == 4) and q707 not in (1, -1, -2) and d5 not in (-1, -2) and d1 not in (-1, -2):
                    st, dat = "S", 0.0
            rows.append(dict(pubid=x["PUBID|1997"].iat[i], round=r, year=y, cmonth=int(cm[i]),
                             arr_state=a or None, state=st or None, dating=dat,
                             union_partner=pid[i, j] if 0 <= j < len(months) else np.nan,
                             union_link=LK[i, j] if 0 <= j < len(months) else np.nan))
    p = pd.DataFrame(rows)
    bm, by = x["KEY!BDATE_M|1997"].to_numpy(), x["KEY!BDATE_Y|1997"].to_numpy()
    born = dict(zip(x["PUBID|1997"], (by - 1980) * 12 + bm))
    p["age"] = (p.cmonth - p.pubid.map(born)) / 12

    p = p.sort_values(["pubid", "round"]).reset_index(drop=True)
    prev_state = p.groupby("pubid").state.shift(1)
    prev_part = p.groupby("pubid").union_partner.shift(1)
    same = pd.Series(np.nan, index=p.index)
    v707 = np.full(len(p), np.nan)
    pos = dict(zip(x["PUBID|1997"], range(len(x))))
    for k, (pid_, y) in enumerate(zip(p.pubid, p.year)):
        c = f"YSAQ2-707|{y}"
        if c in x:
            v707[k] = x[c].iat[pos[pid_]]
    p["q707"] = v707
    fromD = prev_state == "D"
    same[fromD & p.q707.isin([1, 2, 3])] = 1
    same[fromD & (p.q707 == 4)] = 0
    union = prev_state.isin(["C", "M"]) & p.state.isin(["C", "M"]) & prev_part.notna() & p.union_partner.notna()
    kind_same = (prev_part // 100) == (p.union_partner // 100)
    same[union & kind_same & (prev_part == p.union_partner)] = 1
    same[union & kind_same & (prev_part != p.union_partner)] = 0

    prev_link = p.groupby("pubid").union_link.shift(1)
    both = prev_state.isin(["C", "M"]) & p.state.isin(["C", "M"]) & prev_link.notna() & p.union_link.notna()
    same[both] = (prev_link[both] == p.union_link[both]).astype(float)
    p["same_partner"] = same

    n_recoded = 0
    while True:
        prev = p.groupby("pubid").state.shift(1)
        fix = (prev == "M") & (p.state == "C") & (p.same_partner == 1)
        if not fix.any():
            break
        n_recoded += int(fix.sum())
        p.loc[fix, "state"] = "M"
        p.loc[fix, "arr_state"] = "M"
    print("married -> same-partner cohabiting recoded as continuing marriage:", n_recoded)
    p.to_csv(CACHE / "nlsy97_panel.csv", index=False)
    print(p.shape)
    print(pd.crosstab(p.year, p.state.fillna("missing")).to_string())

def step_nlsy_covariates():

    from pathlib import Path
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    CACHE = ROOT / "cache"
    x = pd.read_csv(CACHE / "nlsy97_extract.csv").set_index("PUBID|1997")
    p = pd.read_csv(CACHE / "nlsy97_panel.csv")
    mo = np.load(CACHE / "nlsy97_monthly.npz")
    months, arr, part = mo["months"], mo["state"], mo["partner"]
    row = {pid: i for i, pid in enumerate(mo["pubid"])}
    raw_status = x[[c for c in x if c.startswith("MAR_STATUS_")]].to_numpy()

    def nonneg(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.where(s >= 0)

    F = pd.DataFrame(index=x.index)
    F["female"] = (x["KEY!SEX|1997"] == 2).astype(float)
    F["black"] = (x["KEY!RACE_ETHNICITY|1997"] == 1).astype(float)
    F["hispanic"] = (x["KEY!RACE_ETHNICITY|1997"] == 2).astype(float)
    F["birth_year"] = x["KEY!BDATE_Y|1997"]
    F["afqt"] = nonneg(x["ASVAB_MATH_VERBAL_SCORE_PCT|XRND"]) / 1000
    mom = nonneg(x["CV_HGC_BIO_MOM|1997"]).where(lambda s: s != 95)
    F["mom_edu"] = mom.fillna(nonneg(x["CV_HGC_RES_MOM|1997"]).where(lambda s: s != 95))
    rel12 = nonneg(x["CV_YTH_REL_HH_AGE_12|1997"])
    F["both_parents12"] = (rel12 == 1).astype(float).where(rel12.notna())
    F["fam_risk"] = nonneg(x["FP_ADHRISKI|1997"]) / 100
    F["parent_relig"] = nonneg(x["FP_PPRELIG|1997"]) / 100
    F["cross_sectional"] = x["CV_SAMPLE_TYPE|1997"].astype(float)
    t = {k: nonneg(x[f"YTEL-TIPIA~{k:06d}|2008"]) for k in range(1, 11)}
    F["tipi_extra"] = (t[1] + 8 - t[6]) / 2
    F["tipi_agree"] = (8 - t[2] + t[7]) / 2
    F["tipi_consc"] = (t[3] + 8 - t[8]) / 2
    F["tipi_stable"] = (8 - t[4] + t[9]) / 2
    F["tipi_open"] = (t[5] + 8 - t[10]) / 2
    p = p.join(F, on="pubid")

    def at_round(name, transform=lambda s: s):
        out = np.full(len(p), np.nan)
        for y in p.year.unique():
            col = f"{name}|{y}"
            if col in x:
                m = (p.year == y).to_numpy()
                out[m] = transform(nonneg(x[col].reindex(p.pubid[m]))).to_numpy()
        return out

    p["edu_years"] = at_round("CV_HGC_EVER_EDT", lambda s: s.where(s != 95))
    p["degree"] = at_round("CV_HIGHEST_DEGREE_EVER_EDT")
    enr = at_round("CV_ENROLLSTAT")
    p["enrolled"] = np.where(np.isnan(enr), np.nan, (enr >= 8).astype(float))
    p["pov_ratio"] = at_round("CV_HH_POV_RATIO") / 100

    def at_round_kids(name):
        out = np.full(len(p), np.nan)
        for y in p.year.unique():
            col = f"{name}|{y}"
            if col in x:
                m = (p.year == y).to_numpy()
                v = x[col].reindex(p.pubid[m]).to_numpy().astype(float)
                out[m] = np.where(v == -4, 0.0, np.where(v >= 0, v, np.nan))
        return out
    kh, kn = at_round_kids("CV_BIO_CHILD_HH"), at_round_kids("CV_BIO_CHILD_NR")
    kh2, kn2 = at_round_kids("CV_BIO_CHILD_HH_U18"), at_round_kids("CV_BIO_CHILD_NR_U18")
    p["kids"] = np.where(np.isnan(kh), kh2, kh) + np.where(np.isnan(kn), kn2, kn)
    urb = at_round("CV_URBAN-RURAL")
    p["urban"] = np.where(urb == 1, 1.0, np.where(urb == 0, 0.0, np.nan))
    p["region"] = at_round("CV_CENSUS_REGION")
    p["weight_panel"] = at_round("SAMPLING_PANEL_WEIGHT") / 100

    dur, nprev, evdiv = (np.full(len(p), np.nan) for _ in range(3))
    for k, (pid, cm, st) in enumerate(zip(p.pubid, p.cmonth, p.arr_state)):
        i = row[pid]
        j = cm - months[0]
        if j < 0 or j >= len(months) or not isinstance(st, str):
            continue
        a, pc, rs = arr[i, : j + 1], part[i, : j + 1], raw_status[i, : j + 1]
        evdiv[k] = float(np.isin(rs, [3, 4]).any())

        starts, cur = [], None
        for m in range(len(a)):
            in_u = a[m] in ("C", "M")
            if not in_u:
                cur = None
                continue
            new = cur is None
            if cur is not None and not np.isnan(pc[m]) and not np.isnan(cur[1]):
                if pc[m] // 100 == cur[1] // 100 and pc[m] != cur[1]:
                    new = True
            if new:
                starts.append(m)
            cur = (a[m], pc[m] if not np.isnan(pc[m]) else (cur[1] if cur else np.nan))
        if st in ("C", "M"):

            m = j
            while m - 1 >= 0 and a[m - 1] == st and not (
                    not np.isnan(pc[m - 1]) and not np.isnan(pc[m]) and pc[m - 1] // 100 == pc[m] // 100 and pc[m - 1] != pc[m]):
                m -= 1
            first_valid = np.argmax(a != "")
            dur[k] = (j - m + 1) / 12 if m > first_valid else np.nan
            nprev[k] = max(len(starts) - 1, 0)
        else:
            nprev[k] = len(starts)
    p["dur_state_yrs"] = dur
    p["n_unions_before"] = nprev
    p["ever_divorced"] = evdiv

    ddur = np.full(len(p), np.nan)
    p = p.sort_values(["pubid", "round"]).reset_index(drop=True)
    last_start = {}
    born12 = ((x["KEY!BDATE_Y|1997"] + 12 - 1980) * 12 + x["KEY!BDATE_M|1997"]).to_dict()
    for k, (pid, y, st, cm) in enumerate(zip(p.pubid, p.year, p.state, p.cmonth)):
        if not 2002 <= y <= 2011:
            continue
        d5 = x.at[pid, f"YSAQ2-298D5A|{y}"]
        sm, sy = x.at[pid, f"YSAQ2-298D4~M|{y}"], x.at[pid, f"YSAQ2-298D4~Y|{y}"]
        q707 = x.at[pid, f"YSAQ2-707|{y}"] if f"YSAQ2-707|{y}" in x else -4
        start = (sy - 1980) * 12 + sm if (sm > 0 and sy > 0) else np.nan
        if st == "D":
            if q707 == 1 and pid in last_start:
                s0 = last_start[pid]
            elif d5 == 1:
                s0 = start
            else:
                s0 = np.nan
            if not np.isnan(s0) and s0 <= cm and s0 >= born12[pid]:
                ddur[k] = (cm - s0 + 1) / 12
            last_start[pid] = s0
        else:
            last_start.pop(pid, None)
    p["dur_dating_yrs"] = ddur
    p.to_csv(CACHE / "nlsy97_panel_cov.csv", index=False)

    cols = ["female", "black", "hispanic", "afqt", "mom_edu", "both_parents12", "fam_risk", "parent_relig",
            "tipi_extra", "tipi_agree", "tipi_consc", "tipi_stable", "tipi_open", "edu_years", "degree",
            "enrolled", "pov_ratio", "kids", "urban", "age", "dur_state_yrs", "dur_dating_yrs",
            "n_unions_before", "ever_divorced"]
    w = p[(p.year >= 2002) & (p.year <= 2011)]
    print(w[cols].describe().T[["count", "mean", "std", "min", "max"]].round(2).to_string())
    print("dur_state_yrs by state:", w.groupby("state").dur_state_yrs.median().to_dict())
    print("dur_dating_yrs coverage among D:", w[w.state == "D"].dur_dating_yrs.notna().mean().round(3))

if __name__ == "__main__":
    import sys
    steps = [step_nlsy_index, step_nlsy_extract, step_nlsy_verify, step_nlsy_build, step_nlsy_covariates]
    for s in steps:
        if len(sys.argv) == 1 or s.__name__ in sys.argv[1:]:
            s()
