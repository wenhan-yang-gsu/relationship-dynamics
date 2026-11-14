# NLSY79: extraction, codebook checks and the external confirmation panel.
def step_nlsy79_extract():

    from pathlib import Path
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    RAW = ROOT / "raw_data/NLSY79/raw/nlsy79_all_1979-2022.csv"
    CACHE = ROOT / "cache"
    ix = pd.read_csv(CACHE / "nlsy79_index.csv", dtype=str)

    EXACT = ["CASEID", "HHID", "Q1-3_A~M", "Q1-3_A~Y", "SAMPLE_ID", "SAMPLE_RACE", "SAMPLE_SEX",
             "HGC-MOTHER", "HGC-FATHER", "FAM-7", "AFQT-3", "CURDATE~M", "CURDATE~Y", "RNI",
             "MARSTAT-KEY", "HH5-5", "Q2-10A", "TNFI_TRUNC", "SAMPWEIGHT", "AGEATINT", "NUMKID"]
    PATTERNS = [r"^HGCREV\d\d$", r"^NUMKID\d\d$", r"^HHI_FINAL_RELCODE\.\d+$", r"^HHI_FINAL_GENCODE\.\d+$"]

    sel = ix.qname.isin(EXACT)
    for p in PATTERNS:
        sel |= ix.qname.str.match(p, na=False)
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
    assert len(out) == 12686 and out["CASEID|1979"].is_unique
    out.to_csv(CACHE / "nlsy79_extract.csv", index=False)
    vars_[["ref", "qname", "year", "desc", "name"]].to_csv(CACHE / "nlsy79_extract_vars.csv", index=False)
    print(out.shape)
    print(vars_.assign(fam=vars_.qname.str.replace(r"\d\d$|\.\d+$", "", regex=True)).groupby("fam").size().to_string())

def step_nlsy79_verify():

    from pathlib import Path
    import re
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    CDB = ROOT / "raw_data/NLSY79/raw/nlsy79_all_1979-2022.cdb"
    CACHE = ROOT / "cache"
    x = pd.read_csv(CACHE / "nlsy79_extract.csv")
    v = pd.read_csv(CACHE / "nlsy79_extract_vars.csv", dtype=str)
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

def step_nlsy79_build():

    from pathlib import Path
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    CACHE = ROOT / "cache"
    x = pd.read_csv(CACHE / "nlsy79_extract.csv")
    v = pd.read_csv(CACHE / "nlsy79_extract_vars.csv", dtype=str)

    def nonneg(s):
        return s.where(s >= 0)

    keep = x["SAMPLE_ID|1979"].isin([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14])
    x = x[keep].reset_index(drop=True)
    YEARS = sorted(int(y) for y in v[v.qname == "MARSTAT-KEY"].year)

    by = x["Q1-3_A~Y|1981"].where(x["Q1-3_A~Y|1981"] > 0, x["Q1-3_A~Y|1979"]) + 1900
    bm = x["Q1-3_A~M|1981"].where(x["Q1-3_A~M|1981"] > 0, x["Q1-3_A~M|1979"])
    fixed = pd.DataFrame(dict(
        caseid=x["CASEID|1979"],
        female=(x["SAMPLE_SEX|1979"] == 2).astype(float),
        black=(x["SAMPLE_RACE|78SCRN"] == 2).astype(float),
        hispanic=(x["SAMPLE_RACE|78SCRN"] == 1).astype(float),
        afqt=nonneg(x["AFQT-3|1981"]) / 1000,
        mom_edu=nonneg(x["HGC-MOTHER|1979"]).where(lambda s: s != 95),
        both_parents12=(x["FAM-7|1979"] == 11).astype(float).where(x["FAM-7|1979"] >= 0),
    ))

    def col(stem, yr):
        c = f"{stem}|{yr}"
        return x[c] if c in x else pd.Series(np.nan, index=x.index)

    def edu(yr):
        c = [n for n in v[(v.year == str(yr)) & v.qname.str.match(r"^HGCREV\d\d$")].name]
        return x[c[0]] if c else pd.Series(np.nan, index=x.index)

    def kids(yr):
        c = [n for n in v[(v.year == str(yr)) & v.qname.str.match(r"^NUMKID(\d\d)?$")].name]
        return x[c[0]] if c else pd.Series(np.nan, index=x.index)

    rows = []
    for wave, yr in enumerate(YEARS, 1):
        ms = col("MARSTAT-KEY", yr)
        inter = col("RNI", yr).isna() | (col("RNI", yr) < 0)
        inter &= ms.notna() & (ms != -5)
        rel = [n for n in v[(v.year == str(yr)) & v.qname.str.startswith("HHI_FINAL_RELCODE")].name]
        R = x[rel].to_numpy()
        spouse_hh = (R == 1).any(1)
        partner_hh = (R == 33).any(1) | (col("HH5-5", yr) == 1).to_numpy()
        married = ms.isin([1, 5]).to_numpy() | ((ms == 2).to_numpy() & spouse_hh)
        known = (ms >= 0).to_numpy()
        state = np.where(~known, None, np.where(married, "M", np.where(partner_hh, "C", "N")))
        m = col("CURDATE~M", yr)
        y = col("CURDATE~Y", yr).where(lambda s: s > 0, yr) if f"CURDATE~Y|{yr}" in x else pd.Series(yr, index=x.index)
        y = y.where(y >= 100, y + 1900)
        cmonth = (y - 1980) * 12 + m
        age = ((y * 12 + m) - (by * 12 + bm) + 0.5) / 12
        d = pd.DataFrame(dict(caseid=x["CASEID|1979"], wave=wave, year=yr, cmonth=cmonth, t_years=(cmonth - 0.5) / 12,
                              state=state, age=age, edu_years=nonneg(edu(yr)).where(lambda s: s != 95),
                              income=nonneg(col("TNFI_TRUNC", yr)), kids=nonneg(kids(yr)),
                              weight=nonneg(col("SAMPWEIGHT", yr)) / 100, ms=ms, spouse_hh=spouse_hh,
                              roster_partner=(R == 33).any(1), hh55=col("HH5-5", yr)))
        rows.append(d[inter.to_numpy() & (m > 0).to_numpy() & known])
    P = pd.concat(rows).sort_values(["caseid", "wave"]).reset_index(drop=True)
    for c in ["edu_years", "income", "kids"]:
        P[c] = P.groupby("caseid")[c].ffill()
    P = P.merge(fixed, on="caseid", how="left")
    P["log_income"] = np.log1p(P.income)
    P["kids_any"] = (P.kids > 0).astype(float).where(P.kids.notna())

    P["_union"] = ((P.state == "M") | ((P.state == "C") & (P.year >= 1983))).astype(int)
    P["prior_union"] = (P.groupby("caseid")._union.transform(lambda s: s.cumsum().shift(fill_value=0)) > 0).astype(float)
    P = P.drop(columns="_union")
    P["dataset"] = "NLSY79"; P["cohort"] = "NLSY79"; P["pid"] = "L" + P.caseid.astype(str)
    P["household"] = "LH" + P.caseid.map(dict(zip(x["CASEID|1979"], x["HHID|1979"]))).astype(int).astype(str)
    for c in ["same_partner", "dur_state_yrs", "rel_quality", "rel_close", "rel_conflict", "rel_future", "parent_relig",
              "tipi_extra", "tipi_agree", "tipi_consc", "tipi_stable", "tipi_open"]:
        P[c] = np.nan
    P["end_tag"] = ""; P["split"] = "external"
    P.to_csv(CACHE / "nlsy79_panel.csv", index=False)
    print(P.shape, P.pid.nunique(), "persons")
    print(pd.crosstab(P.year, P.state).to_string())

if __name__ == "__main__":
    import sys
    steps = [step_nlsy79_extract, step_nlsy79_verify, step_nlsy79_build]
    for s in steps:
        if len(sys.argv) == 1 or s.__name__ in sys.argv[1:]:
            s()
