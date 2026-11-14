# Covariate groups, data loading, forecast targets, forecasting, scoring with inverse-probability weights, real income and the stage-unknown baseline.
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from chain import Params, generator, propagate, delayed_entry, edge_rates, propagate_sparse
from sequences import SOURCES
from chain import Space
from sequences import build, loglik, fit

CONT = {"D": 0, "C": 1, "M": 2}

def window(h):
    return 0.75 if h <= 4 else 1.25

def episode_set(rows):

    s0 = rows.state.iat[0]
    status = "cont"
    tag = None
    for k in range(1, len(rows)):
        st, same, et = rows.state.iat[k], rows.same_partner.iat[k], rows.end_tag.iat[k]
        if status == "ended":
            break
        if st == "E" or st == "S" or (same == 0):
            status, tag = "ended", (et if isinstance(et, str) and et in ("B", "X") else None)
            continue
        if st == "N":

            status = "ambiguous" if status == "cont" else status
            continue
        if same == 1:
            continue

        status = "ambiguous"
    out = np.zeros(5, bool)
    last = rows.state.iat[-1]
    if status == "ended":
        if tag == "B":
            out[3] = True
        elif tag == "X":
            out[4] = True
        else:
            out[3] = out[4] = True
        return out
    cont_state = {"D": "D", "C": "C", "M": "M", "N": "D", "U": None}.get(last)
    if cont_state is not None:
        out[CONT[cont_state]] = True
    if last == "U":
        out[0] = out[1] = True
    if status == "ambiguous":
        out[3] = out[4] = True
    return out

def person_class(st):
    return {"S": 0, "D": 0, "N": 0, "C": 1, "M": 2}.get(st, -1)

def pairs(panel: pd.DataFrame, origins: pd.DataFrame, horizons):

    panel = panel.sort_values(["pid", "t_years"])
    by = {pid: g.reset_index(drop=True) for pid, g in panel.groupby("pid", sort=False)}
    out = []
    for _, o in origins.iterrows():
        g = by[o.pid]
        k0 = int(np.where(np.isclose(g.t_years.to_numpy(), o.t_years))[0][0])
        el = g.t_years.to_numpy() - o.t_years
        for h in horizons:
            c = np.where((el > 0) & (np.abs(el - h) <= window(h)))[0]
            if len(c) == 0:
                continue
            k = c[np.argmin(np.abs(el[c] - h))]
            rows = g.iloc[k0:k + 1]
            pc = person_class(g.state.iat[k]) if o.cohort != "HCMST2009" else -1
            es = episode_set(rows) if o.state in ("D", "C", "M") else np.zeros(5, bool)
            out.append(dict(pid=o.pid, origin_t=o.t_years, h=h, elapsed=el[k], person_y=pc,
                            ep_set=es, ep_full=bool(es.sum() == 1), ep_any=bool(es.any())))
    return pd.DataFrame(out)

def featurize(rows: pd.DataFrame, xcols, stats):
    X = rows[xcols].to_numpy(float)
    mu, sd = stats
    Z = (X - mu) / sd
    miss = np.isnan(Z)
    Z[miss] = 0.0
    return torch.tensor(np.concatenate([Z, miss.astype(float)], 1))

def person_probs(sp, v):
    names = sp.names
    grp = np.zeros((sp.n, 3))
    for i, nm in enumerate(names):
        st = nm.split("-")[1][0]
        grp[i, {"S": 0, "D": 0, "C": 1, "M": 2}[st]] = 1
    return v @ torch.tensor(grp)

def episode_probs(sp, v):
    grp = np.zeros((sp.n, 5))
    for i, nm in enumerate(sp.names):
        blk, st = nm.split("-")
        if blk == "R":
            grp[i, {"D": 0, "C": 1, "M": 2}[st[0]]] = 1
        elif blk == "B":
            grp[i, 3] = 1
        else:
            grp[i, 4] = 1
    return v @ torch.tensor(grp)

@torch.no_grad()
def forecast(par: Params, O: pd.DataFrame, xcols, stats, elapsed, use_order=True, stage=None):

    sp = par.sp
    N = len(O)
    x = featurize(O, xcols, stats)
    src = torch.tensor([SOURCES.index(c) for c in O.cohort])
    age = torch.tensor(O.age.to_numpy(float))
    order = torch.tensor(np.nan_to_num(O.prior_union.to_numpy(float))) if use_order else torch.zeros(N)
    v = torch.zeros(N, sp.n)
    if stage is None:
        st = O.state.to_numpy()
        v[torch.tensor(np.where(st == "S")[0]), sp.idx["B-S"]] = 1
        for s in "DCM":
            r = np.where(st == s)[0]
            if len(r) == 0:
                continue
            rr = torch.tensor(r)
            dist = delayed_entry(par, s, torch.tensor(O.dur_state_yrs.to_numpy(float)[r]), x[rr], src[rr], age[rr], order[rr])
            v[rr[:, None], torch.tensor(sp.phase_block(s))[None, :]] = dist
    else:
        pi = torch.tensor(stage["pi"])
        v[:, sp.idx["B-S"]] = pi[:, 0]
        for j, s in enumerate("DCM", 1):
            dist = delayed_entry(par, s, torch.full((N,), float(stage["dur"][s])), x, src, age, order)
            v[:, torch.tensor(sp.phase_block(s))] = pi[:, j][:, None] * dist
    e = np.asarray(elapsed, float)
    out = torch.zeros(N, sp.n)
    k = 0
    while True:
        due = np.where((e >= k) & (e < k + 1))[0]
        if len(due):
            d = torch.tensor(due)
            r_ = edge_rates(par, x[d], src[d], age[d] + k + (torch.tensor(e[due]) - k) / 2, order[d])
            out[d] = propagate_sparse(v[d], r_, sp, torch.tensor(e[due] - k))
        if (e < k + 1).all():
            break
        live = torch.tensor(np.where(e >= k + 1)[0])
        r_ = edge_rates(par, x[live], src[live], age[live] + k + 0.5, order[live])
        v[live] = propagate_sparse(v[live], r_, sp, torch.ones(len(live)))
        k += 1
    return person_probs(sp, out), episode_probs(sp, out)

def all_pairs(panel, origins, horizons):

    panel = panel.sort_values(["pid", "t_years"])
    by = {pid: g.reset_index(drop=True) for pid, g in panel.groupby("pid", sort=False)}
    rows = []
    for oi, o in origins.reset_index(drop=True).iterrows():
        g = by[o.pid]
        k0 = int(np.argmin(np.abs(g.t_years.to_numpy() - o.t_years)))
        el = g.t_years.to_numpy() - o.t_years
        for h in horizons:
            c = np.where((el > 0) & (np.abs(el - h) <= window(h)))[0]
            r = dict(oi=oi, pid=o.pid, h=h, observed=len(c) > 0, elapsed=np.nan, person_y=-1, ep_set=None)
            if len(c):
                k = c[np.argmin(np.abs(el[c] - h))]
                r["elapsed"] = el[k]
                r["person_y"] = person_class(g.state.iat[k]) if o.cohort != "HCMST2009" else -1
                r["ep_set"] = episode_set(g.iloc[k0:k + 1]) if o.state in ("D", "C", "M") else None
            rows.append(r)
    return pd.DataFrame(rows)

def reachable(panel, origins, horizons):

    times = {c: np.sort(g.t_years.unique()) for c, g in panel.groupby("cohort")}
    out = np.zeros((len(origins), len(horizons)), bool)
    for i, (c, t) in enumerate(zip(origins.cohort, origins.t_years)):
        tt = times[c]
        for j, h in enumerate(horizons):
            out[i, j] = (np.abs(tt - (t + h)) <= window(h)).any()
    return out

def obs_model(train_origins, train_pairs, feat_cols):

    X = _obs_X(train_origins, train_pairs, feat_cols)
    m = LogisticRegression(max_iter=2000, C=1.0).fit(X, train_pairs["observed"].to_numpy())
    return m

def _obs_X(origins, pairs, feat_cols):
    O = origins.reset_index(drop=True).iloc[pairs.oi.to_numpy()]
    F = O[feat_cols].to_numpy(float)
    F = np.where(np.isnan(F), 0.0, F)
    st = np.stack([(O.state == s).to_numpy(float) for s in "SDCM"], 1)
    src = np.stack([(O.cohort == c).to_numpy(float) for c in SOURCES], 1)
    h = pairs.h.to_numpy(float)[:, None]
    return np.concatenate([F, st, src, h, h ** 2 / 10, st * h, src * h], 1)

def weights(model, origins, pairs, feat_cols):
    p = model.predict_proba(_obs_X(origins, pairs, feat_cols))[:, 1]
    w = 1.0 / np.clip(p, 1e-3, 1)
    import os
    if os.environ.get("LOVE_NOTRIM") == "1":
        return w
    lo, hi = np.nanpercentile(w, [1, 99])
    return np.clip(w, lo, hi)

def log_loss_person(pp, y, w):
    p = pp[np.arange(len(y)), y].clip(1e-12, 1)
    return float(np.sum(-np.log(p) * w) / np.sum(w))

def log_loss_set(pe, sets, w):
    p = (pe * sets).sum(1).clip(1e-12, 1)
    return float(np.sum(-np.log(p) * w) / np.sum(w))

ROOT = os.environ.get("LOVE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

XC_MAIN = ["female", "black", "hispanic", "edu_years", "log_income", "kids_any", "rel_quality",
           "rel_close", "rel_conflict", "rel_future", "afqt", "mom_edu", "both_parents12", "parent_relig"]
XC_TIPI = XC_MAIN + ["tipi_extra", "tipi_agree", "tipi_consc", "tipi_stable", "tipi_open"]
OBS_FEATS = ["age", "female", "black", "hispanic", "edu_years", "log_income"]

def load(variant=None):

    import os
    A = pd.read_csv(f"{ROOT}/cache/panel.csv", low_memory=False)
    A["end_tag"] = A.end_tag.fillna("")
    variant = variant or os.environ.get("LOVE_VARIANT")
    if variant in ("same", "diff"):
        A = A.sort_values(["pid", "t_years"]).reset_index(drop=True)
        prev = A.groupby("pid").state.shift(1)
        m = (prev == "D") & A.state.isin(["C", "M"]) & A.same_partner.isna()
        A.loc[m, "same_partner"] = 1.0 if variant == "same" else 0.0
        print(f"variant {variant}: set {int(m.sum())} unknown identities", flush=True)
    if variant == "noother":

        h9 = pd.read_csv(f"{ROOT}/cache/hcmst2009_panel.csv")
        bad = h9[h9.end_type.isin(["ended_other", "ended_per_official_flag", "marriage_ended"])]
        key = set(zip("H9_" + bad.caseid.astype(str), bad.wave))
        m = A.cohort.eq("HCMST2009") & pd.Series(list(zip(A.pid, A.wave))).isin(key).to_numpy()
        A = A[~m].reset_index(drop=True)
        print(f"variant noother: dropped {int(m.sum())} HCMST 2009 endings of other or unknown type", flush=True)
    return A

def landmark_nlsy(df, years, min_age=18):
    o = df[(df.cohort == "NLSY97") & df.year.isin(years) & (df.age >= min_age) & df.state.isin(list("SDCM"))]
    return o.sort_values("t_years").groupby("pid").head(1)

def origins_hcmst(df, cohort):
    o = df[(df.cohort == cohort) & (df.wave == 1) & (df.age >= 18) & df.state.isin(list("SDCM"))]
    return o

def score(par, stats, xcols, panel, origins, horizons, obsm):

    P = all_pairs(panel, origins, horizons)
    O = origins.reset_index(drop=True)
    reach = reachable(panel, O, horizons)
    P["reach"] = reach[P.oi.to_numpy(), [horizons.index(h) for h in P.h]]
    P = P[P.reach]
    w_all = weights(obsm, O, P, OBS_FEATS)
    P = P.assign(w=w_all)
    Pobs = P[P["observed"]].reset_index(drop=True)
    Oo = O.iloc[Pobs.oi.to_numpy()].reset_index(drop=True)
    pp, pe = forecast(par, Oo, xcols, stats, Pobs.elapsed.to_numpy())
    pp, pe = pp.numpy(), pe.numpy()
    res = {}
    for h in horizons:
        m = (Pobs.h == h).to_numpy()
        mp = m & (Pobs.person_y >= 0).to_numpy()
        me = m & Pobs.ep_set.notna().to_numpy()
        r = {}
        if mp.sum():
            r["person"] = log_loss_person(pp[mp], Pobs.person_y.to_numpy()[mp], Pobs.w.to_numpy()[mp])
            r["n_person"] = int(mp.sum())
        if me.sum():
            S = np.stack(Pobs.ep_set[me].to_numpy())
            r["episode"] = log_loss_set(pe[me], S, Pobs.w.to_numpy()[me])
            r["n_episode"] = int(me.sum())
        res[h] = r
    return res, Pobs, pp, pe

def fit_obs_model(train_panel, horizons_map):

    Os, Ps = [], []
    off = 0
    for O, H in horizons_map:
        O = O.reset_index(drop=True)
        P = all_pairs(train_panel, O, H)
        reach = reachable(train_panel, O, H)
        P = P[reach[P.oi.to_numpy(), [H.index(h) for h in P.h]]].copy()
        P["oi"] = P.oi + off
        off += len(O)
        Os.append(O); Ps.append(P)
    O = pd.concat(Os, ignore_index=True); P = pd.concat(Ps, ignore_index=True)
    return obs_model(O, P, OBS_FEATS)

CPI_U = {1978: 65.233, 1979: 72.575, 1980: 82.408, 1981: 90.925, 1982: 96.5, 1983: 99.6, 1984: 103.883, 1985: 107.567, 1986: 109.608, 1987: 113.625, 1988: 118.258, 1989: 123.967, 1990: 130.658, 1991: 136.192, 1992: 140.317, 1993: 144.458, 1994: 148.225, 1995: 152.383, 1996: 156.85, 1997: 160.517, 1998: 163.008, 1999: 166.575, 2000: 172.2, 2001: 177.067, 2002: 179.875, 2003: 183.958, 2004: 188.883, 2005: 195.292, 2006: 201.592, 2007: 207.342, 2008: 215.303, 2009: 214.537, 2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736, 2015: 237.017, 2016: 240.007, 2017: 245.12, 2018: 251.107, 2019: 255.657, 2020: 258.811, 2021: 270.97, 2022: 292.655, 2023: 304.702, 2024: 313.689}

def add_real_income(panel, base_year=2020):
    cpi = pd.Series(CPI_U)
    year = np.floor(1980 + panel.t_years.to_numpy(float)).astype(int)
    ref = np.where(panel.cohort.isin(["NLSY97", "NLSY79"]).to_numpy(), year - 1, year)
    ref = np.clip(ref, cpi.index.min(), cpi.index.max())
    factor = cpi[base_year] / cpi.reindex(ref).to_numpy()
    out = panel.copy()
    out["log_income_real"] = np.log1p(np.expm1(panel.log_income.to_numpy(float)) * factor)
    return out

GROUP_VARS = {"G1": ["rel_quality", "rel_close", "rel_conflict", "rel_future"],
              "G2": ["black", "hispanic", "edu_years", "log_income", "kids_any"],
              "G3": ["afqt", "mom_edu", "both_parents12", "parent_relig"],
              "G4": ["tipi_extra", "tipi_agree", "tipi_consc", "tipi_stable", "tipi_open"]}

def xcols_for(groups):
    cols = ["female"]
    for g in ("G1", "G2", "G3", "G4"):
        if g in groups:
            cols += GROUP_VARS[g]
    return cols

def stage_model(train_origins):

    def X(O):
        a = (O.age.to_numpy(float) - 30) / 10
        return np.column_stack([a, a ** 2, a ** 3, O.female.to_numpy(float)] + [(O.cohort == c).to_numpy(float) for c in ("NLSY97", "HCMST2009", "HCMST2017")])
    y = train_origins.state.map({"S": 0, "D": 1, "C": 2, "M": 3}).to_numpy()
    m = LogisticRegression(max_iter=3000).fit(X(train_origins), y)
    dur = train_origins.groupby("state").dur_state_yrs.median().to_dict()
    def pi(O):
        p = np.zeros((len(O), 4)); p[:, m.classes_] = m.predict_proba(X(O)); return p
    return pi, {s: float(dur.get(s, 2.0)) for s in "DCM"}
