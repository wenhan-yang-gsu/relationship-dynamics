# Information decomposition under the final structure: Shapley values of four information sources withheld at forecast time, out of fold.
import sys, os, json, itertools, math, time
import numpy as np, pandas as pd, torch
import common as C, cv, structures as LT
from chain import Params

torch.set_num_threads(4)
GROUPS = {"feelings": ["rel_quality", "rel_close", "rel_conflict", "rel_future"],
          "background": ["black", "hispanic", "edu_years", "log_income", "kids_any"],
          "family": ["afqt", "mom_edu", "both_parents12", "parent_relig"],
          "history": []}
NAMES = list(GROUPS)
OUTD = f"{C.ROOT}/runs/info_decomp"
tag = lambda S: "+".join(S) if S else "none"

def load_mixture(k):
    ck = torch.load(f"{cv.OUT}/phase_lat_f{k}.pt", weights_only=False)
    sp = cv.make_space(ck["cfg"]); xc, st = ck["xcols"], ck["stats"]
    par = Params(sp, p=len(xc) * 2, n_src=3, init=ck["par"]); par.age_range = ck["age_range"]
    return LT.Mixture(par, ck["G"], delta=ck["delta"], W=ck["W"]), xc, st

def predictor(m, panel, xc, st, withheld):
    def predict(Oo, el):
        O = Oo.copy()
        for g in withheld:
            if GROUPS[g]:
                O[GROUPS[g]] = np.nan
        if "history" in withheld:
            O["prior_union"] = 0.0
            post = LT.origin_posterior(m, panel.iloc[:0], O, m.par.sp, xc, st)
        else:
            post = LT.origin_posterior(m, panel, O, m.par.sp, xc, st)
        return LT.forecast_mixture(m, O, xc, st, el, post)
    return predict

def run_fold(k):
    os.makedirs(OUTD, exist_ok=True)
    m, xc, st = load_mixture(k)
    A = C.load(); A = A[A.split.isin(["train", "valid"])].merge(cv.folds(A), on="household", how="left")
    tr, ev = A[A.fold != k], A[A.fold == k]
    obsm = C.fit_obs_model(tr, [(cv.origins(tr, lm), H) for lm, H in cv.PRIMARY.items()])
    for r in range(len(NAMES) + 1):
        for S in itertools.combinations(NAMES, r):
            withheld = [g for g in NAMES if g not in S]
            t = time.time()
            res, pairs, pm = cv.score_all(predictor(m, ev, xc, st, withheld), ev, obsm)
            pairs.to_csv(f"{OUTD}/f{k}_{tag(S)}_pairs.csv", index=False)
            print(f"fold {k} known={tag(S)}: mean {pm:.5f}, {time.time() - t:.0f}s", flush=True)

def shapley(L):

    n = len(NAMES); phi = {}
    for g in NAMES:
        tot = 0.0
        for r in range(n):
            for S in itertools.combinations([x for x in NAMES if x != g], r):
                w = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                tot += w * (L[frozenset(S)] - L[frozenset(S) | {g}])
        phi[g] = tot
    return phi

def summary():
    A = C.load(); hh = A.drop_duplicates("pid").set_index("pid").household
    O = pd.concat([cv.origins(A, lm).assign(landmark=lm) for lm in cv.PRIMARY]).reset_index(drop=True)
    subsets = [frozenset(S) for r in range(len(NAMES) + 1) for S in itertools.combinations(NAMES, r)]
    D = None
    for S in subsets:
        d = pd.concat([pd.read_csv(f"{OUTD}/f{k}_{tag(sorted(S, key=NAMES.index))}_pairs.csv") for k in range(5)]).reset_index(drop=True)
        p = d[["p0", "p1", "p2"]].to_numpy()
        loss = -np.log(np.clip(p[np.arange(len(d)), d.person_y.astype(int)], 1e-12, None))
        if D is None:
            D = d[["landmark", "pid", "h", "w", "oi"]].copy()
            D["household"] = D.pid.map(hh)
        else:
            assert (d.pid.to_numpy() == D.pid.to_numpy()).all() and (d.h.to_numpy() == D.h.to_numpy()).all()
        D[tag(sorted(S, key=NAMES.index))] = loss
    st = {}
    for lm, g in O.groupby("landmark"):
        st.update({(lm, i): s for i, s in zip(g.index - g.index.min(), g.state)})
    D["state"] = [st.get((lm, oi), "?") for lm, oi in zip(D.landmark, D.oi)]
    D["state"] = D.state.replace({"N": "S"})
    def phi_of(g):
        L = {S: float(np.average(g[tag(sorted(S, key=NAMES.index))], weights=g.w)) for S in subsets}
        return shapley(L), L[frozenset(NAMES)], L[frozenset()]
    rng = np.random.default_rng(20261111)
    out = {"by_cell": {}, "by_cell_state": {}}
    for (lm, h), g in D.groupby(["landmark", "h"]):
        phi, full, none = phi_of(g)
        boots = {n: [] for n in NAMES}
        gh = g.household.to_numpy(); uhg = np.unique(gh); pos = {u: np.where(gh == u)[0] for u in uhg}
        for _ in range(1000):
            idx = np.concatenate([pos[u] for u in rng.choice(uhg, len(uhg))])
            ph, _, _ = phi_of(g.iloc[idx])
            for n in NAMES: boots[n].append(ph[n])
        out["by_cell"][f"{lm}|{h}"] = dict(n=int(len(g)), loss_all=full, loss_none=none, phi=phi,
                                          phi_ci90={n: list(np.percentile(boots[n], [5, 95])) for n in NAMES})
        for s, gs in g.groupby("state"):
            if len(gs) < 100: continue
            phi_s, full_s, none_s = phi_of(gs)
            out["by_cell_state"][f"{lm}|{h}|{s}"] = dict(n=int(len(gs)), loss_all=full_s, loss_none=none_s, phi=phi_s)
        print(lm, h, {n: round(phi[n], 4) for n in NAMES}, "all", round(full, 4), "none", round(none, 4), flush=True)
    json.dump(out, open(f"{OUTD}/summary.json", "w"), indent=1)
    rows = [dict(landmark=k.split("|")[0], h=int(k.split("|")[1]), state="all", n=v["n"], loss_all=v["loss_all"], loss_none=v["loss_none"], **{f"phi_{n}": v["phi"][n] for n in NAMES}) for k, v in out["by_cell"].items()]
    rows += [dict(landmark=k.split("|")[0], h=int(k.split("|")[1]), state=k.split("|")[2], n=v["n"], loss_all=v["loss_all"], loss_none=v["loss_none"], **{f"phi_{n}": v["phi"][n] for n in NAMES}) for k, v in out["by_cell_state"].items()]
    pd.DataFrame(rows).to_csv(f"{OUTD}/summary.csv", index=False)

if __name__ == "__main__":
    summary() if sys.argv[1] == "summary" else run_fold(int(sys.argv[1]))
