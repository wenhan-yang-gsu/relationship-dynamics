# External confirmation on NLSY79: refit with the shared information set, score once at the pre-specified landmarks and horizons, apply the decision table.
import os, sys, json
import numpy as np, pandas as pd, torch
import common as C, cv, structures as LT, structures as FL
import competitors as K
from chain import Params
from competitors import training_pairs, HorizonClassifier
from common import stage_model
from sklearn.linear_model import LogisticRegression

STANDIN = os.environ.get("NLSY79_STANDIN") == "1"
MARGINS = {"nlsy2002": {"1": 0.04754, "5": 0.018726, "10": 0.014771, "20": 0.011179}, "nlsy2008": {"1": 0.112344, "5": 0.051749, "10": 0.044997, "14": 0.033526}, "hcmst2017": {"3": 0.102822, "5": 0.093505}}
LMS = {"L1": dict(year=1983, ages=(18, 25), H=[1, 5, 10, 20], X=[30, 39], ref="nlsy2002"),
       "L2": dict(year=1989, ages=(24, 31), H=[1, 5, 10, 14], X=[20, 33], ref="nlsy2008")}
if STANDIN:
    LMS = {"L1": dict(year=2002, ages=(18, 25), H=[1, 5, 10, 20], X=[], ref="nlsy2002"),
           "L2": dict(year=2008, ages=(24, 31), H=[1, 5, 10, 14], X=[], ref="nlsy2008")}
OUTDIR = f"{C.ROOT}/runs/nlsy79" + ("_standin" if STANDIN else "")

def load_panels():
    A = C.load()
    A = A[A.split.isin(["train", "valid"])].merge(cv.folds(A), on="household", how="left")
    A = C.add_real_income(A)
    if STANDIN:
        P = A[(A.fold == 0) & (A.cohort == "NLSY97")].copy()
        P.loc[P.state.isin(["S", "D"]), "state"] = "N"
        P["cohort"] = "NLSY79"
        P[["rel_quality", "rel_close", "rel_conflict", "rel_future", "parent_relig", "same_partner"]] = np.nan
    else:
        P = pd.read_csv(f"{C.ROOT}/cache/nlsy79_panel.csv")
        P = C.add_real_income(P)
        P["end_tag"] = P.end_tag.fillna("")
    for D in (A, P):
        D["birth_year"] = 1980 + D.t_years - D.age
    return A, P

def landmark(P, spec):
    o = P[(P.year == spec["year"]) & (P.age >= spec["ages"][0]) & (P.age < spec["ages"][1]) & P.state.isin(["N", "C", "M"])]
    return o.sort_values("t_years").groupby("pid").head(1).reset_index(drop=True)

def chain_models(name):
    fold = os.environ.get("NLSY79_FITFOLD", "-1")
    ck = torch.load(f"{cv.OUT}/{name}_f{fold}.pt", weights_only=False)
    cfg = ck["cfg"]; sp = cv.make_space(cfg); xc, st = ck["xcols"], ck["stats"]
    par = Params(sp, p=len(xc) * 2, n_src=3, init=ck["par"]); par.age_range = ck["age_range"]
    par.age_linear = os.environ.get("AGE_LINEAR") == "1"
    G = int(ck.get("G") or 1)
    m = LT.Mixture(par, G, delta=ck["delta"], W=ck["W"]) if G > 1 else None
    return dict(par=par, m=m, sp=sp, xc=xc, st=st, use_order="G1" in cfg["groups"], G=G)

def chain_predict(M, P, pi, dur):
    def f(Oo, el):
        Oo = Oo.reset_index(drop=True)
        from common import featurize
        fem = featurize(Oo, M["xc"], M["st"])[:, M["xc"].index("female")] if "female" in M["xc"] else torch.zeros(len(Oo))
        if M["m"] is not None:
            chains = M["m"].chains()
            with torch.no_grad():
                lp = M["m"].log_prior(LT.z_from(fem, torch.zeros(len(Oo), dtype=torch.long))).numpy()
        else:
            chains, lp = [M["par"]], np.zeros((len(Oo), 1))
        p = pi(Oo.assign(cohort="NLSY97"))
        pD = p[:, 1] / np.clip(p[:, 0] + p[:, 1], 1e-12, None)
        Vn, post = FL.origin_filter(chains, lp, P, Oo, M["sp"], M["xc"], M["st"], pD, dur, M["use_order"])
        return FL.forecast_filtered(chains, Vn, post, Oo.assign(cohort="NLSY97"), M["xc"], M["st"], el, M["use_order"])
    return f

def competitor_predictors(A, xc_comp):

    K.XC = xc_comp
    Am = A.copy(); Am.loc[Am.state.isin(["S", "D", "N"]), "state"] = "S"
    Am["dur_state_yrs"] = Am.dur_state_yrs.where(Am.state != "S")
    itr, iva, iobsm = cv.inner_split(Am, -1)
    Htr = sorted({h for s in LMS.values() for h in s["H"]} | {3})
    chosen = {}
    for name, (grid, make) in K.builders(itr, iobsm, Htr).items():
        if os.environ.get("STANDIN_FAST") == "1":
            grid = grid[:1]
        sc = {str(hp): cv.score_all(make(hp), iva, iobsm)[2] for hp in grid}
        chosen[name] = grid[int(np.argmin([sc[str(hp)] for hp in grid]))]
        print(f"competitor {name}: inner {sc} -> {chosen[name]}", flush=True)
    obsm = C.fit_obs_model(Am, [(cv.origins(Am, lm), H) for lm, H in cv.PRIMARY.items()])
    preds = {n: make(chosen[n]) for n, (grid, make) in K.builders(Am, obsm, Htr).items()}
    def merged(f):
        return lambda Oo, el: f(Oo.assign(state=Oo.state.replace({"N": "S"}), dur_state_yrs=Oo.dur_state_yrs.where(~Oo.state.isin(["N"]))), el)
    return {n: merged(f) for n, f in preds.items()}, chosen

def pair_table(P, O, H, obsm_fit):
    Pr = C.all_pairs(P, O, H)
    reach = C.reachable(P, O, H)
    Pr = Pr[reach[Pr.oi.to_numpy(), [H.index(h) for h in Pr.h]]]
    Pr = Pr.assign(w=C.weights(obsm_fit, O, Pr, C.OBS_FEATS))
    if os.environ.get("NLSY79_WEIGHTED") == "1":
        sw = O.weight.to_numpy(float)[Pr.oi.to_numpy()]
        Pr["w"] = Pr.w * sw / np.nanmean(sw)
    Pr = Pr[Pr["observed"] & (Pr.person_y >= 0)].reset_index(drop=True)
    Pr["household"] = O.household.to_numpy()[Pr.oi.to_numpy()]
    return Pr

def loss_rows(pp, y):
    return -np.log(np.clip(pp[np.arange(len(y)), y], 1e-12, None))

def calib_slope(y, p, w):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return float(LogisticRegression(C=1e6, max_iter=2000).fit(z[:, None], y, sample_weight=w).coef_[0, 0])

def main():
    struct = sys.argv[1]; info = sys.argv[2] if len(sys.argv) > 2 else "79"
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    os.makedirs(OUTDIR, exist_ok=True)
    A, P = load_panels()
    sfx = {"79": "79", "full": "", "79cal": "79cal"}[info]
    M = chain_models(f"{struct}{sfx}")
    Mnull = chain_models("none")
    tr_orig = pd.concat([C.landmark_nlsy(A, [2002, 2003]), C.landmark_nlsy(A, [2008, 2009])])
    pi, dur = stage_model(tr_orig)
    xc_comp = cv.XC79 if info in ("79", "79cal") else [c if c != "log_income" else "log_income_real" for c in K.XC]
    comps, chosen = competitor_predictors(A, xc_comp)
    O = {k: landmark(P, s) for k, s in LMS.items()}
    allH = {k: s["H"] + s["X"] for k, s in LMS.items()}
    obsm79 = C.fit_obs_model(P, [(O[k], allH[k]) for k in LMS])
    margins = MARGINS
    hh_all = P.drop_duplicates("pid").set_index("pid").household
    result = dict(struct=struct, info=info, standin=STANDIN, competitor_hp={k: str(v) for k, v in chosen.items()},
                  n_origins={k: len(v) for k, v in O.items()}, landmarks={})
    rng = np.random.default_rng(20261111)
    for k, spec in LMS.items():
        Pr = pair_table(P, O[k], allH[k], obsm79)
        Oo = O[k].iloc[Pr.oi.to_numpy()].reset_index(drop=True)
        el = Pr.elapsed.to_numpy(); y = Pr.person_y.to_numpy().astype(int); w = Pr.w.to_numpy()

        On = Oo.assign(cohort="NLSY97")
        pnull, _ = C.forecast(Mnull["par"], On, Mnull["xc"], Mnull["st"], el, use_order=False, stage={"pi": pi(On), "dur": dur})
        preds = {"chain": chain_predict(M, P, pi, dur)(Oo, el), "null": pnull.numpy()}
        for n, f in comps.items():
            preds[n] = f(Oo, el)
        L = {n: loss_rows(p, y) for n, p in preds.items()}
        hh = Pr.household.to_numpy(); H_ = np.unique(hh); hi = np.searchsorted(H_, hh)
        cells = {}
        for h in allH[k]:
            mh = (Pr.h == h).to_numpy()
            cells[h] = dict(n=int(mh.sum()), loss={n: float(np.average(l[mh], weights=w[mh])) for n, l in L.items()})
        boot = {h: {n: [] for n in L} for h in allH[k]}
        for _ in range(1000):
            cnt = np.bincount(rng.integers(0, len(H_), len(H_)), minlength=len(H_)); ww = w * cnt[hi]
            for h in allH[k]:
                mh = (Pr.h == h).to_numpy()
                for n, l in L.items():
                    boot[h][n].append(np.sum(ww[mh] * l[mh]) / np.sum(ww[mh]))
        for h in allH[k]:
            B = {n: np.array(v) for n, v in boot[h].items()}
            c = cells[h]
            q = lambda a: [float(np.percentile(a, 5)), float(np.percentile(a, 95))]
            c["skill"] = 1 - c["loss"]["chain"] / c["loss"]["null"]; c["skill_ci90"] = q(1 - B["chain"] / B["null"])
            c["diff"] = {n: dict(est=c["loss"]["chain"] - c["loss"][n], ci90=q(B["chain"] - B[n])) for n in L if n != "chain"}
            mh = (Pr.h == h).to_numpy()
            c["calibration_slope"] = {cls: calib_slope((y[mh] == j).astype(int), preds["chain"][mh, j], w[mh]) for j, cls in enumerate(("none", "cohab", "married"))}
            if h in spec["H"]:
                mg = margins[spec["ref"]][str(h)]
                c["E1_pass"] = c["skill_ci90"][0] > 0
                c["E2"] = {n: dict(margin=mg, pass_=c["diff"][n]["ci90"][1] < mg) for n in ("logitT", "gbmT")}
            else:
                c["E3a_pass"] = c["diff"]["null"]["ci90"][1] < 0
                best = min(("itm_gbm", "itm_logit", "phd"), key=lambda n: c["loss"][n])
                lo, hi_ = c["diff"][best]["ci90"]
                c["E3b"] = dict(best=best, verdict="advantage" if hi_ < 0 else ("worse" if lo > 0 else "comparable"))
        result["landmarks"][k] = {str(h): v for h, v in cells.items()}
        print(k, json.dumps({str(h): dict(skill=round(v["skill"], 4), chain=round(v["loss"]["chain"], 4)) for h, v in cells.items()}), flush=True)
    tag = ("_agelin" if os.environ.get("AGE_LINEAR") == "1" else "") + ("_weighted" if os.environ.get("NLSY79_WEIGHTED") == "1" else "")
    result.update(age_linear=os.environ.get("AGE_LINEAR") == "1", weighted=os.environ.get("NLSY79_WEIGHTED") == "1")
    json.dump(result, open(f"{OUTDIR}/{struct}_{info}{tag}.json", "w"), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())

if __name__ == "__main__":
    main()
