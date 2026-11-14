# Post-hoc descriptive supplements requested after review: simpler structures on NLSY79, calibration intercepts and reliability tables, a state-aware baseline for the skill curve, and realized outcomes by latent class.
import os, sys, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
import common as C, cv
import nlsy79_eval as E

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
OUT = f"{C.ROOT}/runs"
CLS = ("none", "cohab", "married")


def calib(y, p, w):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    slope = float(LogisticRegression(C=1e6, max_iter=2000).fit(z[:, None], y, sample_weight=w).coef_[0, 0])
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(z + a)))
        g = np.sum(w * (y - q)); h = np.sum(w * q * (1 - q))
        step = g / max(h, 1e-12); a += step
        if abs(step) < 1e-10:
            break
    return dict(slope=slope, intercept_given_slope1=float(a), mean_pred=float(np.average(p, weights=w)), observed=float(np.average(y, weights=w)))


def reliability(pp, y, w):
    return {cls: calib((y == j).astype(int), pp[:, j], w) for j, cls in enumerate(CLS)}


def nlsy79_structures():
    A, P = E.load_panels()
    tr_orig = pd.concat([C.landmark_nlsy(A, [2002, 2003]), C.landmark_nlsy(A, [2008, 2009])])
    pi, dur = C.stage_model(tr_orig)
    names = ["base79", "lat79", "phase79", "phase_lat79"]
    Ms = {n: E.chain_models(n) for n in names}
    Mnull = E.chain_models("none")
    O = {k: E.landmark(P, s) for k, s in E.LMS.items()}
    allH = {k: s["H"] + s["X"] for k, s in E.LMS.items()}
    obsm79 = C.fit_obs_model(P, [(O[k], allH[k]) for k in E.LMS])
    rng = np.random.default_rng(20261111)
    out, pairs_out = {}, []
    for k, spec in E.LMS.items():
        Pr = E.pair_table(P, O[k], allH[k], obsm79)
        Oo = O[k].iloc[Pr.oi.to_numpy()].reset_index(drop=True)
        el = Pr.elapsed.to_numpy(); y = Pr.person_y.to_numpy().astype(int); w = Pr.w.to_numpy()
        On = Oo.assign(cohort="NLSY97")
        pnull, _ = C.forecast(Mnull["par"], On, Mnull["xc"], Mnull["st"], el, use_order=False, stage={"pi": pi(On), "dur": dur})
        preds = {"null": pnull.numpy()}
        for n in names:
            preds[n] = E.chain_predict(Ms[n], P, pi, dur)(Oo, el)
        L = {n: E.loss_rows(p, y) for n, p in preds.items()}
        hh = Pr.household.to_numpy(); H_ = np.unique(hh); hi = np.searchsorted(H_, hh)
        boot = {h: {n: [] for n in L} for h in allH[k]}
        for _ in range(1000):
            cnt = np.bincount(rng.integers(0, len(H_), len(H_)), minlength=len(H_)); ww = w * cnt[hi]
            for h in allH[k]:
                mh = (Pr.h == h).to_numpy()
                for n, l in L.items():
                    boot[h][n].append(np.sum(ww[mh] * l[mh]) / np.sum(ww[mh]))
        q = lambda a: [float(np.percentile(a, 5)), float(np.percentile(a, 95))]
        cells = {}
        for h in allH[k]:
            mh = (Pr.h == h).to_numpy()
            loss = {n: float(np.average(l[mh], weights=w[mh])) for n, l in L.items()}
            B = {n: np.array(v) for n, v in boot[h].items()}
            cells[str(h)] = dict(n=int(mh.sum()), loss=loss,
                                 skill={n: 1 - loss[n] / loss["null"] for n in names},
                                 skill_ci90={n: q(1 - B[n] / B["null"]) for n in names},
                                 final_minus={n: dict(est=loss["phase_lat79"] - loss[n], ci90=q(B["phase_lat79"] - B[n])) for n in names[:-1]},
                                 reliability=reliability(preds["phase_lat79"][mh], y[mh], w[mh]))
        out[k] = cells
        d = Pr[["oi", "h", "person_y", "w", "household"]].copy(); d["pid"] = Oo.pid.to_numpy(); d["landmark"] = k
        for n in names + ["null"]:
            d[[f"{n}_p0", f"{n}_p1", f"{n}_p2"]] = preds[n]
        pairs_out.append(d)
        print(k, json.dumps({h: {n: round(v["skill"][n], 4) for n in names} for h, v in cells.items()}), flush=True)
    json.dump(out, open(f"{OUT}/nlsy79/structures_79.json", "w"), indent=1)
    pd.concat(pairs_out).to_csv(f"{OUT}/nlsy79/structures_79_pairs.csv", index=False)


def cv_supplement():
    def load(name):
        d = pd.concat([pd.read_csv(f"{cv.OUT}/{name}_f{k}_pairs.csv") for k in range(5)]).reset_index(drop=True)
        return d[d.person_y >= 0]
    F, N, S = load("phase_lat"), load("none"), load("none_su")
    key = ["landmark", "pid", "h"]
    M = F.merge(N[key + ["p0", "p1", "p2"]], on=key, suffixes=("", "_none")).merge(S[key + ["p0", "p1", "p2"]].rename(columns={"p0": "p0_su", "p1": "p1_su", "p2": "p2_su"}), on=key)
    y = M.person_y.astype(int).to_numpy()
    def loss(cols):
        p = M[cols].to_numpy(); return -np.log(np.clip(p[np.arange(len(M)), y], 1e-12, None))
    M["l_final"], M["l_none"], M["l_su"] = loss(["p0", "p1", "p2"]), loss(["p0_none", "p1_none", "p2_none"]), loss(["p0_su", "p1_su", "p2_su"])
    out = {}
    for (lm, h), g in M.groupby(["landmark", "h"]):
        gw = g.w.to_numpy(); yy = g.person_y.astype(int).to_numpy()
        lf, ln, ls = (float(np.average(g[c], weights=gw)) for c in ("l_final", "l_none", "l_su"))
        out[f"{lm}|{h}"] = dict(n=int(len(g)), loss_final=lf, loss_state_known_baseline=ln, loss_stage_unknown_baseline=ls,
                               skill_vs_stage_unknown=1 - lf / ls, skill_vs_state_known=1 - lf / ln,
                               share_of_total_skill_from_state=(ls - ln) / (ls - lf) if ls > lf else None,
                               reliability=reliability(g[["p0", "p1", "p2"]].to_numpy(), yy, gw))
        print(lm, h, "skill vs stage-unknown", round(out[f"{lm}|{h}"]["skill_vs_stage_unknown"], 4), "vs state-known", round(out[f"{lm}|{h}"]["skill_vs_state_known"], 4), flush=True)
    json.dump(out, open(f"{OUT}/cv/supplement_cv.json", "w"), indent=1)


def class_outcomes():
    D = pd.read_csv(f"{OUT}/class_forecasts.csv")
    A = C.load(); A = A[A.split.isin(["train", "valid"])]
    rows = []
    for lm, years in (("nlsy2002", [2002, 2003]), ("nlsy2008", [2008, 2009])):
        O = C.landmark_nlsy(A, years).reset_index(drop=True)
        d = D[D.landmark == lm].merge(O[["pid", "t_years"]], on="pid")
        obs = A[["pid", "t_years", "state"]].rename(columns={"t_years": "t_obs", "state": "state_obs"})
        m = d.merge(obs, on="pid"); m = m[(m.t_obs - m.t_years - 10).abs() <= 1.25]
        m = m.assign(dist=(m.t_obs - m.t_years - 10).abs()).sort_values("dist").groupby("pid").head(1)
        m["y"] = m.state_obs.map({"S": "none", "N": "none", "D": "none", "C": "cohab", "M": "married"})
        for (s, c), g in m.dropna(subset=["y"]).groupby(["state", "modal"]):
            if len(g) < 30: continue
            rows.append(dict(landmark=lm, origin_state=s, modal_class=int(c), n=int(len(g)),
                             obs_none=float((g.y == "none").mean()), obs_cohab=float((g.y == "cohab").mean()), obs_married=float((g.y == "married").mean()),
                             pred_none=float(g.p_none.mean()), pred_cohab=float(g.p_cohab.mean()), pred_married=float(g.p_married.mean())))
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/class_outcomes_10y.csv", index=False); print(T.round(3).to_string())


def nlsy79_calibration():
    d = pd.read_csv(f"{OUT}/nlsy79/structures_79_pairs.csv")
    out = {}
    for (lm, h), g in d.groupby(["landmark", "h"]):
        out[f"{lm}|{h}"] = dict(n=int(len(g)), reliability=reliability(g[["phase_lat79_p0", "phase_lat79_p1", "phase_lat79_p2"]].to_numpy(), g.person_y.astype(int).to_numpy(), g.w.to_numpy()))
    json.dump(out, open(f"{OUT}/nlsy79/calibration_79.json", "w"), indent=1)
    for k, v in out.items():
        print(k, {c: (round(r["slope"], 2), round(r["intercept_given_slope1"], 2), round(r["mean_pred"], 3), round(r["observed"], 3)) for c, r in v["reliability"].items()})


if __name__ == "__main__":
    {"nlsy79": nlsy79_structures, "cv": cv_supplement, "classes": class_outcomes, "calib79": nlsy79_calibration}[sys.argv[1]]()
