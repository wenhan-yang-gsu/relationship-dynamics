# Household cross-validation: configurations, inner selection, scoring, the inclusion rule and the stage-unknown skill baseline.
import os, sys, time, json
import numpy as np, pandas as pd, torch
from chain import Space, Params
from sequences import build, loglik, fit
from common import xcols_for
import common as C
import structures as SM
from common import stage_model

OUT = f"{C.ROOT}/runs/cv"
PRIMARY = {"nlsy2002": [1, 5, 10, 20], "nlsy2008": [1, 5, 10, 14], "hcmst2017": [3, 5]}
CONFIGS = {
    "base": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0),
    "none": dict(K=(1, 1, 1), groups="none", lam_beta=1.0),
    "lat": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, G="inner"),
    "smooth": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, smooth="inner"),
    "lat_smooth": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, smooth="inner", G="inner"),

    "phase": dict(K=(8, 8, 15), m=1, groups="G1G2G3", lam_beta=1.0),

    "phase_lat": dict(K=(8, 8, 15), m=1, groups="G1G2G3", lam_beta=1.0, G="from_lat"),

    "monthly": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, monthly=True),

    "lat_s1": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, G="from_lat", start=1),
    "lat_s2": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, G="from_lat", start=2),

    "noise": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, noise=True),

    "base79": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, xset="79"),
    "lat79": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, xset="79", G="inner"),
    "phase79": dict(K=(8, 8, 15), m=1, groups="G1G2G3", lam_beta=1.0, xset="79", base_src="base79"),
    "phase_lat79": dict(K=(8, 8, 15), m=1, groups="G1G2G3", lam_beta=1.0, xset="79", base_src="base79", G="from_lat", lat_src="lat79"),

    "base79cal": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, xset="79cal"),
    "lat79cal": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, xset="79cal", G="from_lat", lat_src="lat79"),
    "phase79cal": dict(K=(8, 8, 15), m=1, groups="G1G2G3", lam_beta=1.0, xset="79cal", base_src="base79cal"),
    "phase_lat79cal": dict(K=(8, 8, 15), m=1, groups="G1G2G3", lam_beta=1.0, xset="79cal", base_src="base79cal", G="from_lat", lat_src="lat79"),

    "trunc_base": dict(K=(1, 1, 1), groups="G1G2G3", lam_beta=1.0, trunc=2008.0),
    "trunc_none": dict(K=(1, 1, 1), groups="none", lam_beta=1.0, trunc=2008.0),
}
TRUNC_SCORE = {"nlsy2002": [10, 20]}

def make_space(cfg):
    m = cfg.get("m", 1)
    sp = Space(K={s: k * m for s, k in zip("DCM", cfg["K"])})
    sp.prog_rate = float(m)
    return sp
G_CHOICES = [1, 2, 3]
XC79 = ["female", "black", "hispanic", "edu_years", "log_income_real", "kids_any", "afqt", "mom_edu", "both_parents12"]
S_CHOICES = [1.0, 10.0, 100.0]

def folds(A):
    path = f"{C.ROOT}/cache/cv_folds.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    hh = np.sort(A[A.split.isin(["train", "valid"])].household.unique())
    rng = np.random.default_rng(20261111)
    perm = rng.permutation(len(hh))
    f = pd.DataFrame(dict(household=hh[perm], fold=np.arange(len(hh)) % 5))
    f.to_csv(path, index=False)
    return f

def origins(df, name):
    if name == "nlsy2002":
        return C.landmark_nlsy(df, [2002, 2003])
    if name == "nlsy2008":
        return C.landmark_nlsy(df, [2008, 2009])
    return C.origins_hcmst(df, "HCMST2017")

def score_with(predict, ev, lm, H, obsm):

    O = origins(ev, lm).reset_index(drop=True)
    P = C.all_pairs(ev, O, H)
    reach = C.reachable(ev, O, H)
    P = P[reach[P.oi.to_numpy(), [H.index(h) for h in P.h]]]
    P = P.assign(w=C.weights(obsm, O, P, C.OBS_FEATS))
    Pobs = P[P["observed"] & (P.person_y >= 0)].reset_index(drop=True)
    Oo = O.iloc[Pobs.oi.to_numpy()].reset_index(drop=True)
    pp = predict(Oo, Pobs.elapsed.to_numpy())
    r = {h: dict(person=C.log_loss_person(pp[m], Pobs.person_y.to_numpy()[m], Pobs.w.to_numpy()[m]), n_person=int(m.sum()))
         for h in H for m in [(Pobs.h == h).to_numpy()]}
    d = Pobs[["oi", "h", "person_y", "w"]].copy()
    d["pid"] = Oo.pid.to_numpy(); d["landmark"] = lm
    d[["p0", "p1", "p2"]] = pp
    return r, d

def fit_config(tr, cfg, threads, lam_s=None, init=None):

    torch.set_num_threads(threads)
    xc = xcols_for(cfg["groups"]) + (SM.smooth_names() if cfg.get("smooth") else []) + (["noise_x"] if cfg.get("noise") else [])
    if cfg.get("xset") == "79":
        xc = XC79
    elif cfg.get("xset") == "79cal":
        xc = XC79 + ["birth_year"]
    sp = make_space(cfg)
    if cfg.get("monthly"):
        import structures as MO
        tr = MO.augment(tr)
    b, st = build(tr, sp, xc, use_order="G1" in cfg["groups"])
    par = Params(sp, p=b.x0.shape[1], n_src=3, init=init)
    par.age_range = (float(b.age0.min()), float(b.iv_agemid.max()))
    if init is None:
        with torch.no_grad():
            par.alpha[:] = -1.5
    t = time.time()
    fit(par, b, pen=pen_for(cfg, xc, lam_s), maxiter=300, verbose=False)
    return par, st, xc, b, time.time() - t

def pen_for(cfg, xc, lam_s):

    pen = dict(lam_beta=cfg["lam_beta"], lam_gamma=1, lam_rw=float(cfg.get("m", 1)))
    if cfg.get("smooth"):
        pen["beta_w"] = SM.beta_weights(xc, lam_s)
    return pen

def inner_split(tr, k):
    hh = np.sort(tr.household.unique())
    rng = np.random.default_rng(20261117 if k < 0 else 20261112 + k)
    inner_va = set(rng.choice(hh, int(0.2 * len(hh)), replace=False))
    itr, iva = tr[~tr.household.isin(inner_va)], tr[tr.household.isin(inner_va)]
    return itr, iva, C.fit_obs_model(itr, [(origins(itr, lm), H) for lm, H in PRIMARY.items()])

def select_lam_s(tr, cfg, threads, k, log):
    itr, iva, obsm = inner_split(tr, k)
    scores = {}
    for ls in S_CHOICES:
        par, st, xc, b, _ = fit_config(itr, cfg, threads, ls)
        scores[ls] = score_all(chain_predict(par, xc, st), iva, obsm)[2]
        log(f"  inner lam_s={ls}: {scores[ls]:.5f}")
    return min(scores, key=scores.get), scores

def score_all(predict, ev, obsm):
    res, pairs = {}, []
    for lm, H in PRIMARY.items():
        r, d = score_with(predict, ev, lm, H, obsm)
        res[lm] = r; pairs.append(d)
    pm = float(np.mean([res[lm][h]["person"] for lm, H in PRIMARY.items() for h in H]))
    return res, pd.concat(pairs), pm

def mixture_predict(m, panel, xc, st):
    import structures as LT
    def predict(Oo, el):
        post = LT.origin_posterior(m, panel, Oo, m.par.sp, xc, st)
        return LT.forecast_mixture(m, Oo, xc, st, el, post)
    return predict

def chain_predict(par, xc, st, use_order=True):
    return lambda Oo, el: C.forecast(par, Oo, xc, st, el, use_order=use_order)[0].numpy()

def select_G(tr, cfg, threads, k, log, lam_s=None):

    import structures as LT
    itr, iva, obsm = inner_split(tr, k)
    par, st, xc, b, _ = fit_config(itr, cfg, threads, lam_s)
    scores = {1: score_all(chain_predict(par, xc, st), iva, obsm)[2]}
    for G in G_CHOICES[1:]:
        m = LT.Mixture(Params(par.sp, p=par.p, n_src=3, init=par.state_dict()), G)
        m.par.age_range = par.age_range
        LT.fit_mixture(m, b, xc, pen_for(cfg, xc, lam_s))
        scores[G] = score_all(mixture_predict(m, iva, xc, st), iva, obsm)[2]
        log(f"  inner G={G}: {scores[G]:.5f} (G=1 {scores[1]:.5f})")
    return min(scores, key=scores.get), scores

def main_fit():
    name, k = sys.argv[1], int(sys.argv[2])
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    cfg = CONFIGS[name]
    os.makedirs(OUT, exist_ok=True)
    log = lambda msg: print(msg, flush=True)
    A = C.load()
    F = folds(A)
    A = A[A.split.isin(["train", "valid"])].merge(F, on="household", how="left")
    assert A.fold.notna().all()
    if cfg.get("noise"):
        pids = np.sort(A.pid.unique())
        A["noise_x"] = A.pid.map(dict(zip(pids, np.random.default_rng(20261120).standard_normal(len(pids)))))
    extra = {}
    lam_s = None
    if cfg.get("smooth"):

        A, _ = SM.add_smooth(A, SM.knots_from(A[A.fold != k]))
    if cfg.get("xset") in ("79", "79cal"):
        A = C.add_real_income(A)
        A["birth_year"] = 1980 + A.t_years - A.age
    tr, ev = (A, A.iloc[:0]) if k < 0 else (A[A.fold != k], A[A.fold == k])
    if cfg.get("trunc"):
        tt = tr[(tr.cohort == "NLSY97") & (tr.t_years + 1980 < cfg["trunc"])]
        par, st, xc, b, secs = fit_config(tt, cfg, threads)
        tobsm = C.fit_obs_model(tr, [(origins(tr, "nlsy2002"), [10, 20])])
        r, d = score_with(chain_predict(par, xc, st, use_order="G1" in cfg["groups"]), ev, "nlsy2002", [10, 20], tobsm)
        torch.save({"par": par.state_dict(), "stats": st, "cfg": cfg, "xcols": xc, "fold": k, "fit_seconds": secs,
                    "age_range": par.age_range}, f"{OUT}/{name}_f{k}.pt")
        json.dump(dict(config=name, cfg=cfg, fold=k, fit_seconds=secs, scores={"nlsy2002": r}, age_range=par.age_range),
                  open(f"{OUT}/{name}_f{k}.json", "w"), indent=1, default=float)
        d.to_csv(f"{OUT}/{name}_f{k}_pairs.csv", index=False)
        log(f"{name} fold {k}: fit {secs:.0f}s; " + " ".join(f"h{h} {v['person']:.4f}" for h, v in r.items()))
        return
    if cfg.get("smooth"):
        lam_s, inner_s = select_lam_s(tr, cfg, threads, k, log)
        extra.update(lam_s=lam_s, inner_s=inner_s)
        log(f"{name} fold {k}: lam_s = {lam_s}")

    core = {kk: v for kk, v in cfg.items() if kk not in ("G", "start", "lat_src")}
    src_name = next((n for n, c in CONFIGS.items() if c == core), None)
    base_path = f"{OUT}/{src_name}_f{k}.pt"
    t0 = time.time()
    if cfg.get("G") is not None and not cfg.get("smooth") and src_name and os.path.exists(base_path):
        ck = torch.load(base_path, weights_only=False)
        xc, st = ck["xcols"], ck["stats"]
        sp = make_space(cfg)
        b, _ = build(tr, sp, xc, st, use_order="G1" in cfg["groups"])
        par = Params(sp, p=b.x0.shape[1], n_src=3, init=ck["par"]); par.age_range = ck["age_range"]
        log(f"{name} fold {k}: warm start from {base_path}")
    else:
        init = None
        bsrc = cfg.get("base_src", "base")
        if tuple(cfg["K"]) != (1, 1, 1) and not cfg.get("smooth") and os.path.exists(f"{OUT}/{bsrc}_f{k}.pt"):
            init = torch.load(f"{OUT}/{bsrc}_f{k}.pt", weights_only=False)["par"]
            log(f"{name} fold {k}: phases start from {bsrc}_f{k} (equal rates across phases)")
        par, st, xc, b, _ = fit_config(tr, cfg, threads, lam_s, init)
    predict = chain_predict(par, xc, st, use_order="G1" in cfg["groups"])
    state = {"par": par.state_dict()}
    if cfg.get("G") is not None:
        import structures as LT
        if cfg["G"] == "inner":
            G, inner = select_G(tr, cfg, threads, k, log, lam_s)
        elif cfg["G"] == "from_lat":
            G, inner = int(json.load(open(f"{OUT}/{cfg.get('lat_src', 'lat')}_f{k}.json"))["G"]), None
        else:
            G, inner = cfg["G"], None
        extra.update(G=G, inner=inner)
        log(f"{name} fold {k}: G = {G}")
        if G > 1:
            m = LT.Mixture(par, G, delta=LT.random_start(G, 20261100 + 10 * k + cfg["start"]) if cfg.get("start") else None)
            LT.fit_mixture(m, b, xc, pen_for(cfg, xc, lam_s))
            extra["pen_obj"] = LT.objective(m, b, xc, pen_for(cfg, xc, lam_s))
            if cfg.get("start"):

                ck0 = torch.load(f"{OUT}/lat_f{k}.pt", weights_only=False)
                m0 = LT.Mixture(Params(par.sp, p=par.p, n_src=3, init=ck0["par"]), G, delta=ck0["delta"], W=ck0["W"])
                extra["pen_obj_lat"] = LT.objective(m0, b, xc, pen_for(cfg, xc, lam_s))
                log(f"{name} fold {k}: objective {extra['pen_obj']:.3f} vs original start {extra['pen_obj_lat']:.3f}")
            predict = mixture_predict(m, ev, xc, st)
            state = m.state()
            with torch.no_grad():
                extra["class_shares"] = torch.softmax(m.log_prior(LT.seq_z(b, xc)), 1).mean(0).tolist()
                extra["delta_list"] = m.delta.tolist()
    secs = time.time() - t0
    log(f"{name} fold {k}: fit {secs:.0f}s, sequences {len(b.s0)}")
    torch.save({**state, "stats": st, "cfg": cfg, "xcols": xc, "fold": k, "fit_seconds": secs,
                "age_range": par.age_range, **extra}, f"{OUT}/{name}_f{k}.pt")
    if k < 0:
        json.dump(dict(config=name, cfg=cfg, fold=k, fit_seconds=secs, **extra), open(f"{OUT}/{name}_f{k}.json", "w"), indent=1, default=float)
        return
    obsm = C.fit_obs_model(tr, [(origins(tr, lm), H) for lm, H in PRIMARY.items()])
    res, pairs, pm = score_all(predict, ev, obsm)
    out = dict(config=name, cfg=cfg, fold=k, fit_seconds=secs, scores=res, primary_mean=pm, **extra)
    json.dump(out, open(f"{OUT}/{name}_f{k}.json", "w"), indent=1, default=float)
    pairs.to_csv(f"{OUT}/{name}_f{k}_pairs.csv", index=False)
    log(json.dumps({lm: {h: round(res[lm][h]["person"], 4) for h in H} for lm, H in PRIMARY.items()}))

RULE_MEAN, RULE_WORST = 0.003, 0.003

def load_pairs(name):
    d = pd.concat([pd.read_csv(f"{OUT}/{name}_f{k}_pairs.csv") for k in range(5)])
    d = d[d.person_y >= 0]
    p = d[["p0", "p1", "p2"]].to_numpy()
    d["loss"] = -np.log(np.clip(p[np.arange(len(d)), d.person_y.astype(int)], 1e-12, None))
    return d[["landmark", "pid", "h", "w", "loss"]]

def cells(d):
    return {(lm, h): float(np.average(g.loss, weights=g.w)) for (lm, h), g in d.groupby(["landmark", "h"])}

def main_compare():
    ref, cands = sys.argv[1], sys.argv[2:]
    A = C.load()
    hh = A.drop_duplicates("pid").set_index("pid").household
    R = load_pairs(ref)
    out = {}
    for cand in cands:
        D = load_pairs(cand)
        M = R.merge(D, on=["landmark", "pid", "h"], suffixes=("_r", "_c"))
        assert len(M) == len(R) == len(D), (len(M), len(R), len(D))

        assert np.allclose(M.w_r, M.w_c, rtol=2e-2), float((np.abs(M.w_r - M.w_c) / M.w_r).max())
        print(f"{cand}: max relative IPCW weight difference to {ref} {float((np.abs(M.w_r - M.w_c) / M.w_r).max()):.1e}")
        M["hh"] = M.pid.map(hh)

        keys = [(lm, h) for lm, H in PRIMARY.items() for h in H if ((R.landmark == lm) & (R.h == h)).any()]
        keys += [k for k in sorted(set(zip(R.landmark, R.h))) if k not in keys]
        diff = {}
        for key in keys:
            g = M[(M.landmark == key[0]) & (M.h == key[1])]
            diff[key] = float(np.average(g.loss_c - g.loss_r, weights=g.w_r))

        H_ = M.hh.unique()
        rng = np.random.default_rng(1)
        idx = {h: i for i, h in enumerate(H_)}
        M["hi"] = M.hh.map(idx)
        boots = []
        for _ in range(1000):
            cnt = np.bincount(rng.integers(0, len(H_), len(H_)), minlength=len(H_))
            ww = M.w_r.to_numpy() * cnt[M.hi.to_numpy()]
            row = []
            for key in keys:
                m = ((M.landmark == key[0]) & (M.h == key[1])).to_numpy()
                row.append(np.sum(ww[m] * (M.loss_c - M.loss_r).to_numpy()[m]) / np.sum(ww[m]))
            boots.append(row)
        B = np.array(boots)
        mean_gain = -float(np.mean(list(diff.values())))
        worst = float(max(diff.values()))
        keep = mean_gain >= RULE_MEAN and worst <= RULE_WORST
        res = {"mean_gain_nats": mean_gain, "mean_gain_ci90": [float(-np.percentile(B.mean(1), 95)), float(-np.percentile(B.mean(1), 5))],
               "worst_cell_change": worst, "keep": keep,
               "cells": {f"{lm}_h{h}": dict(ref=cells(R)[(lm, h)], cand=cells(D)[(lm, h)], diff=diff[(lm, h)],
                                           ci90=[float(np.percentile(B[:, i], 5)), float(np.percentile(B[:, i], 95))])
                         for i, (lm, h) in enumerate(keys)}}
        out[cand] = res
        print(f"{cand} vs {ref}: mean gain {mean_gain:+.4f} nats (90% CI {res['mean_gain_ci90'][0]:+.4f}, "
              f"{res['mean_gain_ci90'][1]:+.4f}); worst cell {worst:+.4f}; keep = {keep}")
        for kk, v in res["cells"].items():
            print(f"   {kk:16s} ref {v['ref']:.4f}  cand {v['cand']:.4f}  diff {v['diff']:+.4f}  [{v['ci90'][0]:+.4f}, {v['ci90'][1]:+.4f}]")
    json.dump(out, open(f"{OUT}/compare_{ref}__{'_'.join(cands)}.json", "w"), indent=1)

def main_null():

    import torch, numpy as np, pandas as pd
    import common as C
    from chain import Params
    torch.set_num_threads(4)
    A = C.load(); A = A[A.split.isin(["train", "valid"])].merge(folds(A), on="household", how="left")
    for k in range(5):
        tr, ev = A[A.fold != k], A[A.fold == k]
        ck = torch.load(f"{OUT}/none_f{k}.pt", weights_only=False)
        sp = make_space(ck["cfg"]); xc, st = ck["xcols"], ck["stats"]
        par = Params(sp, p=len(xc) * 2, n_src=3, init=ck["par"]); par.age_range = ck["age_range"]
        pi, dur = stage_model(pd.concat([C.landmark_nlsy(tr, [2002, 2003]), C.landmark_nlsy(tr, [2008, 2009]), C.origins_hcmst(tr, "HCMST2017")]))
        predict = lambda Oo, el: C.forecast(par, Oo, xc, st, el, use_order=False, stage={"pi": pi(Oo), "dur": dur})[0].numpy()
        obsm = C.fit_obs_model(tr, [(origins(tr, lm), H) for lm, H in PRIMARY.items()])
        res, pairs, pm = score_all(predict, ev, obsm)
        pairs.to_csv(f"{OUT}/none_su_f{k}_pairs.csv", index=False)
        print(k, {lm: {h: round(res[lm][h]["person"], 4) for h in H} for lm, H in PRIMARY.items()}, flush=True)

if __name__ == "__main__":
    {"fit": main_fit, "compare": main_compare, "null": main_null}[sys.argv.pop(1)]()
