# Direct and iterated learners trained and tuned in the same folds: horizon-specific, pooled-horizon, iterated transition, discrete-time, and the truncation experiment.
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from common import all_pairs, reachable, weights
import common as C
import sys, json, itertools
from cv import PRIMARY, OUT, folds, origins, score_all, score_with, inner_split

STATES = ["S", "D", "C", "M"]

def origin_features(O: pd.DataFrame, xcols, elapsed=None, means=None):

    F = pd.DataFrame(index=range(len(O)))
    O = O.reset_index(drop=True)
    for s in STATES:
        F[f"st_{s}"] = (O.state == s).astype(float)
    d = O.dur_state_yrs.to_numpy(float)
    F["dur"] = np.nan_to_num(d); F["dur_miss"] = np.isnan(d).astype(float)
    F["logdur"] = np.log1p(np.nan_to_num(d))
    for c in xcols:
        v = O[c].to_numpy(float)
        F[c] = np.nan_to_num(v, nan=means[c])
        F[c + "_m"] = np.isnan(v).astype(float)
    F["prior_union"] = np.nan_to_num(O.prior_union.to_numpy(float))
    for s in C.__dict__.get("SOURCES", ["NLSY97", "HCMST2009", "HCMST2017"]):
        F[f"src_{s}"] = (O.cohort == s).astype(float)
    F["age"] = O.age.to_numpy(float)
    F["age2"] = (F.age - 30) ** 2 / 100
    if elapsed is not None:
        F["elapsed"] = np.asarray(elapsed, float)
    return F

def training_pairs(train_panel, horizons, obsm):

    O = train_panel[(train_panel.age >= 18) & train_panel.state.isin(STATES) & (train_panel.cohort != "HCMST2009")]
    O = O.reset_index(drop=True)
    P = all_pairs(train_panel, O, horizons)
    reach = reachable(train_panel, O, horizons)
    P = P[reach[P.oi.to_numpy(), [horizons.index(h) for h in P.h]]].copy()
    P["w"] = weights(obsm, O, P, C.OBS_FEATS)
    P = P[P["observed"] & (P.person_y >= 0)].reset_index(drop=True)
    return O, P

class HorizonClassifier:
    def __init__(self, kind, hp):
        self.kind, self.hp, self.models = kind, hp, {}

    def fit(self, O, P, xcols):
        rows = O.iloc[P.oi.unique()]
        with np.errstate(all="ignore"):
            self.means = {c: (float(np.nanmean(rows[c].to_numpy(float))) if rows[c].notna().any() else 0.0) for c in xcols}
        for h, d in P.groupby("h"):
            X = origin_features(O.iloc[d.oi.to_numpy()], xcols, d.elapsed.to_numpy(), self.means)
            y, w = d.person_y.to_numpy(), d.w.to_numpy()
            if self.kind == "logit":
                sc = StandardScaler().fit(X)
                m = LogisticRegression(C=self.hp, max_iter=3000).fit(sc.transform(X), y, sample_weight=w)
                self.models[h] = (sc, m)
            else:
                lr, it = self.hp[:2]
                leaves = self.hp[2] if len(self.hp) > 2 else 15
                m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves,
                                                   l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=w)
                self.models[h] = (None, m)
        return self

    def predict(self, O, h, elapsed, xcols):
        sc, m = self.models[h]
        X = origin_features(O, xcols, elapsed, self.means)
        X = sc.transform(X) if sc is not None else X
        p = np.full((len(O), 3), 1e-6)
        p[:, m.classes_] = m.predict_proba(X)
        return p / p.sum(1, keepdims=True)

BANDS = 4

def _band(d):
    return np.clip(np.floor(np.nan_to_num(d, nan=3.0)), 0, BANDS - 1).astype(int)

def _dt_features(state, band, age, X):
    F = [np.eye(4)[state][:, :, None] * np.eye(BANDS)[band][:, None, :]]
    F = [F[0].reshape(len(state), -1)]
    a = (age - 30) / 10
    F += [a[:, None], (a ** 2)[:, None], np.eye(4)[state] * a[:, None], X]
    return np.concatenate(F, 1)

class DiscreteTime:
    def __init__(self, C_reg=1.0):
        self.C = C_reg

    def _x(self, rows, xcols, fit=False):
        X = rows[xcols + ["prior_union"]].to_numpy(float)
        if fit:
            self.mu = np.nanmean(X, 0); self.mu = np.where(np.isnan(self.mu), 0, self.mu)
            self.sd = np.nanstd(X, 0); self.sd = np.where((self.sd == 0) | np.isnan(self.sd), 1, self.sd)
        Z = (X - self.mu) / self.sd
        return np.concatenate([np.nan_to_num(Z), np.isnan(Z).astype(float)], 1)

    def fit(self, train_panel, xcols):
        n = train_panel[train_panel.cohort == "NLSY97"].sort_values(["pid", "t_years"])
        g = n.groupby("pid")
        n = n.assign(nstate=g.state.shift(-1), nt=g.t_years.shift(-1))
        d = n[n.state.isin(STATES) & n.nstate.isin(STATES) & ((n.nt - n.t_years).between(0.75, 1.25)) & (n.age >= 18)]
        st = d.state.map({s: i for i, s in enumerate(STATES)}).to_numpy()
        y = d.nstate.map({s: i for i, s in enumerate(STATES)}).to_numpy()
        self.xcols = xcols
        F = _dt_features(st, _band(d.dur_state_yrs.to_numpy()), d.age.to_numpy(float), self._x(d, xcols, fit=True))
        self.sc = StandardScaler().fit(F)
        self.m = LogisticRegression(C=self.C, max_iter=3000).fit(self.sc.transform(F), y)
        self.n_train = len(d)
        return self

    def predict(self, O, elapsed):

        O = O.reset_index(drop=True)
        N = len(O)
        X = self._x(O, self.xcols)
        st0 = O.state.map({s: i for i, s in enumerate(STATES)}).to_numpy()
        b0 = _band(O.dur_state_yrs.to_numpy())
        v = np.zeros((N, 4, BANDS)); v[np.arange(N), st0, b0] = 1
        e = np.asarray(elapsed, float)
        H = int(np.ceil(e.max()))
        hist = [v.copy()]
        age = O.age.to_numpy(float)
        for k in range(H):
            nv = np.zeros_like(v)
            for s in range(4):
                for b in range(BANDS):
                    F = _dt_features(np.full(N, s), np.full(N, b), age + k, X)
                    p = np.zeros((N, 4)); p[:, self.m.classes_] = self.m.predict_proba(self.sc.transform(F))
                    mass = v[:, s, b][:, None] * p
                    for t in range(4):
                        if t == s:
                            nv[:, t, min(b + 1, BANDS - 1)] += mass[:, t]
                        else:
                            nv[:, t, 0] += mass[:, t]
            v = nv
            hist.append(v.copy())
        hist = np.stack(hist)
        lo = np.floor(e).astype(int); hi = np.minimum(lo + 1, H); f = e - lo
        pv = (1 - f)[:, None] * hist[lo, np.arange(N)].sum(-1) + f[:, None] * hist[hi, np.arange(N)].sum(-1)
        out = np.stack([pv[:, 0] + pv[:, 1], pv[:, 2], pv[:, 3]], 1)
        return out / out.sum(1, keepdims=True)

XC = ["female", "black", "hispanic", "edu_years", "log_income", "kids_any", "rel_quality", "rel_close",
      "rel_conflict", "rel_future", "afqt", "mom_edu", "both_parents12", "parent_relig"]
LOGIT_GRID = [0.01, 0.1, 1.0]
GBM_GRID = list(itertools.product([0.05, 0.1], [100, 300], [15, 31]))
PSTATE = {"S": 0, "D": 0, "N": 0, "C": 1, "M": 2}
SRC = ["NLSY97", "HCMST2009", "HCMST2017"]
PHD_H = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]

def make_model(kind, hp):
    if kind == "logit":
        return LogisticRegression(C=hp, max_iter=3000)
    lr, it, leaves = hp
    return HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves,
                                          l2_regularization=1.0, random_state=0)

class ITM:

    def __init__(self, kind, hp, gaps=(0.25, 6.0)):
        self.kind, self.hp, self.gaps = kind, hp, gaps

    def _x(self, rows, ps, age, gap):
        X = [np.eye(3)[ps], age[:, None], ((age - 30) ** 2 / 100)[:, None], gap[:, None]]
        for c in XC:
            v = rows[c].to_numpy(float)
            X += [np.nan_to_num(v, nan=self.means[c])[:, None], np.isnan(v).astype(float)[:, None]]
        X.append(np.nan_to_num(rows.prior_union.to_numpy(float))[:, None])
        X.append(np.stack([(rows.cohort == s).to_numpy(float) for s in SRC], 1))
        return np.concatenate(X, 1)

    def fit(self, panel):
        p = panel.sort_values(["pid", "t_years"])
        g = p.groupby("pid")
        nxt, nt = g.state.shift(-1), g.t_years.shift(-1)
        first = p[p.age >= 18].groupby("pid").head(1).set_index("pid")
        ok = p.pid.isin(first.index) & p.state.isin(PSTATE.keys()) & nxt.isin(PSTATE.keys()) & (p.age >= 18) & (nt - p.t_years).between(*self.gaps)
        d = p[ok]
        rows = first.loc[d.pid].reset_index()
        rows["prior_union"] = d.prior_union.to_numpy()
        with np.errstate(all="ignore"):
            self.means = {c: (float(np.nanmean(rows[c].to_numpy(float))) if rows[c].notna().any() else 0.0) for c in XC}
        X = self._x(rows, d.state.map(PSTATE).to_numpy(), d.age.to_numpy(float), (nt[ok] - d.t_years).to_numpy())
        y = nxt[ok].map(PSTATE).to_numpy()
        self.sc = StandardScaler().fit(X) if self.kind == "logit" else None
        self.m = make_model(self.kind, self.hp).fit(self.sc.transform(X) if self.sc else X, y)
        self.n_train = len(y)
        return self

    def _step(self, rows, v, age, gap):
        out = np.zeros_like(v)
        for s in range(3):
            X = self._x(rows, np.full(len(rows), s), age, gap)
            pr = np.zeros((len(rows), 3)); pr[:, self.m.classes_] = self.m.predict_proba(self.sc.transform(X) if self.sc else X)
            out += v[:, [s]] * pr
        return out

    def predict(self, O, elapsed):
        O = O.reset_index(drop=True)
        e = np.asarray(elapsed, float)
        n = np.maximum(np.ceil(e - 1e-9), 1).astype(int)
        gap = e / n
        v = np.eye(3)[O.state.map(PSTATE).to_numpy()]
        age = O.age.to_numpy(float).copy()
        for k in range(int(n.max())):
            act = k < n
            if act.any():
                v[act] = self._step(O[act].reset_index(drop=True), v[act], age[act], gap[act])
                age[act] += gap[act]
        return v / v.sum(1, keepdims=True)

class PHD:
    def __init__(self, hp):
        self.hp = hp

    def fit(self, O, P):
        rows = O.iloc[P.oi.unique()]
        with np.errstate(all="ignore"):
            self.means = {c: (float(np.nanmean(rows[c].to_numpy(float))) if rows[c].notna().any() else 0.0) for c in XC}
        X = origin_features(O.iloc[P.oi.to_numpy()], XC, P.elapsed.to_numpy(), self.means)
        self.m = make_model("gbm", self.hp).fit(X, P.person_y.to_numpy(), sample_weight=P.w.to_numpy())
        return self

    def predict(self, O, elapsed):
        X = origin_features(O.reset_index(drop=True), XC, elapsed, self.means)
        p = np.full((len(O), 3), 1e-6); p[:, self.m.classes_] = self.m.predict_proba(X)
        return p / p.sum(1, keepdims=True)

def horizon_predict(m):
    def predict(Oo, el):
        hs = np.array(sorted(m.models))
        hh = hs[np.abs(np.asarray(el)[:, None] - hs[None, :]).argmin(1)]
        p = np.zeros((len(Oo), 3))
        for h in np.unique(hh):
            r = hh == h
            p[r] = m.predict(Oo[r].reset_index(drop=True), h, np.asarray(el)[r], XC)
        return p
    return predict

def builders(tr, obsm, Htr):

    cache = {}
    def pairs(H):
        key = tuple(H)
        if key not in cache:
            cache[key] = training_pairs(tr, list(H), obsm)
        return cache[key]
    return {
        "logitT": (LOGIT_GRID, lambda hp: horizon_predict(HorizonClassifier("logit", hp).fit(*pairs(Htr), XC))),
        "gbmT": (GBM_GRID, lambda hp: horizon_predict(HorizonClassifier("gbm", hp).fit(*pairs(Htr), XC))),
        "itm_logit": (LOGIT_GRID, lambda hp: ITM("logit", hp).fit(tr).predict),
        "itm_gbm": (GBM_GRID, lambda hp: ITM("gbm", hp).fit(tr).predict),
        "phd": (GBM_GRID, lambda hp: PHD(hp).fit(*pairs(PHD_H)).predict),
    }

def main_tuned():
    k = int(sys.argv[1])
    A = C.load()
    A = A[A.split.isin(["train", "valid"])].merge(folds(A), on="household", how="left")
    tr, ev = A[A.fold != k], A[A.fold == k]
    Htr = sorted({h for H in PRIMARY.values() for h in H})

    itr, iva, iobsm = inner_split(tr, k)
    chosen, inner = {}, {}
    for name, (grid, make) in builders(itr, iobsm, Htr).items():
        sc = {str(hp): score_all(make(hp), iva, iobsm)[2] for hp in grid}
        inner[name] = sc
        chosen[name] = grid[int(np.argmin([sc[str(hp)] for hp in grid]))]
        print(f"fold {k} {name}: inner {sc} -> {chosen[name]}", flush=True)

    obsm = C.fit_obs_model(tr, [(origins(tr, lm), H) for lm, H in PRIMARY.items()])
    for name, (grid, make) in builders(tr, obsm, Htr).items():
        res, pairs, pm = score_all(make(chosen[name]), ev, obsm)
        json.dump(dict(config=name, fold=k, hp=str(chosen[name]), inner=inner[name], scores=res, primary_mean=pm),
                  open(f"{OUT}/{name}_f{k}.json", "w"), indent=1, default=float)
        pairs.to_csv(f"{OUT}/{name}_f{k}_pairs.csv", index=False)
        print(f"fold {k} {name}: outer primary mean {pm:.4f}", flush=True)

    tt = tr[(tr.cohort == "NLSY97") & (tr.t_years + 1980 < 2008.0)]
    tobsm = C.fit_obs_model(tr, [(origins(tr, "nlsy2002"), [10, 20])])
    preds = {"logit_h5": horizon_predict(HorizonClassifier("logit", chosen["logitT"]).fit(*training_pairs(tt, [5], tobsm), XC)),
             "gbm_h5": horizon_predict(HorizonClassifier("gbm", chosen["gbmT"]).fit(*training_pairs(tt, [5], tobsm), XC)),
             "itm_logit": ITM("logit", chosen["itm_logit"]).fit(tt).predict,
             "itm_gbm": ITM("gbm", chosen["itm_gbm"]).fit(tt).predict,
             "dtime": DiscreteTime(1.0).fit(tt, XC).predict}
    Ott, Ptt = training_pairs(tt, PHD_H, tobsm)
    preds["phd"] = PHD(chosen["phd"]).fit(Ott, Ptt).predict
    print(f"fold {k} truncation: phd trained on horizons {sorted(Ptt.h.unique().tolist())}", flush=True)
    for name, f in preds.items():
        r, d = score_with(f, ev, "nlsy2002", [10, 20], tobsm)
        d.to_csv(f"{OUT}/trunc_{name}_f{k}_pairs.csv", index=False)
        json.dump(dict(config=f"trunc_{name}", fold=k, scores={"nlsy2002": r}), open(f"{OUT}/trunc_{name}_f{k}.json", "w"), indent=1, default=float)
        print(f"fold {k} trunc {name}: " + " ".join(f"h{h} {v['person']:.4f}" for h, v in r.items()), flush=True)

def main_fixed():
    k = int(sys.argv[1])
    A = C.load()
    A = A[A.split.isin(["train", "valid"])].merge(folds(A), on="household", how="left")
    tr, ev = A[A.fold != k], A[A.fold == k]
    obsm = C.fit_obs_model(tr, [(origins(tr, lm), H) for lm, H in PRIMARY.items()])
    Htr = sorted({h for H in PRIMARY.values() for h in H})
    Otr, Ptr = training_pairs(tr, Htr, obsm)
    models = {"logit": HorizonClassifier("logit", 0.1).fit(Otr, Ptr, XC),
              "gbm": HorizonClassifier("gbm", (0.05, 100)).fit(Otr, Ptr, XC)}
    dt = DiscreteTime(1.0).fit(tr, XC)

    def horizon_predict(m):
        def predict(Oo, el):

            hs = np.array(sorted(m.models))
            hh = hs[np.abs(np.asarray(el)[:, None] - hs[None, :]).argmin(1)]
            p = np.zeros((len(Oo), 3))
            for h in np.unique(hh):
                r = hh == h
                p[r] = m.predict(Oo[r].reset_index(drop=True), h, np.asarray(el)[r], XC)
            return p
        return predict

    preds = {**{n: horizon_predict(m) for n, m in models.items()}, "dtime": lambda Oo, el: dt.predict(Oo, el)}
    for name, f in preds.items():
        res, pairs, pm = score_all(f, ev, obsm)
        json.dump(dict(config=name, fold=k, scores=res, primary_mean=pm), open(f"{OUT}/{name}_f{k}.json", "w"), indent=1, default=float)
        pairs.to_csv(f"{OUT}/{name}_f{k}_pairs.csv", index=False)
        print(name, k, round(pm, 4), flush=True)

ANNUAL = (0.75, 1.25)

def main_annual():
    k = int(sys.argv[1])
    A = C.load()
    A = A[A.split.isin(["train", "valid"])].merge(folds(A), on="household", how="left")
    tr, ev = A[A.fold != k], A[A.fold == k]
    itr, iva, iobsm = inner_split(tr, k)
    sc = {str(hp): score_all(ITM("gbm", hp, ANNUAL).fit(itr).predict, iva, iobsm)[2] for hp in GBM_GRID}
    hp = GBM_GRID[int(np.argmin([sc[str(h)] for h in GBM_GRID]))]
    print(f"fold {k} itm_gbm_annual: inner {sc} -> {hp}", flush=True)
    obsm = C.fit_obs_model(tr, [(origins(tr, lm), H) for lm, H in PRIMARY.items()])
    m = ITM("gbm", hp, ANNUAL).fit(tr)
    res, pairs, pm = score_all(m.predict, ev, obsm)
    json.dump(dict(config="itm_gbm_annual", fold=k, hp=str(hp), inner=sc, n_train=m.n_train, scores=res, primary_mean=pm),
              open(f"{OUT}/itm_gbm_annual_f{k}.json", "w"), indent=1, default=float)
    pairs.to_csv(f"{OUT}/itm_gbm_annual_f{k}_pairs.csv", index=False)
    print(f"fold {k} itm_gbm_annual: transitions {m.n_train}, outer primary mean {pm:.4f}", flush=True)
    tt = tr[(tr.cohort == "NLSY97") & (tr.t_years + 1980 < 2008.0)]
    tobsm = C.fit_obs_model(tr, [(origins(tr, "nlsy2002"), [10, 20])])
    r, d = score_with(ITM("gbm", hp, ANNUAL).fit(tt).predict, ev, "nlsy2002", [10, 20], tobsm)
    d.to_csv(f"{OUT}/trunc_itm_gbm_annual_f{k}_pairs.csv", index=False)
    json.dump(dict(config="trunc_itm_gbm_annual", fold=k, scores={"nlsy2002": r}), open(f"{OUT}/trunc_itm_gbm_annual_f{k}.json", "w"), indent=1, default=float)
    print(f"fold {k} trunc itm_gbm_annual: " + " ".join(f"h{h} {v['person']:.4f}" for h, v in r.items()), flush=True)

if __name__ == "__main__":
    {"fixed": main_fixed, "tuned": main_tuned, "annual": main_annual}[sys.argv.pop(1)]()
