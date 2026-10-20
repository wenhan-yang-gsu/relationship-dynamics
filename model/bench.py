"""Benchmarks on the same transitions: predict next displayed state directly."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from dyn import load_transitions, CANDIDATES, DISPLAY, DATA

STATES5 = ["NR", "NP", "RP", "CR", "M"]


def feats(d, names):
    X = pd.DataFrame({f"s_{s}": (d.istate == s).astype(float) for s in STATES5})
    ldt = np.log(d.dt.to_numpy())
    X["log_dt"] = ldt
    for s in STATES5:
        X[f"s_{s}_x_logdt"] = X[f"s_{s}"] * ldt
    for nm in names:
        X[nm] = d["x_" + nm].to_numpy()
    return X


def y_of(d):
    return d.dest_display.map({s: i for i, s in enumerate(DISPLAY)}).to_numpy()


def full_proba(model, X, classes):
    p = np.full((len(X), 5), 1e-6)
    p[:, classes] = model.predict_proba(X)
    return p / p.sum(1, keepdims=True)


def fit_predict(kind, tr, te, names, hp):
    Xtr, Xte, ytr = feats(tr, names), feats(te, names), y_of(tr)
    if kind == "freq":
        tab = pd.crosstab(tr.istate, tr.dest_display).reindex(index=STATES5, columns=DISPLAY, fill_value=0) + 0.5
        tab = tab.div(tab.sum(1), axis=0)
        return tab.loc[te.istate].to_numpy()
    if kind == "logistic":
        mu = Xtr.mean()
        sc = StandardScaler().fit(Xtr.fillna(mu))
        m = LogisticRegression(C=hp, max_iter=3000).fit(sc.transform(Xtr.fillna(mu)), ytr)
        return full_proba(m, sc.transform(Xte.fillna(mu)), m.classes_)
    if kind == "boosting":
        lr, it = hp
        m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
        return full_proba(m, Xte, m.classes_)


def score(p, d):
    y = y_of(d)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    ew = (d.source == "Eastwick").to_numpy()
    return dict(nll=float(nll.mean()), nll_EW=float(nll[ew].mean()), nll_HC=float(nll[~ew].mean()),
                acc=float((p.argmax(1) == y).mean()), n=len(y))


def cv(kind, names, hp, tr_all):
    ps = np.zeros((len(tr_all), 5))
    for f in sorted(tr_all.cv_fold.unique()):
        k = (tr_all.cv_fold == f).to_numpy()
        ps[k] = fit_predict(kind, tr_all[~k], tr_all[k], names, hp)
    return score(ps, tr_all)


if __name__ == "__main__":
    d = load_transitions(DATA)
    tr = d[d.split == "train"].reset_index(drop=True)
    ALL = list(CANDIDATES)
    R = {}
    R["freq"] = cv("freq", [], None, tr)
    for tag, names in (("base", []), ("all", ALL)):
        R[f"logistic_{tag}"] = min(({**cv("logistic", names, C, tr), "hp": C} for C in (0.01, 0.1, 1.0)), key=lambda r: r["nll"])
        R[f"boosting_{tag}"] = min(({**cv("boosting", names, hp, tr), "hp": hp} for hp in ((0.05, 100), (0.05, 300), (0.1, 200))), key=lambda r: r["nll"])
        print(tag, {k: round(v["nll"], 5) for k, v in R.items()}, flush=True)
    json.dump(R, open("results_bench_cv.json", "w"), indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
