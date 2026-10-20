"""Refit the chosen models on all training people and evaluate once on the test people."""
import json, numpy as np, pandas as pd
from dyn import *
import bench

d = load_transitions(DATA)
tr = d[d.split == "train"].reset_index(drop=True)
te = d[d.split == "test"].reset_index(drop=True)
CV = json.load(open("results_cv.json"))
BC = json.load(open("results_bench_cv.json"))
lam = CV["chosen_lam"]
ALL = list(CANDIDATES)

# pick the best personalised dynamic model by CV
lam = np.asarray(lam)
FULL = "all_lam%d_%d" % tuple(CV["chosen_lam_pair"])
main_keys = [k for k in CV if isinstance(CV[k], dict) and (k.startswith("top") or k == FULL)]
best_key = min(main_keys, key=lambda k: CV[k]["nll"])      # main personalised model, chosen by CV
best = CV[best_key]
inter_key = min([k for k in CV if k.startswith("inter_")], key=lambda k: CV[k]["nll"])
print("main:", best_key, round(best["nll"], 5), "| best interaction:", inter_key, round(CV[inter_key]["nll"], 5))

preds, models = {}, {}
for tag, names, inter, l in (("dyn_group", [], [], 1.0),
                              ("dyn_personal", best["names"], best["inter"], np.asarray(best["lam"])),
                              ("dyn_personal_inter", CV[inter_key]["names"], CV[inter_key]["inter"], lam),
                              ("dyn_all25", ALL, [], lam)):
    Xtr, Xte = design(tr, te, names, [tuple(x) for x in inter])
    m = DynModel(l).fit(Xtr, tr)
    preds[tag] = m.predict_display(Xte, te)
    models[tag] = (m, names, inter, Xtr)
preds["freq"] = bench.fit_predict("freq", tr, te, [], None)
for tag in ("base", "all"):
    names = [] if tag == "base" else ALL
    preds[f"logistic_{tag}"] = bench.fit_predict("logistic", tr, te, names, BC[f"logistic_{tag}"]["hp"])
    hp = BC[f"boosting_{tag}"]["hp"]
    preds[f"boosting_{tag}"] = bench.fit_predict("boosting", tr, te, names, tuple(hp) if isinstance(hp, list) else eval(hp))

y = bench.y_of(te)
nll = {k: -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)) for k, p in preds.items()}
res = {k: bench.score(p, te) for k, p in preds.items()}

# person-level bootstrap for differences vs the personalised dynamic model
rng = np.random.default_rng(0)
people = te.person_id.to_numpy()
up = np.unique(people)
pos = {p: np.where(people == p)[0] for p in up}
boot = {k: [] for k in nll if k != "dyn_personal"}
for _ in range(2000):
    idx = np.concatenate([pos[p] for p in rng.choice(up, len(up))])
    for k in boot:
        boot[k].append(nll["dyn_personal"][idx].mean() - nll[k][idx].mean())
res["diff_dyn_personal_minus"] = {k: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
res["best_key"] = best_key; res["inter_key"] = inter_key
res["n_test_people"] = int(len(up))

# rates of the group model and parameter effects (rate multipliers per 1 SD) in personalised model
mg = models["dyn_group"][0]
res["group_rates_per_year"] = {f"{a}->{b}": float(np.exp(v)) for (a, b), v in zip(EDGES, mg.a.numpy())}
mp, names, inter, _ = models["dyn_personal"]
res["main_names"] = names
labels = names + [f"{a}*{b}" for a, b in inter]
eff = pd.DataFrame(np.exp(mp.B.numpy()), index=[f"{a}->{b}" for a, b in EDGES], columns=labels)
eff.to_csv("personal_rate_multipliers.csv")
# curves from each start state for the group model
times = np.linspace(0, 10, 121)
curves = {s: mg.curve(np.zeros(0), s, times).tolist() for s in ["NR", "NP", "RP", "CR", "M"]}
json.dump({"times": times.tolist(), "states": DISPLAY, "curves": curves}, open("group_curves.json", "w"))
json.dump(res, open("results_test.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("group_rates_per_year",)}, indent=1))
