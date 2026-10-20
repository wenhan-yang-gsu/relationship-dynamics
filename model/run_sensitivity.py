"""5-fold CV (train people only): group model + parameter sensitivity (warm-started)."""
import json, time, itertools
import numpy as np, torch
from dyn import *

OUT = "results_cv.json"
d = load_transitions(DATA)
tr_all = d[d.split == "train"].reset_index(drop=True)
FOLDS = sorted(tr_all.cv_fold.dropna().unique())
ALL = list(CANDIDATES)
R = json.load(open(OUT)) if __import__("os").path.exists(OUT) else {}
CACHE = {}   # key -> list of (a,B,names) per fold


def init_from(src_key, names, f):
    """Warm start: copy matching columns from a cached fit, zeros elsewhere."""
    if src_key not in CACHE:
        return None
    a, B, nm = CACHE[src_key][f]
    newB = torch.zeros(len(EDGES), len(names))
    for j, n in enumerate(names):
        if n in nm:
            newB[:, j] = B[:, nm.index(n)]
    return a, newB


def cv(key, names, lam, inter=(), warm=None):
    tot = {"all": [0.0, 0], "Eastwick": [0.0, 0], "HCMST": [0.0, 0]}
    per_fold, fits = [], []
    cols = list(names) + [f"{a}*{b}" for a, b in inter]
    for fi, f in enumerate(FOLDS):
        tr, va = tr_all[tr_all.cv_fold != f], tr_all[tr_all.cv_fold == f]
        Xtr, Xva = design(tr, va, list(names), list(inter))
        m = DynModel(np.asarray(lam) if isinstance(lam, list) else lam).fit(Xtr, tr, init=init_from(warm, cols, fi) if warm else None)
        fits.append((m.a, m.B, cols))
        nll = display_nll(m.predict_display(Xva, va), va)
        per_fold.append(float(nll.mean()))
        tot["all"][0] += nll.sum(); tot["all"][1] += len(nll)
        for s in ("Eastwick", "HCMST"):
            k = (va.source == s).to_numpy()
            tot[s][0] += nll[k].sum(); tot[s][1] += k.sum()
    CACHE[key] = fits
    return {"nll": tot["all"][0] / tot["all"][1], "nll_EW": tot["Eastwick"][0] / tot["Eastwick"][1],
            "nll_HC": tot["HCMST"][0] / tot["HCMST"][1], "per_fold": per_fold,
            "names": list(names), "inter": [list(x) for x in inter], "lam": lam}


def run(key, names, lam, inter=(), warm=None, force=False):
    if key in R and not force and key in CACHE:
        return R[key]
    t = time.time()
    R[key] = cv(key, names, lam, inter, warm)
    json.dump(R, open(OUT, "w"), indent=1)
    print(f"{key:45s} nll={R[key]['nll']:.5f} EW={R[key]['nll_EW']:.5f} HC={R[key]['nll_HC']:.5f} ({time.time()-t:.0f}s)", flush=True)
    return R[key]


run("base", [], 1.0)
run("diag_base+dataset_indicator", ["is_HCMST"], 1.0, warm="base")
GRID = [(30, 3), (30, 10), (30, 30), (10, 3), (100, 3)]
for lew, lhc in GRID:
    run(f"all_lam{lew}_{lhc}", ALL, edge_lams(lew, lhc).tolist())
lew, lhc = min(GRID, key=lambda p: R[f"all_lam{p[0]}_{p[1]}"]["nll"])
lam = edge_lams(lew, lhc).tolist()
R["chosen_lam"] = lam; R["chosen_lam_pair"] = [lew, lhc]
FULL = f"all_lam{lew}_{lhc}"
print("chosen lam", lam, flush=True)
for nm in ALL:
    run(f"add_{nm}", [nm], lam, warm="base")
for nm in ALL:
    run(f"drop_{nm}", [x for x in ALL if x != nm], lam, warm=FULL)
gain = sorted(ALL, key=lambda n: R[f"add_{n}"]["nll"])
R["single_rank"] = gain
for k in (2, 3, 5, 8, 12, 18):
    run(f"top{k}", gain[:k], lam, warm="base")
cands = {f"top{k}": gain[:k] for k in (2, 3, 5, 8, 12, 18)}
cands[FULL] = ALL
best_main_key = min(cands, key=lambda k: R[k]["nll"])
best_names = cands[best_main_key]
R["best_main"] = best_main_key
top4 = gain[:4]
pairs = list(itertools.combinations(top4, 2))
names_i = list(dict.fromkeys(best_names + top4))
for p in pairs:
    run(f"inter_{p[0]}*{p[1]}", names_i, lam, [p], warm=best_main_key)
run("inter_all_top4", names_i, lam, pairs, warm=best_main_key)
run("diag_full+dataset_indicator", ALL + ["is_HCMST"], lam, warm=FULL)
json.dump(R, open(OUT, "w"), indent=1)
print("DONE", flush=True)
