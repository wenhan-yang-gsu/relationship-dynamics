from pathlib import Path as _P
HERE = _P(__file__).resolve().parent
ROOT = HERE.parent
import sys, json, torch, numpy as np
sys.path.insert(0, str(ROOT) + "/model")
from dyn import *
m = torch.load(str(ROOT) + "/model/personal_model.pt", weights_only=False)
nm, st = m["names"], m["stats"]
d = load_transitions(str(ROOT) + "/dataset/relationship_five_state_dataset.csv")
te = d[d.split == "test"].sort_values(["pair_id", "wave"]).groupby("pair_id").head(1).reset_index(drop=True)
te = te[te.istate.isin(DISPLAY)].reset_index(drop=True)
X = torch.tensor(np.column_stack([np.nan_to_num((te["x_" + n].to_numpy() - st[n][0]) / st[n][1]) for n in nm]))
h = torch.tensor((te.source == "HCMST").to_numpy(float))
def Pm(dt):
    with torch.no_grad():
        return transition_probs(m["a"], m["B"], X, torch.full((len(te),), float(dt)), h, m["g"]).numpy()  # (N,7,7)
T = [0, 1, 5, 10]
n = len(INTERNAL)
p = np.zeros((len(te), n)); p[np.arange(len(te)), te.istate.map(IDX).to_numpy()] = 1
coll = np.zeros((n, 5))
for i, s in enumerate(INTERNAL):
    coll[i, DISPLAY.index(s) if s in DISPLAY else 0] = 1
cols, flows = [(p @ coll).sum(0).tolist()], []
for t0, t1 in zip(T[:-1], T[1:]):
    M = Pm(t1 - t0)
    F = np.einsum("na,nab->ab", p, M)            # expected pairs moving a -> b (internal)
    flows.append((coll.T @ F @ coll).tolist())
    p = np.einsum("na,nab->nb", p, M)
    cols.append((p @ coll).sum(0).tolist())
json.dump({"T": T, "states": DISPLAY, "cols": cols, "flows": flows, "N": len(te)}, open("alluvial.json", "w"), indent=1)
for c in cols: print(np.round(c, 1))
