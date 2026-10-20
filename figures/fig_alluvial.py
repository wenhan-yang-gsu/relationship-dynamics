from pathlib import Path as _P
HERE = _P(__file__).resolve().parent
ROOT = HERE.parent
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
plt.rcParams["font.family"] = "Carlito"
INK, MUTED = "#222222", "#6B6B6B"
SC = {"NR": "#8C8C8C", "NP": "#1baf7a", "RP": "#eda100", "CR": "#2a78d6", "M": "#e87ba4"}
LAB = {"NR": "no tie", "NP": "friends", "RP": "potential", "CR": "couple", "M": "married"}
A = json.load(open(str(HERE) + "/alluvial.json"))
S = A["states"]; ORDER = ["NP", "RP", "NR", "CR", "M"]          # top -> bottom
idx = [S.index(s) for s in ORDER]
cols = [np.array(c)[idx] for c in A["cols"]]
flows = [np.array(F)[np.ix_(idx, idx)] for F in A["flows"]]
W, H = 9.0, 4.9
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
X = [1.75, 3.85, 5.95, 8.05]; bw = 0.22; gap = 0.16; top = H - 0.75
k = (top - 0.15 - gap * 4) / A["N"]                    # inches per pair

def layout(c):
    y, out = top, []
    for v in c:
        out.append((y - v * k, y)); y -= v * k + gap
    return out

L = [layout(c) for c in cols]
def band(x0, x1, a0, a1, b0, b1, color):
    t = np.linspace(0, 1, 40); s = 3 * t**2 - 2 * t**3
    xs = x0 + (x1 - x0) * t
    up = a1 + (b1 - a1) * s; lo = a0 + (b0 - a0) * s
    ax.add_patch(Polygon(np.r_[np.c_[xs, up], np.c_[xs[::-1], lo[::-1]]], closed=True, fc=color, ec="none", alpha=0.42, zorder=1))

for c in range(3):
    F = flows[c]; out_ptr = [L[c][i][1] for i in range(5)]; in_ptr = [L[c + 1][j][1] for j in range(5)]
    for i in range(5):
        for j in range(5):
            v = F[i, j] * k
            if F[i, j] < 1.0: continue
            band(X[c] + bw / 2, X[c + 1] - bw / 2, out_ptr[i] - v, out_ptr[i], in_ptr[j] - v, in_ptr[j], SC[ORDER[i]])
            out_ptr[i] -= v; in_ptr[j] -= v
for c in range(4):
    for i, s in enumerate(ORDER):
        lo, hi = L[c][i]
        ax.add_patch(Rectangle((X[c] - bw / 2, lo), bw, hi - lo, fc=SC[s], ec="white", lw=1, zorder=3))
    ax.text(X[c], H - 0.12, ["Today", "1 year", "5 years", "10 years"][c], ha="center", va="top", fontsize=26,
            weight="bold", color=INK)
for i, s in enumerate(ORDER):
    lo, hi = L[0][i]
    ax.text(X[0] - bw / 2 - 0.12, (lo + hi) / 2, f"{LAB[s]}  {cols[0][i]:.0f}", ha="right", va="center", fontsize=24, color=INK)
    lo, hi = L[3][i]
    ax.text(X[3] + bw / 2 + 0.1, (lo + hi) / 2, f"{cols[3][i]:.0f}", ha="left", va="center", fontsize=24, color=INK)
fig.savefig(str(HERE) + "/figures/alluvial.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
