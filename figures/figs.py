"""Result figures for the relationship dynamics model."""
from pathlib import Path as _P
HERE = _P(__file__).resolve().parent
ROOT = HERE.parent
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for f in ("Regular", "Bold", "Italic"):
    fm.fontManager.addfont(f"/usr/share/fonts/truetype/crosextra/Carlito-{f}.ttf")
plt.rcParams.update({
    "font.family": "Carlito", "font.size": 28, "mathtext.fontset": "stix",
    "axes.edgecolor": "#8C8C8C", "axes.labelcolor": "#222222", "xtick.color": "#6B6B6B",
    "ytick.color": "#6B6B6B", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0, "xtick.labelsize": 28, "ytick.labelsize": 28, "savefig.dpi": 300,
})
INK, GREY, LGREY, MUTED = "#222222", "#8C8C8C", "#B5B5B5", "#6B6B6B"
PRIMARY, SECONDARY, WARN = "#1F4E79", "#C07A1E", "#B03A2E"
# one colour per state
SC = {"NR": "#8C8C8C", "NP": "#1baf7a", "RP": "#eda100", "CR": "#2a78d6", "M": "#e87ba4", "D": "#5A5A5A"}
NAME = {"NR": "No\nrelationship", "NP": "No\npotential", "RP": "Romantic\npotential",
        "CR": "Committed", "M": "Married"}
OUT = str(HERE) + "/figures/"
import os; os.makedirs(OUT, exist_ok=True)
RES = json.load(open(str(ROOT) + "/model/results_test.json"))


def sep(ax):
    ax.spines["left"].set_position(("outward", 10)); ax.spines["bottom"].set_position(("outward", 10))


def save(fig, name):
    fig.savefig(OUT + name, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    from PIL import Image as _I
    _I.open(OUT + name).convert("RGB").save(OUT + name)


def tint(hexc, a=0.16):
    c = np.array(matplotlib.colors.to_rgb(hexc))
    return tuple(1 - a * (1 - c))


SHORT = {"NR": "no tie", "NP": "friends", "RP": "potential", "CR": "couple", "M": "married", "D": "died",
         "NP+RP": "know"}


def box2(ax, x, y, s, w=1.35, h=1.1, fs=30):
    """box with the code and its short name inside"""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=tint(SC[s]), ec=SC[s], lw=3.5, zorder=3))
    ax.text(x, y + 0.2, s, ha="center", va="center", fontsize=fs, weight="bold", color=INK, zorder=4)
    ax.text(x, y - 0.24, SHORT[s], ha="center", va="center", fontsize=28, color=MUTED, zorder=4)




def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def arrow(ax, p, q, lw=3, color=GREY, rad=0.0, ls="-", shrinkA=0, shrinkB=0, z=2):
    a = FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle=f"-|>,head_length={0.25+lw*0.03},head_width={0.13+lw*0.02}",
                        mutation_scale=28, lw=lw, color=color, ls=ls, shrinkA=shrinkA, shrinkB=shrinkB, zorder=z,
                        capstyle="round", joinstyle="round")
    ax.add_patch(a)


# ---------------------------------------------------------------- 1. rate diagram
from matplotlib.path import Path
ROW = ["NR", "NP", "RP", "CR", "M"]
BW, BH, YR = 1.52, 1.05, 2.3
XS = dict(zip(ROW, [0.8, 2.7, 4.6, 6.5, 8.4]))
TOP, BOT = YR + BH / 2, YR - BH / 2
DW, DH = 1.2, 0.95
DY = TOP + 0.55 + DH / 2          # D sits above M


def rate_label(ax, xy, r, learned, va="center", ha="center"):
    if not learned:
        return
    lab = f"{r:.2f}" if r < 1 else f"{r:.1f}"
    ax.text(*xy, lab, ha=ha, va=va, fontsize=28, color=INK,
            bbox=dict(fc="white", ec="none", boxstyle="square,pad=0.08"), zorder=6)


def lw_of(r, learned):
    return float(np.clip(1.6 + 1.9 * np.log10(r / 0.005), 1.6, 8.0)) if learned else 2.6


def rounded(pts, r=0.18):
    verts, codes = [pts[0]], [Path.MOVETO]
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = map(np.array, (pts[i - 1], pts[i], pts[i + 1]))
        d1 = (p1 - p0) / np.linalg.norm(p1 - p0); d2 = (p2 - p1) / np.linalg.norm(p2 - p1)
        verts += [tuple(p1 - d1 * r), tuple(p1), tuple(p1 + d2 * r)]; codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(pts[-1]); codes.append(Path.LINETO)
    return Path(verts, codes)


def lane(ax, pts, lw, color):
    pa = FancyArrowPatch(path=rounded(pts), arrowstyle=f"-|>,head_length={0.25+lw*0.03},head_width={0.13+lw*0.02}",
                         mutation_scale=28, lw=lw, color=color, zorder=2, joinstyle="round", capstyle="round")
    ax.add_patch(pa)


def fig_rates(learned=True):
    W, H = 9.2, 4.6
    fig, ax = canvas(W, H)
    R = RES["group_rates_per_year"]
    for s_ in ROW:
        box2(ax, XS[s_], YR, s_, w=BW, h=BH)
    box2(ax, XS["M"], DY, "D", w=DW, h=DH)
    col = (lambda s_: SC[s_]) if learned else (lambda s_: LGREY)
    g = 0.05
    for a_, b_ in [("NR", "NP"), ("NP", "RP"), ("RP", "CR"), ("CR", "M")]:
        y = YR + 0.2
        p, q = (XS[a_] + BW / 2 + g, y), (XS[b_] - BW / 2 - g, y)
        r = R[f"{a_}->{b_}"]; arrow(ax, p, q, lw=lw_of(r, learned), color=col(a_))
        rate_label(ax, ((p[0] + q[0]) / 2, TOP + 0.06), r, learned, va="bottom")
    for a_, b_ in [("NP", "NR"), ("RP", "NP")]:
        y = YR - 0.22
        p, q = (XS[a_] - BW / 2 - g, y), (XS[b_] + BW / 2 + g, y)
        r = R[f"{a_}->{b_}"]; arrow(ax, p, q, lw=lw_of(r, learned), color=col(a_))
        rate_label(ax, ((p[0] + q[0]) / 2, BOT - 0.06), r, learned, va="top")
    for s_, key, depth, land in [("RP", "RP->NR", 0.72, 0.35), ("CR", "CR->NRe", 1.1, 0.0), ("M", "M->NRe", 1.48, -0.35)]:
        yl = BOT - depth
        pts = [(XS[s_], BOT - g), (XS[s_], yl), (XS["NR"] + land, yl), (XS["NR"] + land, BOT - g)]
        lane(ax, pts, lw_of(R[key], learned), col(s_))
        rate_label(ax, ((XS[s_] + XS["NR"] + land) / 2 + 0.3, yl), R[key], learned)
    # NR -> RP over the row
    yl = TOP + 0.85
    pts = [(XS["NR"], TOP + g), (XS["NR"], yl), (XS["RP"] - 0.3, yl), (XS["RP"] - 0.3, TOP + g)]
    lane(ax, pts, lw_of(R["NR->RP"], learned), col("NR")); rate_label(ax, (XS["NP"], yl), R["NR->RP"], learned)
    # deaths: CR -> D (lane), M -> D (straight up)
    pts = [(XS["CR"], TOP + g), (XS["CR"], DY), (XS["M"] - DW / 2 - g, DY)]
    lane(ax, pts, lw_of(R["CR->D"], learned), col("CR")); rate_label(ax, ((XS["CR"] + XS["M"] - DW / 2) / 2 + 0.05, DY), R["CR->D"], learned)
    arrow(ax, (XS["M"], TOP + g), (XS["M"], DY - DH / 2 - g), lw=lw_of(R["M->D"], learned), color=col("M"))
    rate_label(ax, (XS["M"] + 0.12, (TOP + DY - DH / 2) / 2), R["M->D"], learned, ha="left")
    if not learned:
        y = YR + 0.2
        p, q = (XS["RP"] + BW / 2 + g, y), (XS["CR"] - BW / 2 - g, y)
        arrow(ax, p, q, lw=5, color=PRIMARY)
        ax.text((p[0] + q[0]) / 2, TOP + 0.38, r"$q_{\mathrm{RP}\to\mathrm{CR}}(x)$", ha="center", va="center",
                fontsize=32, color=INK, bbox=dict(fc="white", ec="none", boxstyle="square,pad=0.05"), zorder=6)
    save(fig, "rates_learned.png" if learned else "rates_symbolic.png")


# ---------------------------------------------------------------- 2. model comparison (dot plot)
def fig_compare():
    rows = [("Frequency baseline", "freq", LGREY), ("Logistic, state only", "logistic_base", LGREY),
            ("Group dynamic model", "dyn_group", LGREY), ("Gradient boosting", "boosting_all", LGREY),
            ("Logistic regression", "logistic_all", SECONDARY), ("Personal dynamic model", "dyn_personal", PRIMARY)]
    fig, ax = plt.subplots(figsize=(9.0, 4.4), layout="constrained")
    for i, (lab, k, c) in enumerate(rows):
        v = RES[k]["nll"]
        ax.plot([0.39, v], [i, i], color="#E6E6E6", lw=2, zorder=1)
        ax.scatter(v, i, s=330, color=c, zorder=3, edgecolor="white", linewidth=2)
        ax.text(v + 0.0040, i, f"{v:.3f}", va="center", ha="left", fontsize=28,
                color=INK, weight="bold" if k == "dyn_personal" else "normal")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=28, color=INK)
    ax.set_xlim(0.39, 0.485); ax.set_xticks([0.40, 0.42, 0.44, 0.46, 0.48])
    ax.set_xlabel("Test log loss (lower is better)", fontsize=28)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.invert_yaxis(); ax.set_ylim(len(rows) - 0.4, -0.6)
    save(fig, "compare.png")


# ---------------------------------------------------------------- 3. calibration (HCMST committed couples, test set)
def fig_calib():
    pred = [0.051, 0.087, 0.134, 0.223, 0.454]; obs = [0.022, 0.075, 0.109, 0.237, 0.473]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), layout="constrained")
    ax.plot([0, 0.55], [0, 0.55], color=LGREY, lw=2, ls=(0, (4, 3)), zorder=1)
    ax.text(0.40, 0.49, "perfect", rotation=0, color=MUTED, fontsize=28, ha="right")
    ax.scatter(pred, obs, s=380, color=PRIMARY, edgecolor="white", linewidth=2, zorder=3)
    ax.set_xlim(0, 0.55); ax.set_ylim(0, 0.55)
    ax.set_xticks([0, 0.25, 0.5]); ax.set_yticks([0, 0.25, 0.5])
    ax.set_xticklabels(["0%", "25%", "50%"]); ax.set_yticklabels(["0%", "25%", "50%"])
    ax.set_xlabel("Predicted breakup"); ax.set_ylabel("Observed breakup"); sep(ax)
    save(fig, "calibration.png")


# ---------------------------------------------------------------- 4. curves
def line_end_labels(ax, t, ys, labels, colors, xpad=0.15, min_gap=0.11):
    """direct labels at the right end, nudged apart with thin leaders"""
    order = np.argsort([y[-1] for y in ys])
    placed = []
    for i in order:
        y = ys[i][-1]
        yl = max(y, placed[-1] + min_gap) if placed else y
        placed.append(yl)
        ax.plot([t[-1], t[-1] + xpad * 0.6], [y, yl], color=colors[i], lw=1, clip_on=False)
        ax.text(t[-1] + xpad, yl, labels[i], va="center", ha="left", fontsize=28, color=INK, clip_on=False)


def fig_group_curve():
    c = json.load(open(str(ROOT) + "/model/group_curves.json"))
    t = np.array(c["times"]); P = np.array(c["curves"]["CR"])
    fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
    keep = [("NR", 0), ("CR", 3), ("M", 4)]
    ys = []
    for s, j in keep:
        ax.plot(t, P[:, j], color=SC[s], lw=4, solid_capstyle="round"); ys.append(P[:, j])
    line_end_labels(ax, t, ys, [f"{SHORT[s]} {P[-1, j]:.0%}" for s, j in keep], [SC[s] for s, _ in keep])
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "50%", "100%"])
    ax.set_xlabel("Years from now"); ax.set_ylabel("Probability")
    ax.grid(axis="y", color="#EDEDED", lw=1); ax.set_axisbelow(True); sep(ax); ax.set_xticks([0, 5, 10])
    save(fig, "curve_group_CR.png")


def fig_profiles():
    c = json.load(open(str(HERE) + "/profile_curves.json")); t = np.array(c["t"])
    A, B = np.array(c["A"]), np.array(c["B"])
    tog = lambda P: P[:, 3] + P[:, 4]
    fig, ax = plt.subplots(figsize=(9.0, 4.05), layout="constrained")
    ax.plot(t, tog(A), color=PRIMARY, lw=4); ax.plot(t, tog(B), color=SECONDARY, lw=4)
    i5 = np.argmin(abs(t - 5))
    for P, col in ((A, PRIMARY), (B, SECONDARY)):
        ax.scatter(5, tog(P)[i5], s=160, color=col, zorder=3, edgecolor="white", linewidth=2)
        ax.text(5.25, tog(P)[i5] + 0.03, f"{tog(P)[i5]:.0%}", fontsize=28, color=INK, va="bottom")
    ax.text(10.15, tog(A)[-1], "A", fontsize=28, color=INK, va="center", weight="bold")
    ax.text(10.15, tog(B)[-1], "B", fontsize=28, color=INK, va="center", weight="bold")
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "50%", "100%"])
    ax.set_xlabel("Years from now"); ax.set_ylabel("Still together")
    ax.grid(axis="y", color="#EDEDED", lw=1); ax.set_axisbelow(True); sep(ax); ax.set_xticks([0, 5, 10])
    save(fig, "curve_profiles.png")


def fig_interest():
    c = json.load(open(str(HERE) + "/profile_curves.json")); t = np.array(c["t"])
    H, L = np.array(c["H"]), np.array(c["L"])
    k = t <= 3.0
    fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
    ax.plot(t[k], H[k, 3], color=PRIMARY, lw=4); ax.plot(t[k], L[k, 3], color=SECONDARY, lw=4)
    i1 = np.argmin(abs(t - 1))
    for P, col in ((H, PRIMARY), (L, SECONDARY)):
        ax.scatter(1, P[i1, 3], s=160, color=col, zorder=3, edgecolor="white", linewidth=2)
        ax.text(1.1, P[i1, 3] + 0.004, f"{P[i1, 3]:.0%}", fontsize=28, color=INK, va="bottom")
    ax.text(3.07, H[k, 3][-1], "high", fontsize=28, color=INK, va="center", weight="bold")
    ax.text(3.07, L[k, 3][-1], "low", fontsize=28, color=INK, va="center", weight="bold")
    ax.set_xlim(0, 3); ax.set_ylim(0, 0.14); ax.set_yticks([0, 0.05, 0.10]); ax.set_yticklabels(["0", "5%", "10%"])
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Years from now"); ax.set_ylabel("Committed")
    ax.grid(axis="y", color="#EDEDED", lw=1); ax.set_axisbelow(True); sep(ax)
    save(fig, "curve_interest.png")


# ---------------------------------------------------------------- 5. sensitivity: add one at a time
def fig_sensitivity():
    R = json.load(open(str(ROOT) + "/model/results_cv.json")); b = R["base"]["nll"]
    lab = {"cohabiting": "Cohabiting", "romantic_years": "Years together", "known_years": "Years known",
           "rel_quality": "Relationship quality", "interest_self": "Own interest", "age": "Age",
           "interest_partner_perceived": "Partner's interest", "trust": "Trust", "commitment": "Commitment",
           "income": "Income", "same_sex": "Same-sex couple", "edu_years": "Education"}
    src = {"cohabiting": "H", "romantic_years": "H", "known_years": "H", "rel_quality": "H", "income": "H",
           "edu_years": "H", "interest_self": "E", "interest_partner_perceived": "E", "trust": "E",
           "commitment": "E", "age": "B", "same_sex": "B"}
    col = {"H": PRIMARY, "E": SECONDARY, "B": LGREY}
    keys = [k for k in R["single_rank"] if k in lab][:8]
    vals = [(b - R["add_" + k]["nll"]) * 1000 for k in keys]
    fig, ax = plt.subplots(figsize=(9.0, 4.6), layout="constrained")
    y = np.arange(len(keys))
    ax.barh(y, vals, height=0.62, color=[col[src[k]] for k in keys])
    for yi, v in zip(y, vals):
        ax.text(v + 0.2, yi, f"{v:.1f}", va="center", fontsize=28, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in keys], fontsize=28, color=INK)
    ax.invert_yaxis(); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 15); ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel(r"Drop in CV log loss ($\times 10^{-3}$)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PRIMARY, label="HCMST"), Patch(color=SECONDARY, label="Eastwick"),
                       Patch(color=LGREY, label="both")], loc="lower right", frameon=False, fontsize=28,
              handlelength=0.9, borderaxespad=0.1)
    save(fig, "sensitivity.png")


# ---------------------------------------------------------------- 6. rate multipliers (log scale)
def fig_multipliers():
    rows = [("Quality → marriage", 0.509), ("Education → marriage", 0.267), ("Cohabiting → marriage", 0.265),
            ("Age → marriage", -0.343), ("Years together → breakup", -0.535), ("Quality → divorce", -0.548),
            ("Interest → potential", 0.436)]
    fig, ax = plt.subplots(figsize=(7.9, 4.7), layout="constrained")
    ax.axvline(0, color=GREY, lw=1.5)
    for i, (lab, bta) in enumerate(rows):
        c = PRIMARY if bta > 0 else SECONDARY
        ax.plot([0, bta], [i, i], color=c, lw=3, solid_capstyle="round")
        ax.scatter(bta, i, s=300, color=c, edgecolor="white", linewidth=2, zorder=3)
        ax.text(1.04, i, f"{bta:+.2f}".replace("-", "−"), transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=28, color=INK)
        ax.text(1.42, i, f"×{np.exp(bta):.2f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=28, color=MUTED)
    ax.text(1.04, -0.95, "β", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=28, color=INK, weight="bold")
    ax.text(1.42, -0.95, "rate", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=28, color=MUTED, weight="bold")
    ax.set_xlim(-0.7, 0.7); ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=28, color=INK)
    ax.set_ylim(len(rows) - 0.5, -1.3)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("β per +1 SD")
    save(fig, "multipliers.png")


if __name__ == "__main__":
    fig_rates(True); fig_compare(); fig_calib(); fig_group_curve(); fig_profiles(); fig_interest()
    fig_sensitivity(); fig_multipliers()
