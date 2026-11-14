# Source tables and the four figures: skill by horizon, direct learners and truncation, NLSY79 transport, simulation.
import os, sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

if os.environ.get("NATURE_FIGURE_SCRIPTS"):
    sys.path.insert(0, os.environ["NATURE_FIGURE_SCRIPTS"])
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    def require_matplotlib_panel_alignment(fig, **kwargs):
        return None

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.spines.right": False,
                     "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.labelsize": 7, "legend.fontsize": 6.5})
CHAIN, BASE, GBM, LOGIT, PHD, ITM = "#0F4D92", "#767676", "#B64342", "#9A4D8E", "#42949E", "#D08C2E"
ROOT = Path(os.environ.get("LOVE_ROOT", Path(__file__).resolve().parent))
DATA = ROOT / "figures" / "data"; DATA.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "figures" / "out"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "model"))
LM = {"nlsy2002": "From 2002 (18-24)", "nlsy2008": "From 2008 (24-30)", "hcmst2017": "HCMST 2017"}
CELL_ORDER = [("nlsy2002", 1), ("nlsy2002", 5), ("nlsy2002", 10), ("nlsy2002", 20),
              ("nlsy2008", 1), ("nlsy2008", 5), ("nlsy2008", 10), ("nlsy2008", 14), ("hcmst2017", 3), ("hcmst2017", 5)]
CELL_LABEL = {("nlsy2002", h): f"2002, {h} y" for h in (1, 5, 10, 20)}
CELL_LABEL.update({("nlsy2008", h): f"2008, {h} y" for h in (1, 5, 10, 14)})
CELL_LABEL.update({("hcmst2017", h): f"HCMST, {h} y" for h in (3, 5)})

def panel_label(fig, ax, letter, dx_pt=-14, dy_pt=4):

    from matplotlib.transforms import ScaledTranslation
    off = ScaledTranslation(dx_pt / 72, dy_pt / 72, fig.dpi_scale_trans)
    ax.text(0, 1, letter, transform=ax.transAxes + off, fontsize=8, fontweight="bold", va="bottom", ha="left")

def save(fig, name, **align_kw):
    base = OUT / name
    require_matplotlib_panel_alignment(fig, json_out=f"{base}.alignment.json", overlay_svg=f"{base}.alignment.svg",
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **align_kw)
    fig.savefig(f"{base}.svg"); fig.savefig(f"{base}.pdf"); fig.savefig(f"{base}.png", dpi=400)
    print("saved", base)

import numpy as np, pandas as pd
import common as C
from cv import load_pairs
from cv import PRIMARY
from matplotlib.patches import Patch
import json, glob

def build_data():

    OUT = str(DATA)
    A = C.load(); hh = A.drop_duplicates("pid").set_index("pid").household
    CELLS = [(lm, h) for lm, H in PRIMARY.items() for h in H]
    rng = np.random.default_rng(3)

    def pooled(names):
        d = {n: load_pairs(n) for n in names}
        M = d[names[0]][["landmark", "pid", "h", "w", "loss"]].rename(columns={"loss": names[0]})
        for n in names[1:]:
            M = M.merge(d[n][["landmark", "pid", "h", "loss"]].rename(columns={"loss": n}), on=["landmark", "pid", "h"])
        M["hh"] = M.pid.map(hh)
        return M

    def boot_cells(M, names, cells, fn, n_boot=1000):

        H_ = M.hh.unique(); hi = M.hh.map({h: i for i, h in enumerate(H_)}).to_numpy()
        rows = []
        for lm, h in cells:
            m = ((M.landmark == lm) & (M.h == h)).to_numpy()
            w = M.w.to_numpy()[m]; L = {n: M[n].to_numpy()[m] for n in names}; hi_m = hi[m]
            pt = fn({n: np.average(v, weights=w) for n, v in L.items()})
            bs = []
            for _ in range(n_boot):
                cnt = np.bincount(rng.integers(0, len(H_), len(H_)), minlength=len(H_)); ww = w * cnt[hi_m]
                bs.append(fn({n: np.sum(ww * v) / np.sum(ww) for n, v in L.items()}))
            rows.append(dict(landmark=lm, h=h, n=int(m.sum()), est=pt, lo=float(np.percentile(bs, 5)), hi=float(np.percentile(bs, 95))))
        return pd.DataFrame(rows)

    M = pooled(["none_su", "base", "latbest", "phase", "phase_lat"])
    tabs = []
    for s in ("base", "latbest", "phase", "phase_lat"):
        t = boot_cells(M, ["none_su", s], CELLS, lambda L, s=s: 1 - L[s] / L["none_su"]); t["structure"] = s; tabs.append(t)
    pd.concat(tabs).to_csv(f"{OUT}/fig1_skill_cv.csv", index=False)

    rows = []
    for s in ("latbest", "phase", "phase_lat"):
        d = json.load(open(f"{ROOT}/runs/cv/compare_base__{s}.json"))[s]
        rows.append(dict(structure=s, mean_gain=d["mean_gain_nats"], lo=d["mean_gain_ci90"][0], hi=d["mean_gain_ci90"][1], worst=d["worst_cell_change"]))
        for k, v in d["cells"].items():
            rows.append(dict(structure=s, cell=k, diff=v["diff"], lo=v["ci90"][0], hi=v["ci90"][1]))
    pd.DataFrame(rows).to_csv(f"{OUT}/fig1_structure_gains.csv", index=False)

    rows = []
    margins = json.load(open(f"{ROOT}/runs/margins.json"))["margins"]
    for c in ("gbmT", "logitT", "phd", "itm_gbm"):
        d = json.load(open(f"{ROOT}/runs/cv/compare_phase_lat__{c}.json"))[c]
        for k, v in d["cells"].items():
            lm, h = k.rsplit("_h", 1)
            rows.append(dict(competitor=c, landmark=lm, h=int(h), diff_chain_minus=-v["diff"], lo=-v["ci90"][1], hi=-v["ci90"][0],
                             margin=margins[lm][h], chain_loss=v["ref"], competitor_loss=v["cand"]))
    pd.DataFrame(rows).to_csv(f"{OUT}/fig2_competitors_cv.csv", index=False)
    T = json.load(open(f"{ROOT}/runs/cv/compare_trunc_base__trunc_none_trunc_itm_gbm_trunc_itm_logit_trunc_phd_trunc_gbm_h5_trunc_logit_h5_trunc_dtime.json"))
    rows = []
    for c, d in T.items():
        for k, v in d["cells"].items():
            rows.append(dict(model=c.replace("trunc_", ""), h=int(k.rsplit("_h", 1)[1]), loss=v["cand"], chain_loss=v["ref"],
                             diff_minus_chain=v["diff"], lo=v["ci90"][0], hi=v["ci90"][1]))
    pd.DataFrame(rows).to_csv(f"{OUT}/fig2_truncation.csv", index=False)

    rows = []
    for tag, f in (("primary", "phase_lat_79"), ("weighted", "phase_lat_79_weighted"), ("full_info", "phase_lat_full")):
        d = json.load(open(f"{ROOT}/runs/nlsy79/{f}.json"))
        for lm, L in d["landmarks"].items():
            for h, c in L.items():
                r = dict(analysis=tag, landmark=lm, h=int(h), n=c["n"], skill=c["skill"], skill_lo=c["skill_ci90"][0], skill_hi=c["skill_ci90"][1],
                         chain_loss=c["loss"]["chain"], null_loss=c["loss"]["null"], primary_horizon=("E1_pass" in c))
                for n in ("logitT", "gbmT", "itm_gbm", "itm_logit", "phd"):
                    r[f"diff_{n}"] = c["diff"][n]["est"]; r[f"diff_{n}_lo"] = c["diff"][n]["ci90"][0]; r[f"diff_{n}_hi"] = c["diff"][n]["ci90"][1]
                if "E2" in c:
                    r["margin"] = c["E2"]["gbmT"]["margin"]
                for cls, v in c["calibration_slope"].items():
                    r[f"slope_{cls}"] = v
                rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT}/fig3_nlsy79.csv", index=False)
    print("written:", *sorted(__import__("os").listdir(OUT)))

    S5 = json.load(open(f"{ROOT}/runs/cv/supplement_cv.json"))
    rows = []
    for k, v in S5.items():
        lm, h = k.split("|")
        rows.append(dict(landmark=lm, h=int(h), skill_demo=v["skill_vs_stage_unknown"], skill_demo_lo=v["skill_vs_stage_unknown_ci90"][0], skill_demo_hi=v["skill_vs_stage_unknown_ci90"][1],
                         skill_state=v["skill_vs_state_known"], skill_state_lo=v["skill_vs_state_known_ci90"][0], skill_state_hi=v["skill_vs_state_known_ci90"][1],
                         share_state=v["share_of_total_skill_from_state"], share_lo=v["share_ci90"][0], share_hi=v["share_ci90"][1]))
    pd.DataFrame(rows).to_csv(f"{OUT}/fig5_skill_components.csv", index=False)
    I = json.load(open(f"{ROOT}/runs/info_decomp/summary.json"))["by_cell"]
    rows = []
    for k, v in I.items():
        lm, h = k.split("|"); tot = v["loss_none"] - v["loss_all"]
        for g in ("feelings", "background", "family", "history"):
            rows.append(dict(landmark=lm, h=int(h), source=g, phi=v["phi"][g], lo=v["phi_ci90"][g][0], hi=v["phi_ci90"][g][1], total=tot))
    pd.DataFrame(rows).to_csv(f"{OUT}/fig5_shapley.csv", index=False)
    import torch
    from chain import KINDS
    kinds = [k for k in KINDS if not k.endswith("X")]
    def mult(k):
        ck = torch.load(f"{ROOT}/runs/cv/phase_lat_f{k}.pt", weights_only=False)
        return np.exp(torch.as_tensor(ck["delta"]).detach().numpy()), int(ck["G"])
    fin, _ = mult(-1); folds = [mult(k)[0] for k in (0, 1, 3)]
    rows = []
    for g, name in ((0, "settled"), (1, "mobile")):
        for kind in kinds:
            i = KINDS.index(kind); vals = [f[g, i] for f in folds]
            rows.append(dict(cls=name, kind=kind, mult=float(fin[g, i]), fold_min=float(min(vals + [fin[g, i]])), fold_max=float(max(vals + [fin[g, i]]))))
    pd.DataFrame(rows).to_csv(f"{OUT}/fig6_multipliers.csv", index=False)
    pd.read_csv(f"{ROOT}/runs/class_outcomes_10y.csv").to_csv(f"{OUT}/fig6_outcomes.csv", index=False)
    T79 = json.load(open(f"{ROOT}/runs/nlsy79/structures_79.json"))
    rows = []
    for lm, cells in T79.items():
        for h, v in cells.items():
            rows.append(dict(landmark=lm, h=int(h), skill=v["skill"]["base79"], skill_lo=v["skill_ci90"]["base79"][0], skill_hi=v["skill_ci90"]["base79"][1]))
    pd.DataFrame(rows).to_csv(f"{OUT}/fig3_earlier_model.csv", index=False)

def fig1():

    S = pd.read_csv(DATA / "fig1_skill_cv.csv")
    G_all = pd.read_csv(DATA / "fig1_structure_gains.csv")
    G = G_all[G_all.cell.notna()]

    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.42, left=0.075, right=0.985, bottom=0.2, top=0.9)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    for lm, col, mk in (("nlsy2002", CHAIN, "o"), ("nlsy2008", GBM, "s"), ("hcmst2017", PHD, "^")):
        for struct, ls, lab in (("base", ":", "earlier main model"), ("phase_lat", "-", "Final structure")):
            d = S[(S.structure == struct) & (S.landmark == lm)].sort_values("h")
            x = d.h.to_numpy(); y = d.est.to_numpy() * 100
            if struct == "phase_lat":
                axa.fill_between(x, d.lo * 100, d.hi * 100, color=col, alpha=0.16, lw=0)
            axa.plot(x, y, ls=ls, marker=mk, color=col, lw=1.3, ms=3.4, mfc=(col if struct == "phase_lat" else "white"),
                     label=f"{LM[lm]}, {lab.lower()}")
    axa.axhline(0, color=BASE, lw=0.6, ls="-")
    axa.set_xlabel("Forecast horizon (years)"); axa.set_ylabel("Skill over baseline (%)")
    axa.set_xticks([1, 3, 5, 10, 14, 20]); axa.set_ylim(-2, 65)
    axa.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), handlelength=2.2, borderaxespad=0.2)

    labels = {"latbest": "Latent classes", "phase": "Duration phases", "phase_lat": "Both (final)"}
    cols = {"latbest": ITM, "phase": PHD, "phase_lat": CHAIN}
    ypos = np.arange(len(CELL_ORDER))[::-1]
    for j, s in enumerate(("latbest", "phase", "phase_lat")):
        d = G[G.structure == s].set_index("cell")
        vals = np.array([d.loc[f"{lm}_h{h}", "diff"] for lm, h in CELL_ORDER])
        lo = np.array([d.loc[f"{lm}_h{h}", "lo"] for lm, h in CELL_ORDER]); hi = np.array([d.loc[f"{lm}_h{h}", "hi"] for lm, h in CELL_ORDER])
        off = (j - 1) * 0.26
        axb.errorbar(vals, ypos + off, xerr=[vals - lo, hi - vals], fmt="o", color=cols[s], ms=3, elinewidth=0.8, capsize=0, label=labels[s])
    axb.axvline(0, color=BASE, lw=0.6); axb.axvline(-0.003, color=BASE, lw=0.6, ls="--")
    axb.set_yticks(ypos); axb.set_yticklabels([CELL_LABEL[c] for c in CELL_ORDER])
    axb.set_xlabel("Change in log loss versus earlier main model (nats)")
    axb.set_xlim(-0.021, 0.006); axb.set_ylim(-0.7, len(CELL_ORDER) - 0.3)
    axb.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2)
    axb.text(-0.003, 1.01, "inclusion threshold", fontsize=6, ha="center", va="bottom", color="#272727", transform=axb.get_xaxis_transform())

    panel_label(fig, axa, "a", dx_pt=-30); panel_label(fig, axb, "b", dx_pt=-52)
    require_matplotlib_panel_alignment(fig, json_out=str(OUT / "fig1.alignment.json"), overlay_svg=str(OUT / "fig1.alignment.svg"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, exemptions=[{"panels": ["a"], "checks": ["panel-width"], "reason": "hero panel a is wider by design"}])
    fig.savefig(OUT / "fig1.svg"); fig.savefig(OUT / "fig1.pdf"); fig.savefig(OUT / "fig1.png", dpi=600); fig.savefig(OUT / "fig1.tiff", dpi=600)
    print("saved", OUT / "fig1")

def fig2():

    Cv = pd.read_csv(DATA / "fig2_competitors_cv.csv")
    T = pd.read_csv(DATA / "fig2_truncation.csv")

    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.5, left=0.11, right=0.985, bottom=0.19, top=0.9)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    names = {"gbmT": ("Gradient boosting", GBM), "logitT": ("Logistic regression", LOGIT), "phd": ("Pooled-horizon boosting", PHD),
             "itm_gbm": ("Iterated boosting transitions", ITM)}
    ypos = np.arange(len(CELL_ORDER))[::-1]
    mg = Cv[Cv.competitor == "gbmT"].set_index(["landmark", "h"]).margin
    axa.barh(ypos, [mg.loc[c] for c in CELL_ORDER], left=0, height=0.86, color="#E6E6E6", lw=0, zorder=0)
    for j, (c, (lab, col)) in enumerate(names.items()):
        d = Cv[Cv.competitor == c].set_index(["landmark", "h"])
        v = np.array([d.loc[cell, "diff_chain_minus"] for cell in CELL_ORDER])
        lo = np.array([d.loc[cell, "lo"] for cell in CELL_ORDER]); hi = np.array([d.loc[cell, "hi"] for cell in CELL_ORDER])
        off = (j - 1.5) * 0.2
        axa.errorbar(v, ypos + off, xerr=[v - lo, hi - v], fmt="o", color=col, ms=2.6, elinewidth=0.7, capsize=0, label=lab, zorder=3)
    axa.axvline(0, color="#272727", lw=0.6, zorder=2)
    axa.set_yticks(ypos); axa.set_yticklabels([CELL_LABEL[c] for c in CELL_ORDER])
    axa.set_xlabel("Chain minus competitor, log loss (nats)")
    axa.set_xlim(-0.062, 0.125); axa.set_ylim(-0.7, len(CELL_ORDER) - 0.3)
    h_, l_ = axa.get_legend_handles_labels()
    axa.legend(h_ + [Patch(color="#E6E6E6")], l_ + ["Non-inferiority margin"], loc="upper right", handlelength=1.2, borderaxespad=0.2)

    order = [("itm_gbm", "Iterated boosting transitions", ITM), ("itm_logit", "Iterated logistic transitions", LOGIT),
             ("phd", "Pooled-horizon boosting", PHD), ("gbm_h5", "Gradient boosting, 5-year model", GBM),
             ("logit_h5", "Logistic regression, 5-year model", "#C98A8A"), ("dtime", "Discrete-time hazard", "#9B9B9B"), ("none", "Baseline", BASE)]
    yb = np.arange(len(order))[::-1]
    for j, h in enumerate((10, 20)):
        d = T[T.h == h].set_index("model")
        v = np.array([d.loc[m, "diff_minus_chain"] for m, _, _ in order])
        lo = np.array([d.loc[m, "lo"] for m, _, _ in order]); hi = np.array([d.loc[m, "hi"] for m, _, _ in order])
        off = 0.18 if h == 10 else -0.18
        axb.errorbar(v, yb + off, xerr=[v - lo, hi - v], fmt=("o" if h == 10 else "s"), color=("#272727" if h == 10 else CHAIN),
                     ms=2.8, elinewidth=0.7, capsize=0, label=f"{h} years", zorder=3)
    axb.axvline(0, color="#272727", lw=0.6)
    axb.set_xscale("symlog", linthresh=0.05, linscale=0.8)
    axb.set_xticks([-0.03, 0, 0.05, 0.2, 0.5, 1.0]); axb.set_xticklabels(["-0.03", "0", "0.05", "0.2", "0.5", "1"])
    axb.set_yticks(yb); axb.set_yticklabels([lab for _, lab, _ in order])
    axb.set_xlabel("Competitor minus chain, log loss (nats)")
    axb.set_xlim(-0.035, 1.6); axb.set_ylim(-0.7, len(order) - 0.3)
    axb.legend(loc="upper right", handlelength=1.2, borderaxespad=0.2)

    panel_label(fig, axa, "a", dx_pt=-52); panel_label(fig, axb, "b", dx_pt=-112)
    require_matplotlib_panel_alignment(fig, json_out=str(OUT / "fig2.alignment.json"), overlay_svg=str(OUT / "fig2.alignment.svg"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, exemptions=[{"panels": ["a"], "checks": ["panel-width"], "reason": "hero panel a is wider by design"}])
    fig.savefig(OUT / "fig2.svg"); fig.savefig(OUT / "fig2.pdf"); fig.savefig(OUT / "fig2.png", dpi=600); fig.savefig(OUT / "fig2.tiff", dpi=600)
    print("saved", OUT / "fig2")

def fig3():

    D = pd.read_csv(DATA / "fig3_nlsy79.csv"); D = D[D.analysis == "primary"]
    LAB = {"L1": "From 1983 (ages 18-24)", "L2": "From 1989 (ages 24-30)"}

    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1], wspace=0.42, hspace=0.6, left=0.075, right=0.985, bottom=0.25, top=0.92)
    axa = fig.add_subplot(gs[:, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 1])

    E3 = pd.read_csv(DATA / "fig3_earlier_model.csv")
    for lm, col in (("L1", CHAIN), ("L2", GBM)):
        d = D[D.landmark == lm].sort_values("h")
        e3 = E3[E3.landmark == lm].sort_values("h")
        axa.plot(e3.h, e3.skill * 100, ":", color=col, lw=1.1, marker="o", ms=3.0, mfc="white", label=f"{LAB[lm]}, earlier main model")
        axa.fill_between(d.h, d.skill_lo * 100, d.skill_hi * 100, color=col, alpha=0.16, lw=0)
        axa.plot(d.h, d.skill * 100, "-", color=col, lw=1.3, label=f"{LAB[lm]}, final structure")
        p = d[d.primary_horizon]; e = d[~d.primary_horizon]
        axa.plot(p.h, p.skill * 100, "o", color=col, ms=3.6)
        axa.plot(e.h, e.skill * 100, "o", color=col, ms=3.6, mfc="white")
    axa.axhline(0, color=BASE, lw=0.6)
    axa.set_xlabel("Forecast horizon (years)"); axa.set_ylabel("Skill over baseline (%)")
    axa.set_xticks([1, 5, 10, 14, 20, 30, 33, 39]); axa.set_xticklabels(["1", "5", "10", "14", "20", "30", "33", "39"])
    axa.set_ylim(-4, 65)
    axa.legend(loc="upper right", handlelength=2.2, borderaxespad=0.2, fontsize=6)
    axa.text(7, 30, "filled: primary horizons\nopen: beyond any training follow-up", fontsize=6, ha="left", va="bottom", color="#272727")

    names = {"gbmT": ("Gradient boosting", GBM), "logitT": ("Logistic regression", LOGIT), "phd": ("Pooled-horizon boosting", PHD),
             "itm_gbm": ("Iterated boosting transitions", ITM)}
    for ax, lm in ((axb, "L1"), (axc, "L2")):
        d = D[D.landmark == lm].sort_values("h")
        H = d.h.to_numpy(); x = np.arange(len(H))
        m = d.margin.to_numpy()
        ax.bar(x[~np.isnan(m)], m[~np.isnan(m)], width=0.8, color="#E6E6E6", lw=0, zorder=0)
        for j, (c, (lab, col)) in enumerate(names.items()):
            v = d[f"diff_{c}"].to_numpy(); lo = d[f"diff_{c}_lo"].to_numpy(); hi = d[f"diff_{c}_hi"].to_numpy()
            ax.errorbar(x + (j - 1.5) * 0.17, v, yerr=[v - lo, hi - v], fmt="o", color=col, ms=2.4, elinewidth=0.7, capsize=0, label=lab, zorder=3)
        ax.axhline(0, color="#272727", lw=0.6, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels([str(h) for h in H])
        ax.set_title(LAB[lm], fontsize=7, loc="left", pad=3)
        ax.set_ylim(-0.24, 0.165)
    axc.set_xlabel("Forecast horizon (years)")
    axb.set_ylabel("Chain minus\ncompetitor (nats)"); axc.set_ylabel("Chain minus\ncompetitor (nats)")
    h_, l_ = axb.get_legend_handles_labels()
    fig.legend(h_ + [Patch(color="#E6E6E6")], l_ + ["Non-inferiority margin"], loc="lower center", bbox_to_anchor=(0.72, 0.0), ncol=3, handlelength=1.2, columnspacing=1.0, fontsize=6.5)

    panel_label(fig, axa, "a", dx_pt=-30); panel_label(fig, axb, "b", dx_pt=-46); panel_label(fig, axc, "c", dx_pt=-46)
    require_matplotlib_panel_alignment(fig, json_out=str(OUT / "fig3.alignment.json"), overlay_svg=str(OUT / "fig3.alignment.svg"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, exemptions=[{"panels": ["a"], "checks": ["panel-width"], "reason": "hero panel a is wider by design"}])
    fig.savefig(OUT / "fig3.svg"); fig.savefig(OUT / "fig3.pdf"); fig.savefig(OUT / "fig3.png", dpi=600); fig.savefig(OUT / "fig3.tiff", dpi=600)
    print("saved", OUT / "fig3")

def fig4():

    DES = [("neither", "Neither"), ("duration", "Duration\nonly"), ("heterogeneity", "Heterogeneity\nonly"), ("both", "Both")]
    STR = [("base", "One phase, one class", BASE), ("lat", "Latent classes", ITM), ("phase", "Duration phases", PHD), ("phase_lat", "Both", CHAIN)]
    rows = []
    for f in sorted(glob.glob(str(ROOT / "runs" / "sim" / "*.json"))):
        d = json.load(open(f))
        rows.append(dict(design=d["design"], rep=d["rep"], selected=d["selected"], **{f"gain_{s}": d["gain"][s] for s, _, _ in STR}))
    R = pd.DataFrame(rows)
    R.to_csv(DATA / "fig4_simulation.csv", index=False)
    n_by = R.groupby("design").size().to_dict()
    print("replicates per design:", n_by)

    fig = plt.figure(figsize=(7.2, 2.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.38, left=0.085, right=0.985, bottom=0.36, top=0.9)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    x = np.arange(len(DES)); bottom = np.zeros(len(DES))
    for s, lab, col in STR:
        share = np.array([((R.design == d) & (R.selected == s)).sum() / max(n_by.get(d, 0), 1) for d, _ in DES])
        axa.bar(x, share, 0.62, bottom=bottom, color=col, lw=0, label=lab); bottom += share
    axa.set_xticks(x); axa.set_xticklabels([f"{lab}\n(n = {n_by.get(d, 0)})" for d, lab in DES], fontsize=6.5)
    axa.set_ylabel("Share of replicates"); axa.set_ylim(0, 1.0)
    axa.set_xlabel("True design")
    h_, l_ = axa.get_legend_handles_labels()
    fig.legend(h_, ["Selected: " + l_[0]] + l_[1:], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, handlelength=1.0, columnspacing=1.2, fontsize=6.5, frameon=False)

    rng = np.random.default_rng(0)
    for i, (d, lab) in enumerate(DES):
        sub = R[R.design == d]
        for j, (s, slab, col) in enumerate(STR[1:]):
            xx = i + (j - 1) * 0.26 + rng.uniform(-0.05, 0.05, len(sub))
            axb.plot(xx, sub[f"gain_{s}"], "o", color=col, ms=2.6, alpha=0.85, mew=0)
    axb.axhline(0.003, color="#272727", lw=0.6, ls="--"); axb.axhline(0, color=BASE, lw=0.6)
    axb.set_xticks(x); axb.set_xticklabels([lab for _, lab in DES], fontsize=6.5)
    axb.set_ylabel("Ten-cell mean gain over one-phase,\none-class fit (nats)")
    axb.set_xlabel("True design")
    axb.text(-0.45, 0.0033, "inclusion threshold", fontsize=6, ha="left", va="bottom", color="#272727")

    panel_label(fig, axa, "a", dx_pt=-30); panel_label(fig, axb, "b", dx_pt=-36)
    require_matplotlib_panel_alignment(fig, json_out=str(OUT / "fig4.alignment.json"), overlay_svg=str(OUT / "fig4.alignment.svg"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, exemptions=[{"panels": ["b"], "checks": ["panel-width"], "reason": "panel b is wider by design"}])
    fig.savefig(OUT / "fig4.svg"); fig.savefig(OUT / "fig4.pdf"); fig.savefig(OUT / "fig4.png", dpi=600); fig.savefig(OUT / "fig4.tiff", dpi=600)
    print("saved", OUT / "fig4")


def fig5():
    S = pd.read_csv(DATA / "fig5_skill_components.csv"); P = pd.read_csv(DATA / "fig5_shapley.csv")
    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1.25], wspace=0.5, left=0.07, right=0.985, bottom=0.36, top=0.92)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    for lm, col, mk in (("nlsy2002", CHAIN, "o"), ("nlsy2008", GBM, "s"), ("hcmst2017", PHD, "^")):
        d = S[S.landmark == lm].sort_values("h")
        axa.fill_between(d.h, d.skill_demo_lo * 100, d.skill_demo_hi * 100, color=col, alpha=0.16, lw=0)
        axa.plot(d.h, d.skill_demo * 100, "-", marker=mk, color=col, lw=1.3, ms=3.4, label=f"{LM[lm]}, demographic baseline")
        axa.fill_between(d.h, d.skill_state_lo * 100, d.skill_state_hi * 100, color=col, alpha=0.16, lw=0)
        axa.plot(d.h, d.skill_state * 100, "--", marker=mk, color=col, lw=1.1, ms=3.4, mfc="white", label=f"{LM[lm]}, state-aware baseline")
        axb.errorbar(d.h, d.share_state * 100, yerr=[(d.share_state - d.share_lo) * 100, (d.share_hi - d.share_state) * 100], fmt="-" , marker=mk, color=col, lw=1.1, ms=3.4, elinewidth=0.7, capsize=0)
    axa.axhline(0, color=BASE, lw=0.6)
    axa.set_xlabel("Forecast horizon (years)"); axa.set_ylabel("Skill of the final structure (%)")
    axa.set_xticks([1, 3, 5, 10, 14, 20]); axa.set_ylim(-2, 65)
    h_, l_ = axa.get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.42, 0.0), ncol=3, handlelength=2.0, columnspacing=1.0, fontsize=6, frameon=False)
    axb.set_xlabel("Forecast horizon (years)"); axb.set_ylabel("Share of skill from knowing\nthe current state (%)")
    axb.set_xticks([1, 3, 5, 10, 14, 20]); axb.set_ylim(0, 105)
    srcs = [("background", "Demographic background", ITM), ("family", "Family background", PHD), ("feelings", "Relationship feelings", GBM), ("history", "Partnership history", LOGIT)]
    x = np.arange(len(CELL_ORDER)); bottom_pos = np.zeros(len(CELL_ORDER)); bottom_neg = np.zeros(len(CELL_ORDER))
    for g, lab, col in srcs:
        vals = np.array([P[(P.landmark == lm) & (P.h == h) & (P.source == g)].phi.iloc[0] for lm, h in CELL_ORDER])
        pos = np.clip(vals, 0, None); neg = np.clip(vals, None, 0)
        axc.bar(x, pos, 0.7, bottom=bottom_pos, color=col, lw=0, label=lab); bottom_pos += pos
        axc.bar(x, neg, 0.7, bottom=bottom_neg, color=col, lw=0); bottom_neg += neg
    axc.axhline(0, color="#272727", lw=0.6)
    axc.set_xticks(x); axc.set_xticklabels([str(h) for _, h in CELL_ORDER], fontsize=6)
    axc.set_xlabel("Forecast horizon (years)", labelpad=24)
    for lab, (i0, i1) in (("From 2002", (0, 3)), ("From 2008", (4, 7)), ("HCMST", (8, 9))):
        axc.text((i0 + i1) / 2, -0.16, lab, transform=axc.get_xaxis_transform(), ha="center", va="top", fontsize=6)
        axc.plot([i0 - 0.35, i1 + 0.35], [-0.13, -0.13], transform=axc.get_xaxis_transform(), color="#272727", lw=0.6, clip_on=False)
    axc.set_ylabel("Loss reduction attributed\nto each source (nats)")
    axc.set_ylim(top=0.235)
    axc.legend(loc="upper right", handlelength=1.0, borderaxespad=0.2, fontsize=5.8)
    panel_label(fig, axa, "a", dx_pt=-30); panel_label(fig, axb, "b", dx_pt=-30); panel_label(fig, axc, "c", dx_pt=-36)
    require_matplotlib_panel_alignment(fig, json_out=str(OUT / "fig5.alignment.json"), overlay_svg=str(OUT / "fig5.alignment.svg"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, exemptions=[{"panels": ["a", "c"], "checks": ["panel-width"], "reason": "outer panels wider by design"}])
    fig.savefig(OUT / "fig5.svg"); fig.savefig(OUT / "fig5.pdf"); fig.savefig(OUT / "fig5.png", dpi=600); fig.savefig(OUT / "fig5.tiff", dpi=600)
    print("saved", OUT / "fig5")


def fig6():
    Mu = pd.read_csv(DATA / "fig6_multipliers.csv"); Oc = pd.read_csv(DATA / "fig6_outcomes.csv")
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.42, left=0.11, right=0.985, bottom=0.27, top=0.9)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    kinds = list(Mu.kind.unique()); y = np.arange(len(kinds))[::-1]
    for j, (cls, lab, col) in enumerate((("settled", "Settled class", PHD), ("mobile", "Mobile class", GBM))):
        d = Mu[Mu.cls == cls].set_index("kind").loc[kinds]
        off = (j - 0.5) * 0.3
        axa.errorbar(d.mult, y + off, xerr=[d.mult - d.fold_min, d.fold_max - d.mult], fmt="o", color=col, ms=3.2, elinewidth=0.8, capsize=0, label=lab)
    axa.axvline(1, color="#272727", lw=0.6)
    axa.set_xscale("log"); axa.set_xticks([0.03, 0.1, 0.3, 1, 3, 10]); axa.set_xticklabels(["0.03", "0.1", "0.3", "1", "3", "10"])
    axa.set_yticks(y); axa.set_yticklabels([k.replace(">", " \u2192 ") for k in kinds])
    axa.set_xlabel("Intensity relative to the intermediate class"); axa.set_ylim(-0.7, len(kinds) - 0.3)
    ha_, la_ = axa.get_legend_handles_labels()
    fig.legend(ha_, la_, loc="lower center", bbox_to_anchor=(0.33, 0.0), ncol=2, handlelength=1.2, columnspacing=1.2, fontsize=6.5, frameon=False)
    d = Oc[(Oc.landmark == "nlsy2008")].copy()
    groups = [("S", "Single"), ("D", "Dating"), ("C", "Cohabiting"), ("M", "Married")]
    cols = {1: ITM, 2: PHD, 3: GBM}; names = {1: "Intermediate", 2: "Settled", 3: "Mobile"}
    outcome = {"S": ("obs_none", "pred_none", "no partner"), "D": ("obs_married", "pred_married", "married"), "C": ("obs_married", "pred_married", "married"), "M": ("obs_married", "pred_married", "married")}
    xpos = []; labels = []; k = 0
    for st, lab in groups:
        oc, pc, what = outcome[st]
        for c in (1, 2, 3):
            r = d[(d.origin_state == st) & (d.modal_class == c)]
            if len(r) == 0:
                continue
            r = r.iloc[0]; p = r[oc]; n = r.n; z = 1.96
            lo = (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
            hi = (p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
            axb.errorbar(k, p * 100, yerr=[[(p - lo) * 100], [(hi - p) * 100]], fmt="o", color=cols[c], ms=3.4, elinewidth=0.8, capsize=0, label=names[c] if st == "S" else None, zorder=3)
            axb.plot(k, r[pc] * 100, "_", color=cols[c], ms=9, mew=1.4, zorder=2)
            xpos.append(k); labels.append(f"{lab}\n{what}" if c == (2 if st == "S" else 1) else ""); k += 1
        k += 0.8
    axb.set_xticks(xpos); axb.set_xticklabels(labels, fontsize=6)
    axb.set_ylabel("Realized share ten years later (%)"); axb.set_ylim(0, 100)
    axb.plot([], [], "_", color="#272727", ms=9, mew=1.4, label="Model forecast")
    hb_, lb_ = axb.get_legend_handles_labels()
    fig.legend(hb_, lb_, loc="lower center", bbox_to_anchor=(0.78, 0.0), ncol=4, handlelength=1.2, columnspacing=1.0, fontsize=6.5, frameon=False)
    panel_label(fig, axa, "a", dx_pt=-40); panel_label(fig, axb, "b", dx_pt=-30)
    require_matplotlib_panel_alignment(fig, json_out=str(OUT / "fig6.alignment.json"), overlay_svg=str(OUT / "fig6.alignment.svg"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, exemptions=[{"panels": ["a"], "checks": ["panel-width"], "reason": "hero panel a is wider by design"}])
    fig.savefig(OUT / "fig6.svg"); fig.savefig(OUT / "fig6.pdf"); fig.savefig(OUT / "fig6.png", dpi=600); fig.savefig(OUT / "fig6.tiff", dpi=600)
    print("saved", OUT / "fig6")

if __name__ == "__main__":
    for f in (build_data, fig1, fig2, fig3, fig4, fig5, fig6):
        f()
