"""Five-state relationship dynamic model: learn transition rates from data.

Internal states: NR, NP, RP, CR, M, D (D = NR because partner died; absorbing).
Displayed states: NR (=NR+D), NP, RP, CR, M.
rate_e(x) = exp(a_e + x . b_e)   for each allowed edge e  (personal parameters multiply the rate)
P(t) = expm(Q(x) t);   trained by maximum likelihood (L-BFGS) with L2 penalty on b.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from pathlib import Path
_ds = Path(__file__).resolve().parents[1] / "dataset" / "relationship_five_state_dataset.csv"
DATA = str(_ds) if _ds.exists() else "data.csv"

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)

INTERNAL = ["NR", "NP", "RP", "CR", "M", "D", "NRe"]   # NRe: NR after a breakup/divorce (HCMST stops following these)
DISPLAY = ["NR", "NP", "RP", "CR", "M"]
IDX = {s: i for i, s in enumerate(INTERNAL)}
EDGES = [("NR", "NP"), ("NR", "RP"), ("NR", "CR"),
         ("NP", "NR"), ("NP", "RP"), ("NP", "CR"),
         ("RP", "NR"), ("RP", "NP"), ("RP", "CR"),
         ("CR", "NRe"), ("CR", "NP"), ("CR", "RP"), ("CR", "M"), ("CR", "D"),
         ("M", "NRe"), ("M", "D")]
E_FROM = torch.tensor([IDX[a] for a, _ in EDGES])
E_TO = torch.tensor([IDX[b] for _, b in EDGES])

# ---------------------------------------------------------------- candidate parameters
# name -> (column, transform)
CANDIDATES = {
    "age": ("resp_age", None),
    "age_gap": ("partner_age_minus_resp", None),
    "female": ("resp_gender", "female"),
    "same_sex": ("same_sex", None),
    "known_years": ("known_years", "log1p"),
    "romantic_years": ("romantic_years", "log1p"),
    "cohabiting": ("cohabiting", None),
    "edu_years": ("resp_edu_years", None),
    "partner_edu_years": ("partner_edu_years", None),
    "income": ("hh_income_band", None),
    "rel_quality": ("rel_quality_1to5", None),
    "interest_self": ("romantic_interest_self", None),
    "interest_partner_perceived": ("partner_interest_perceived", None),
    "open_up": ("ew_R_RLSHP1_3", None),
    "commitment": ("ew_R_RLSHP1_5", None),
    "satisfaction": ("ew_R_RLSHP1_6", None),
    "trust": ("ew_R_RLSHP1_9", None),
    "see_regularly": ("ew_R_RLSHP1_12", None),
    "physical_contact": ("ew_R_PHYS1", "yes1"),
    "partner_attractive": ("ew_R_TRAIT_1", None),
    "partner_taken": ("ew_R_SO1", "yes1"),
    "extraversion": ("ew_Extrav", None),
    "self_esteem": ("ew_SelfEsteem", None),
    "mate_value": ("ew_MateValue", None),
    "attach_avoidance": ("ew_Avoidance", None),
}


def load_transitions(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.sort_values(["pair_id", "wave"]).copy()
    # carry the last known answer forward within a pair (only from earlier waves: no future info)
    for c in ("rel_quality_1to5", "hh_income_band", "cohabiting"):
        df[c] = df.groupby("pair_id")[c].ffill()
    df["istate"] = np.where((df.state == "NR") & (df.partner_deceased == 1), "D", df.state)
    obs = df[(df.state != "missing") & df.time_years.notna()].copy()
    # NR reached after a CR/M relationship ended -> internal state NRe
    tmp = obs.istate.where(~obs.istate.isin(["NR"]))
    prev = tmp.groupby(obs.pair_id).transform(lambda x: x.shift(1).ffill())
    obs.loc[(obs.istate == "NR") & prev.isin(["CR", "M"]), "istate"] = "NRe"
    nxt = obs.groupby("pair_id").shift(-1)
    ok = nxt.istate.notna() & ~obs.istate.isin(["D", "NRe"])
    o, n = obs[ok].copy(), nxt[ok]
    o["dest"] = n.istate.values
    o["dest_display"] = np.where(o.dest.isin(["D", "NRe"]), "NR", o.dest)
    o["dt"] = n.time_years.values - o.time_years.values
    o["gap_waves"] = n.wave.values - o.wave.values
    o = o[o.dt > 0].reset_index(drop=True)
    # raw covariate values
    for name, (col, tf) in CANDIDATES.items():
        v = o[col]
        if tf == "female":
            v = np.where(v == "female", 1.0, np.where(v == "male", 0.0, np.nan))
        elif tf == "yes1":
            v = pd.to_numeric(v, errors="coerce").map({1: 1.0, 2: 0.0})
        else:
            v = pd.to_numeric(v, errors="coerce")
            if tf == "log1p":
                v = np.log1p(v.clip(lower=0))
        o["x_" + name] = np.asarray(v, dtype=float)
    o["x_is_HCMST"] = (o.source == "HCMST").astype(float)   # diagnostic only, not a personal parameter
    return o


def design(tr, te, names, interactions=()):
    """Standardize on training rows where present; missing -> 0 (no effect)."""
    def build(d, stats=None):
        cols, obs, st = [], [], {} if stats is None else stats
        for nm in names:
            v = d["x_" + nm].to_numpy()
            if stats is None:
                m, s = np.nanmean(v), np.nanstd(v)
                st[nm] = (m, s if s > 0 else 1.0)
            m, s = st[nm]
            z = (v - m) / s
            cols.append(np.where(np.isnan(z), 0.0, z)); obs.append(~np.isnan(z))
        for a, b in interactions:
            ia, ib = names.index(a), names.index(b)
            cols.append(cols[ia] * cols[ib]); obs.append(obs[ia] & obs[ib])
        X = np.column_stack(cols) if cols else np.zeros((len(d), 0))
        O = np.column_stack(obs) if obs else np.zeros((len(d), 0), bool)
        return X, O, st
    Xtr, Otr, st = build(tr)
    Xte, _, _ = build(te, st)
    design.last_observed = Otr
    return Xtr, Xte


MIN_ROWS, MIN_SD, MIN_EVENTS = 50, 0.3, 10


def edge_mask(X, O, d):
    """A parameter may change edge i->j only if, among training rows starting in i where it is
    measured, it really varies (sd >= 0.3 SD) and enough i->j events were observed (>= 10)."""
    o = d.istate.to_numpy(); y = d.dest.to_numpy()
    M = np.zeros((len(EDGES), X.shape[1]))
    for e, (a, b) in enumerate(EDGES):
        r = o == a
        for k in range(X.shape[1]):
            rk = r & O[:, k]
            if rk.sum() >= MIN_ROWS and X[rk, k].std() >= MIN_SD and (rk & (y == b)).sum() >= MIN_EVENTS:
                M[e, k] = 1.0
    return M


def tensors(d):
    return (torch.tensor(d.istate.map(IDX).to_numpy()), torch.tensor(d.dest.map(IDX).to_numpy()),
            torch.tensor(d.dt.to_numpy()))


def meas(d):
    return torch.tensor((d.source == "HCMST").to_numpy(float))


MEAS_EDGES = torch.tensor([float(e in (("CR", "NP"), ("CR", "RP"))) for e in EDGES])


def transition_probs(a, B, X, dt, h=None, g=None):
    """h: 1 if the row's dataset cannot record 'stayed friends after a breakup' (HCMST)."""
    n = len(INTERNAL)
    eta = a[None, :] + X @ B.T
    if h is not None and g is not None:
        eta = eta + h[:, None] * (g * MEAS_EDGES)[None, :]
    rates = torch.exp(eta.clamp(-20.0, 6.0))  # (N,E); cap avoids overflow
    N = X.shape[0]
    Q = torch.zeros(N, n, n)
    Q[:, E_FROM, E_TO] = rates
    Q = Q - torch.diag_embed(Q.sum(-1))
    P = torch.linalg.matrix_exp(Q * dt[:, None, None])
    return P.clamp_min(1e-12)


class DynModel:
    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, d, maxiter=300, init=None, O=None):
        if O is None:
            O = design.last_observed
        self.mask = torch.tensor(edge_mask(X, O, d))
        X = torch.tensor(X)
        o, y, dt = tensors(d)
        p = X.shape[1]
        if init is None:
            a = torch.full((len(EDGES),), 0.0, requires_grad=True)
            B = torch.zeros(len(EDGES), p, requires_grad=True)
        else:
            a = init[0].clone().requires_grad_(True)
            B = init[1].clone().requires_grad_(True)
        g = torch.zeros(len(EDGES), requires_grad=True)
        h = meas(d)
        opt = torch.optim.LBFGS([a, B, g], lr=1, max_iter=maxiter, tolerance_grad=1e-7,
                                tolerance_change=1e-10, history_size=50, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            P = transition_probs(a, B * self.mask, X, dt, h, g)
            nll = -torch.log(P[torch.arange(len(y)), o, y]).sum()
            lam = torch.as_tensor(self.lam, dtype=B.dtype)
            lam = lam[:, None] if lam.ndim == 1 else lam          # optional per-edge penalty
            loss = nll + (lam * B ** 2).sum() + 1e-4 * (a ** 2).sum() + 1e-4 * (g ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.a, self.B, self.g = a.detach(), (B * self.mask).detach(), (g * MEAS_EDGES).detach()
        return self

    def predict_display(self, X, d):
        o, _, dt = tensors(d)
        with torch.no_grad():
            P = transition_probs(self.a, self.B, torch.tensor(X), dt, meas(d), self.g)
            row = P[torch.arange(len(o)), o]
        row = row.numpy()
        out = row[:, :5].copy()
        out[:, 0] += row[:, 5] + row[:, 6]                        # D and NRe shown as NR
        return out / out.sum(1, keepdims=True)

    def curve(self, x, start, times, h=1.0):
        """Displayed-state probabilities over time for one profile.
        h=1: breakups recorded as NR (adult-couple survey convention, matches the 5-state definition);
        h=0: college-dating-study convention (breakups often recorded as staying acquaintances/friends)."""
        X = torch.tensor(np.asarray(x, float)[None, :]).repeat(len(times), 1) if np.size(x) else torch.zeros(len(times), 0)
        dt = torch.tensor(np.asarray(times, float))
        with torch.no_grad():
            P = transition_probs(self.a, self.B, X, dt, torch.full((len(times),), float(h)), self.g)[:, IDX[start]].numpy()
        out = P[:, :5].copy(); out[:, 0] += P[:, 5] + P[:, 6]
        return out


HC_EDGES = [e for e, (a, b) in enumerate(EDGES) if a in ("CR", "M")]


def edge_lams(lam_ew, lam_hc):
    """Penalty per edge: early-stage edges (Eastwick data) vs couple/marriage edges (HCMST data)."""
    return np.array([lam_hc if e in HC_EDGES else lam_ew for e in range(len(EDGES))], float)


def display_nll(probs, d):
    y = d.dest_display.map({s: i for i, s in enumerate(DISPLAY)}).to_numpy()
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1))
