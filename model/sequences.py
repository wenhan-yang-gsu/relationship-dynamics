# Hidden-Markov forward recursion over interview sequences with set-valued observations, penalty and L-BFGS fitting.
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from chain import Space, Params, generator, propagate, delayed_entry, edge_rates, propagate_sparse

SOURCES = ["NLSY97", "HCMST2009", "HCMST2017"]

@dataclass
class Batch:
    sp: Space
    x0: torch.Tensor
    src: torch.Tensor
    age0: torch.Tensor
    order0: torch.Tensor
    s0: list
    d0: torch.Tensor

    iv_seq: torch.Tensor
    iv_step: torch.Tensor
    iv_dt: torch.Tensor
    iv_agemid: torch.Tensor
    iv_mask: torch.Tensor
    pid: np.ndarray

def build(panel: pd.DataFrame, sp: Space, xcols, x_stats=None, use_order=True, min_year=None, tv=False):

    panel = panel.sort_values(["pid", "t_years"])
    seqs = []
    for pid, g in panel.groupby("pid", sort=False):
        g = g.reset_index(drop=True)
        ok = (g.age >= 18) & g.state.isin(["S", "D", "C", "M"])
        if min_year is not None and g.cohort.iat[0] == "NLSY97":
            ok &= g.year >= min_year
        if not ok.any():
            continue
        g = g.loc[ok.idxmax():].reset_index(drop=True)
        if len(g) < 2:
            continue
        seqs.append((pid, g))
    if not seqs:
        return None, x_stats
    X = np.array([[g.loc[0, c] for c in xcols] for _, g in seqs], float)
    if x_stats is None:
        with np.errstate(all="ignore"):
            mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        sd = np.where(np.isnan(sd) | (sd == 0), 1.0, sd)
        x_stats = (mu, sd)
    mu, sd = x_stats
    Z = (X - mu) / sd
    miss = np.isnan(Z)
    Z[miss] = 0.0
    Z = np.concatenate([Z, miss.astype(float)[:, [j for j in range(len(xcols)) if miss[:, j].any() or True]]], 1)
    rows = []
    s0, d0, age0, src, order0, pids = [], [], [], [], [], []
    for i, (pid, g) in enumerate(seqs):
        s0.append(g.loc[0, "state"]); d0.append(g.loc[0, "dur_state_yrs"]); age0.append(g.loc[0, "age"])
        src.append(SOURCES.index(g.loc[0, "cohort"])); order0.append(float(np.nan_to_num(g.loc[0].get("prior_union", 0))) if use_order else 0.0); pids.append(pid)
        for k in range(1, len(g)):
            st = g.loc[k, "state"]
            same = g.loc[k, "same_partner"]
            same = None if pd.isna(same) else int(same)
            tag = g.loc[k].get("end_tag", None)
            tag = tag if isinstance(tag, str) and tag in ("B", "X") else None
            dt = g.loc[k, "t_years"] - g.loc[k - 1, "t_years"]
            rows.append((i, k - 1, dt, g.loc[k - 1, "age"] + dt / 2, sp.mask(st, same, tag)))
    iv = sorted(rows, key=lambda r: (r[1], r[0]))
    iv_x = None
    if tv:

        Xs = np.array([[seqs[r[0]][1].loc[r[1], c] for c in xcols] for r in iv], float)
        Zs = (Xs - mu) / sd
        ms = np.isnan(Zs); Zs[ms] = 0.0
        iv_x = torch.tensor(np.concatenate([Zs, ms.astype(float)], 1))
    b = Batch(sp=sp, x0=torch.tensor(Z), src=torch.tensor(src), age0=torch.tensor(age0, dtype=torch.float64),
              order0=torch.tensor(order0, dtype=torch.float64), s0=s0, d0=torch.tensor(np.array(d0, float)),
              iv_seq=torch.tensor([r[0] for r in iv]), iv_step=torch.tensor([r[1] for r in iv]),
              iv_dt=torch.tensor([r[2] for r in iv], dtype=torch.float64),
              iv_agemid=torch.tensor([r[3] for r in iv], dtype=torch.float64),
              iv_mask=torch.tensor(np.stack([r[4] for r in iv])), pid=np.array(pids))
    b.iv_x = iv_x
    return b, x_stats

def initial(par: Params, b: Batch):
    sp, P = b.sp, len(b.s0)
    v = torch.zeros(P, sp.n)
    s0 = np.array(b.s0)
    rows = torch.tensor(np.where(s0 == "S")[0])
    if len(rows):
        v = v.index_put((rows, torch.full((len(rows),), sp.idx["B-S"])), torch.ones(len(rows)))
    for s in "DCM":
        r = torch.tensor(np.where(s0 == s)[0])
        if len(r) == 0:
            continue
        dist = delayed_entry(par, s, b.d0[r], b.x0[r], b.src[r], b.age0[r], b.order0[r])
        cols = torch.tensor(sp.phase_block(s))
        v = v.index_put((r[:, None].expand(-1, len(cols)), cols[None, :].expand(len(r), -1)), dist)
    return v

def loglik(par: Params, b: Batch, per_seq=False, seq_w=None):

    sp = b.sp
    v = initial(par, b)
    ll = torch.zeros(len(b.s0))
    R = sp.reroot
    T = int(b.iv_step.max()) + 1
    for t in range(T):
        sel = (b.iv_step == t).nonzero()[:, 0]
        if len(sel) == 0:
            continue
        seq = b.iv_seq[sel]
        xx = b.iv_x[sel] if getattr(b, "iv_x", None) is not None else b.x0[seq]
        w = propagate_sparse(v[seq], edge_rates(par, xx, b.src[seq], b.iv_agemid[sel], b.order0[seq]), sp, b.iv_dt[sel])
        wm = w * b.iv_mask[sel]
        c = wm.sum(-1)

        bad = c < 1e-250
        if bad.any():
            loglik.n_impossible = getattr(loglik, "n_impossible", 0) + int(bad.sum())
            wm = torch.where(bad[:, None], w, wm)
            c = torch.where(bad, torch.ones_like(c), c)
        ll = ll.index_add(0, seq, torch.log(c))
        w = (wm / c[:, None]) @ R
        v = v.index_copy(0, seq, w)
    if per_seq:
        return ll
    return (ll * seq_w).sum() if seq_w is not None else ll.sum()

def penalty(par: Params, lam_beta=1.0, lam_gamma=1.0, lam_rw=1.0, lam_alpha=1e-3, beta_w=None):

    a = par.alpha
    bw = 1.0 if beta_w is None else torch.as_tensor(beta_w, dtype=par.beta.dtype)[None, :]
    return (lam_beta * (bw * par.beta ** 2).sum() + lam_gamma * (par.gamma ** 2).sum()
            + lam_rw * ((a[:, 1:] - a[:, :-1]) ** 2).sum() + lam_alpha * (a ** 2).sum()
            + 1e-3 * ((par.spline ** 2).sum() + (par.order ** 2).sum() + (par.xtag ** 2).sum()))

def fit(par: Params, b: Batch, pen=None, maxiter=200, verbose=True, seq_w=None):
    pen = pen or {}
    opt = torch.optim.LBFGS(par.tensors(), lr=1, max_iter=maxiter, history_size=30,
                            tolerance_grad=1e-6, tolerance_change=1e-9, line_search_fn="strong_wolfe")
    it = [0]
    def closure():
        opt.zero_grad()
        nll = -loglik(par, b, seq_w=seq_w)
        loss = nll + penalty(par, **pen)
        loss.backward()
        it[0] += 1
        if verbose and it[0] % 10 == 0:
            print(f"  eval {it[0]}: nll {nll.item():.2f}", flush=True)
        return loss
    opt.step(closure)
    return par
