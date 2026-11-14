# State space, transition intensities and sparse uniformization of the continuous-time Markov chain.
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

KINDS = ["S>D", "S>C", "S>M", "D>C", "D>M", "D>B", "D>X", "C>M", "C>D", "C>B", "C>X", "M>B", "M>X"]
KIND_IDX = {k: i for i, k in enumerate(KINDS)}
AGE_KNOTS = [20.0, 25.0, 35.0, 50.0, 65.0]

@dataclass
class Space:
    K: dict = field(default_factory=lambda: {"D": 4, "C": 4, "M": 2})

    def __post_init__(self):
        names = []
        for s in "DCM":
            names += [f"R-{s}{k}" for k in range(1, self.K[s] + 1)]
        for tag in "BX":
            names.append(f"{tag}-S")
            for s in "DCM":
                names += [f"{tag}-{s}{k}" for k in range(1, self.K[s] + 1)]
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}
        self.n = len(names)

        E = []
        def block_edges(pre, end_tag):
            for s in "DCM":
                Ks = self.K[s]
                for k in range(1, Ks + 1):
                    fr = self.idx[f"{pre}{s}{k}"]
                    if k < Ks:
                        E.append((fr, self.idx[f"{pre}{s}{k+1}"], None, k - 1, pre != "R-"))
                    nxt = {"D": ["C", "M"], "C": ["M", "D"], "M": []}[s]
                    for t in nxt:

                        E.append((fr, self.idx[f"{pre}{t}1"], KIND_IDX[f"{s}>{t}"], k - 1, pre != "R-"))
                    for end in "BX":
                        tag = end if end_tag is None else end_tag
                        E.append((fr, self.idx[f"{tag}-S"], KIND_IDX[f"{s}>{end}"], k - 1, pre != "R-"))
        block_edges("R-", None)
        for tag in "BX":
            block_edges(f"{tag}-", tag)
            for t in "DCM":
                E.append((self.idx[f"{tag}-S"], self.idx[f"{tag}-{t}1"], KIND_IDX[f"S>{t}"], 0, True))
        self.edges = E
        self.e_from = torch.tensor([e[0] for e in E])
        self.e_to = torch.tensor([e[1] for e in E])
        self.e_kind = torch.tensor([-1 if e[2] is None else e[2] for e in E])
        self.e_phase = torch.tensor([e[3] for e in E])
        self.e_cont = torch.tensor([float(e[4]) for e in E])
        self.e_fixed = self.e_kind < 0
        self.e_xtag = torch.tensor([float(self.names[e[0]] == "X-S") for e in E])
        self.maxK = max(self.K.values())

        R = torch.zeros(self.n, self.n)
        for i, nm in enumerate(names):
            if nm.endswith("-S"):
                R[i, i] = 1
            else:
                R[i, self.idx["R-" + nm.split("-")[1]]] = 1
        self.reroot = R

    def mask(self, obs, same=None, tag=None):

        m = np.zeros(self.n, bool)
        tags = [tag] if tag else ["B", "X"]
        def ref(s):
            for k in range(1, self.K[s] + 1):
                m[self.idx[f"R-{s}{k}"]] = True
        def cont(s, tg=tags):
            for t in tg:
                if s == "S":
                    m[self.idx[f"{t}-S"]] = True
                else:
                    for k in range(1, self.K[s] + 1):
                        m[self.idx[f"{t}-{s}{k}"]] = True
        if obs in ("D", "C", "M"):
            if same != 0:
                ref(obs)
            if same != 1:
                cont(obs)
        elif obs == "S":
            cont("S")
        elif obs == "N":
            cont("S"); cont("D")
            if same != 0:
                ref("D")
        elif obs == "U":
            ref("D"); ref("C")
        elif obs == "E":
            for s in "SDCM":
                cont(s)
        else:
            raise ValueError(obs)
        return m

    def phase_block(self, s):
        return [self.idx[f"R-{s}{k}"] for k in range(1, self.K[s] + 1)]

def age_basis(a):
    a = torch.as_tensor(a)
    cols = [(a - 30.0) / 10.0] + [torch.clamp(a - k, min=0) / 10.0 for k in AGE_KNOTS]
    return torch.stack(cols, -1)

class Params:
    def __init__(self, sp: Space, p: int, n_src: int, init=None):
        nk, ns = len(KINDS), len(AGE_KNOTS) + 1
        self.sp, self.p, self.n_src = sp, p, n_src
        self.alpha = torch.full((nk, sp.maxK), -2.0, requires_grad=True)
        self.gamma = torch.zeros(nk, n_src, requires_grad=True)
        self.spline = torch.zeros(nk, ns, requires_grad=True)
        self.beta = torch.zeros(nk, p, requires_grad=True)
        self.order = torch.zeros(nk, requires_grad=True)
        self.xtag = torch.zeros(3, requires_grad=True)
        if init is not None:
            for k, v in init.items():
                getattr(self, k).data.copy_(torch.as_tensor(v))

    def tensors(self):
        return [self.alpha, self.gamma, self.spline, self.beta, self.order, self.xtag]

    def state_dict(self):
        return {k: getattr(self, k).detach().clone() for k in ("alpha", "gamma", "spline", "beta", "order", "xtag")}

DEATH_KINDS = [KIND_IDX[k] for k in ("D>X", "C>X", "M>X")]
DEATH_EDGES = torch.zeros(0, dtype=torch.bool)

def generator(par: Params, x, src, age, order_ref, cap=3.0):

    sp = par.sp
    rate = edge_rates(par, x, src, age, order_ref, cap)
    N, E = rate.shape
    Q = torch.zeros(N, sp.n, sp.n)
    Q = Q.index_put((torch.arange(N)[:, None].expand(N, E), sp.e_from[None, :].expand(N, E), sp.e_to[None, :].expand(N, E)),
                    rate, accumulate=True)
    Q = Q - torch.diag_embed(Q.sum(-1))
    return Q

def edge_rates(par: Params, x, src, age, order_ref, cap=3.0):

    sp = par.sp
    E = len(sp.edges)
    kind = sp.e_kind.clamp(min=0)

    age_raw = torch.as_tensor(age, dtype=torch.float64)
    if getattr(par, "age_range", None) is not None:
        age = age_raw.clamp(*par.age_range)
    global DEATH_EDGES
    DEATH_EDGES = torch.tensor([int(k) in DEATH_KINDS for k in sp.e_kind])
    age_eff = age_basis(age) @ par.spline.T
    if getattr(par, "age_linear", False) and getattr(par, "age_range", None) is not None:

        lo, hi = par.age_range
        s_hi = (age_basis(torch.tensor([hi], dtype=torch.float64)) - age_basis(torch.tensor([hi - 1.0], dtype=torch.float64))) @ par.spline.T
        s_lo = (age_basis(torch.tensor([lo + 1.0], dtype=torch.float64)) - age_basis(torch.tensor([lo], dtype=torch.float64))) @ par.spline.T
        age_eff = age_eff + s_hi * (age_raw - hi).clamp(min=0)[:, None] + s_lo * (age_raw - lo).clamp(max=0)[:, None]
    eta = (par.alpha[kind, sp.e_phase][None, :]
           + par.gamma[kind][:, src].T
           + age_eff[:, kind]
           + (x @ par.beta.T)[:, kind]
           + par.order[kind][None, :] * torch.where(sp.e_cont.bool()[None, :], torch.ones(1), order_ref[:, None]))
    s_kinds = {KIND_IDX["S>D"]: 0, KIND_IDX["S>C"]: 1, KIND_IDX["S>M"]: 2}
    sel = (sp.e_xtag > 0) & (sp.e_kind >= 0)
    if sel.any():
        xt_idx = torch.tensor([s_kinds[int(k)] for k in sp.e_kind[sel]])
        eta = eta.clone()
        eta[:, sel] = eta[:, sel] + par.xtag[xt_idx][None, :]

    if DEATH_EDGES.any():
        eta_d = par.alpha[kind, 0][None, :] + age_eff[:, kind]
        eta = torch.where(DEATH_EDGES[None, :], eta_d, eta)
    rate = torch.exp(eta.clamp(max=cap))
    rate = torch.where(sp.e_fixed[None, :], torch.full((1,), float(getattr(sp, "prog_rate", 1.0))), rate)
    return rate

def propagate_sparse(v, rate, sp, t, tol_sd=6.0, buckets=4):

    N = v.shape[0]
    outflow = torch.zeros(N, sp.n, dtype=rate.dtype).index_add(1, sp.e_from, rate)
    L = outflow.max(1).values.detach().clamp(min=1e-3)
    lt_all = (L * t).detach().numpy()
    if len(lt_all) < 2000 or buckets == 1:
        groups = [np.arange(N)]
    else:
        edges = np.quantile(lt_all, np.linspace(0, 1, buckets + 1)[1:-1])
        g = np.searchsorted(edges, lt_all)
        groups = [np.where(g == k)[0] for k in range(buckets)]
    out = torch.zeros_like(v)
    for idx in groups:
        if len(idx) == 0:
            continue
        r = torch.tensor(idx)
        vr, rr, of, Lr, tr = v[r], rate[r], outflow[r], L[r], t[r]
        lt = Lr * tr
        m = float(lt.max())
        kmax = int(math.ceil(m + tol_sd * math.sqrt(m) + 10))
        llt = torch.log(lt.clamp_min(1e-300))
        logw = -lt
        w = vr
        acc = torch.exp(logw)[:, None] * w
        for k in range(1, kmax + 1):
            inflow = torch.zeros_like(w).index_add(1, sp.e_to, w[:, sp.e_from] * rr)
            w = w + (inflow - w * of) / Lr[:, None]
            logw = logw + llt - math.log(k)
            acc = acc + torch.exp(logw)[:, None] * w
        out = out.index_copy(0, r, acc)
    return out

def _unif(v, Q, t, tol_sd):
    L = (-torch.diagonal(Q, dim1=-2, dim2=-1)).max(-1).values.detach().clamp(min=1e-3)
    lt = L * t
    P = torch.eye(Q.shape[-1])[None] + Q / L[:, None, None]
    m = float(lt.max())
    kmax = int(math.ceil(m + tol_sd * math.sqrt(m) + 10))
    logw = -lt
    w = v
    out = torch.exp(logw)[:, None] * w
    llt = torch.log(lt.clamp_min(1e-300))
    for k in range(1, kmax + 1):
        w = torch.bmm(w[:, None, :], P)[:, 0, :]
        logw = logw + llt - math.log(k)
        out = out + torch.exp(logw)[:, None] * w
    return out

def propagate(v, Q, t, tol_sd=6.0, buckets=4):

    L = (-torch.diagonal(Q, dim1=-2, dim2=-1)).max(-1).values.detach()
    lt = (L * t).numpy()
    if len(lt) < 2000 or buckets == 1:
        return _unif(v, Q, t, tol_sd)
    edges = np.quantile(lt, np.linspace(0, 1, buckets + 1)[1:-1])
    grp = np.searchsorted(edges, lt)
    out = torch.zeros_like(v)
    for g in range(buckets):
        r = torch.tensor(np.where(grp == g)[0])
        if len(r):
            out = out.index_copy(0, r, _unif(v[r], Q[r], t[r], tol_sd))
    return out

def delayed_entry(par: Params, s, d, x, src, age, order_ref, grid=None):

    sp = par.sp
    idx = sp.phase_block(s)
    N = x.shape[0]
    Q = generator(par, x, src, age - torch.nan_to_num(d, nan=0.0) / 2, order_ref)
    Qs = Q[:, idx][:, :, idx]
    v0 = torch.zeros(N, len(idx)); v0[:, 0] = 1.0
    dd = torch.nan_to_num(d, nan=float(grid if grid is not None else 3.0)).clamp(min=0)

    v = propagate_sub(v0, Qs, dd)
    v = v / v.sum(-1, keepdim=True).clamp_min(1e-300)
    return v

def propagate_sub(v, Q, t, tol_sd=6.0):

    return _unif(v, Q, t, tol_sd)
