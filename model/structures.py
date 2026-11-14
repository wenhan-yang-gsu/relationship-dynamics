# Latent classes (finite mixture over intensity shifts), restricted cubic splines, origin filtering for histories without a dating module, and monthly NLSY97 observations.
import os
import numpy as np, torch
from chain import KINDS
from sequences import build, loglik, penalty
from common import forecast, featurize, SOURCES
import pandas as pd
from chain import edge_rates, propagate_sparse
from sequences import initial, SOURCES
from common import person_probs

DEATH = torch.tensor([k in ("D>X", "C>X", "M>X") for k in KINDS])
LAM_DELTA = 0.1

class Shifted:

    def __init__(self, base, delta):
        self.sp, self.p, self.n_src = base.sp, base.p, base.n_src
        self.alpha = base.alpha + torch.where(DEATH, torch.zeros_like(delta), delta)[:, None]
        self.gamma, self.spline, self.beta, self.order, self.xtag = base.gamma, base.spline, base.beta, base.order, base.xtag
        self.age_range = getattr(base, "age_range", None)
        self.age_linear = getattr(base, "age_linear", False)

class Mixture:
    def __init__(self, par, G, delta=None, W=None):
        self.par, self.G = par, G
        if delta is None:

            init = torch.linspace(-0.6, 0.6, G - 1) if G > 2 else torch.tensor([0.5])
            delta = init[:, None].repeat(1, len(KINDS)).double()

        self.delta = torch.as_tensor(delta, dtype=torch.float64).clone().requires_grad_(True)
        self.W = (torch.zeros(G - 1, 4, dtype=torch.float64) if W is None else torch.as_tensor(W, dtype=torch.float64).clone()).requires_grad_(True)

    def tensors(self):
        return self.par.tensors() + [self.delta, self.W]

    def chains(self):
        return [self.par] + [Shifted(self.par, self.delta[g]) for g in range(self.G - 1)]

    def log_prior(self, z):
        lg = torch.cat([torch.zeros(len(z), 1, dtype=torch.float64), z @ self.W.T], 1)
        return torch.log_softmax(lg, 1)

    def class_ll(self, b):
        return torch.stack([loglik(c, b, per_seq=True) for c in self.chains()], 1)

    def ll(self, b, z):
        return torch.logsumexp(self.log_prior(z) + self.class_ll(b), 1)

    def posterior(self, b, z):
        with torch.no_grad():
            a = self.log_prior(z) + self.class_ll(b)
            return torch.softmax(a, 1).numpy()

    def state(self):
        return {"par": self.par.state_dict(), "delta": self.delta.detach(), "W": self.W.detach(), "G": self.G}

def z_from(female_std, src):

    src = torch.as_tensor(src)
    return torch.stack([torch.ones(len(src), dtype=torch.float64), torch.as_tensor(female_std, dtype=torch.float64),
                        (src == SOURCES.index("HCMST2009")).double(), (src == SOURCES.index("HCMST2017")).double()], 1)

def seq_z(b, xcols):
    return z_from(b.x0[:, xcols.index("female")], b.src)

def fit_mixture(m, b, xcols, pen, maxiter=200, verbose=False):
    z = seq_z(b, xcols)
    opt = torch.optim.LBFGS(m.tensors(), lr=1, max_iter=maxiter, history_size=30,
                            tolerance_grad=1e-6, tolerance_change=1e-9, line_search_fn="strong_wolfe")
    it = [0]
    def closure():
        opt.zero_grad()
        nll = -m.ll(b, z).sum()
        loss = nll + penalty(m.par, **pen) + LAM_DELTA * (m.delta ** 2).sum() + 1e-2 * (m.W ** 2).sum()
        loss.backward(); it[0] += 1
        if verbose and it[0] % 10 == 0:
            print(f"  eval {it[0]}: nll {nll.item():.2f}", flush=True)
        return loss
    opt.step(closure)
    return m

def objective(m, b, xcols, pen):

    with torch.no_grad():
        return float(-m.ll(b, seq_z(b, xcols)).sum() + penalty(m.par, **pen) + LAM_DELTA * (m.delta ** 2).sum()
                     + 1e-2 * (m.W ** 2).sum())

def random_start(G, seed, sd=0.5):

    g = torch.Generator().manual_seed(seed)
    return torch.randn(G - 1, len(KINDS), generator=g, dtype=torch.float64) * sd

def origin_posterior(m, panel, O, sp, xcols, stats, use_order=True):

    hist = panel.merge(O[["pid", "t_years"]].drop_duplicates("pid").rename(columns={"t_years": "t0"}), on="pid")
    hist = hist[hist.t_years <= hist.t0 + 1e-9].drop(columns="t0")
    fem = featurize(O, xcols, stats)[:, xcols.index("female")]
    src = torch.tensor([SOURCES.index(c) for c in O.cohort])
    with torch.no_grad():
        prior = torch.softmax(m.log_prior(z_from(fem, src)), 1).numpy()
    post = prior.copy()
    if len(hist):
        bh, _ = build(hist, sp, xcols, stats, use_order=use_order)
        if bh is not None and len(bh.s0):
            q = m.posterior(bh, seq_z(bh, xcols))
            pos = {p: i for i, p in enumerate(bh.pid)}
            for i, p in enumerate(O.pid.to_numpy()):
                if p in pos:
                    post[i] = q[pos[p]]
    return post

def forecast_mixture(m, O, xcols, stats, elapsed, post, use_order=True):
    out = 0
    for g, c in enumerate(m.chains()):
        pp, _ = forecast(c, O, xcols, stats, elapsed, use_order=use_order)
        out = out + post[:, [g]] * pp.numpy()
    return out

SMOOTH_COLS = ["edu_years", "log_income", "afqt", "mom_edu"]
PCTS = [5, 35, 65, 95]

NAMES = []

def smooth_names():
    return list(NAMES)

def knots_from(train_panel, cols=SMOOTH_COLS):

    first = train_panel[train_panel.age >= 18].sort_values(["pid", "t_years"]).groupby("pid").head(1)
    out = {}
    for c in cols:
        v = first[c].to_numpy(float)
        k = np.unique(np.nanpercentile(v, PCTS))
        if len(k) < 4:
            k = np.unique(np.nanpercentile(v, [5, 50, 95]))
        out[c] = k
    return out

def rcs(x, k):

    x = np.asarray(x, float)
    s = (k[-1] - k[0]) ** 2
    p = lambda u: np.where(np.isnan(u), np.nan, np.maximum(u, 0) ** 3)
    out = []
    for j in range(len(k) - 2):
        t = (p(x - k[j]) - p(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
             + p(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2]))
        out.append(t / s)
    return np.stack(out, 1)

def add_smooth(panel, knots):

    panel = panel.copy()
    names = []
    NAMES.clear()
    for c, k in knots.items():
        B = rcs(panel[c].to_numpy(float), k)
        for j in range(B.shape[1]):
            n = f"{c}__s{j + 1}"
            panel[n] = B[:, j]
            names.append(n)
    NAMES.extend(names)
    return panel, names

def beta_weights(xcols, mult):

    w = np.array([mult if "__s" in c else 1.0 for c in xcols])
    return np.concatenate([w, w])

def forward(par, b):

    sp = b.sp
    v = initial(par, b)
    ll = torch.zeros(len(b.s0), dtype=torch.float64)
    R = sp.reroot
    for t in range(int(b.iv_step.max()) + 1):
        sel = (b.iv_step == t).nonzero()[:, 0]
        if len(sel) == 0:
            continue
        seq = b.iv_seq[sel]
        w = propagate_sparse(v[seq], edge_rates(par, b.x0[seq], b.src[seq], b.iv_agemid[sel], b.order0[seq]), sp, b.iv_dt[sel])
        wm = w * b.iv_mask[sel]
        c = wm.sum(-1)
        bad = c < 1e-250
        wm = torch.where(bad[:, None], w, wm); c = torch.where(bad, torch.ones_like(c), c)
        ll = ll.index_add(0, seq, torch.log(c))
        v = v.index_copy(0, seq, (wm / c[:, None]) @ R)
    return ll, v

def propagate_forecast(par, v, x, src, age, order, elapsed):

    sp = par.sp
    e = np.asarray(elapsed, float)
    out = torch.zeros_like(v)
    k = 0
    while True:
        due = np.where((e >= k) & (e < k + 1))[0]
        if len(due):
            d = torch.tensor(due)
            r_ = edge_rates(par, x[d], src[d], age[d] + k + (torch.tensor(e[due]) - k) / 2, order[d])
            out[d] = propagate_sparse(v[d], r_, sp, torch.tensor(e[due] - k))
        if (e < k + 1).all():
            break
        live = torch.tensor(np.where(e >= k + 1)[0])
        r_ = edge_rates(par, x[live], src[live], age[live] + k + 0.5, order[live])
        v = v.clone(); v[live] = propagate_sparse(v[live], r_, sp, torch.ones(len(live)))
        k += 1
    return person_probs(sp, out)

def origin_filter(chains, log_prior, panel, O, sp, xcols, stats, pD, med_dur, use_order=True):

    O = O.reset_index(drop=True)
    if O.pid.duplicated().any():
        u = O.drop_duplicates("pid").reset_index(drop=True)
        ui = {p: i for i, p in enumerate(u.pid)}
        rows = np.array([ui[p] for p in O.pid])
        Vu, pu = origin_filter(chains, np.asarray(log_prior)[O.pid.drop_duplicates().index.to_numpy()], panel, u, sp,
                               xcols, stats, np.asarray(pD)[O.pid.drop_duplicates().index.to_numpy()], med_dur, use_order)
        return Vu[:, torch.tensor(rows)], pu[rows]
    G, N = len(chains), len(O)
    hist = panel.merge(O[["pid", "t_years"]].rename(columns={"t_years": "t0"}), on="pid")
    hist = hist[(hist.t_years <= hist.t0 + 1e-9) & (hist.age >= 18)].drop(columns="t0").sort_values(["pid", "t_years"])
    hist = hist.copy()
    hist.loc[~hist.cohort.isin(SOURCES), "cohort"] = "NLSY97"
    for s in ("C", "M"):
        m = (hist.state == s) & hist.dur_state_yrs.isna()
        hist.loc[m, "dur_state_yrs"] = med_dur[s]
    first = hist.groupby("pid").head(1).set_index("pid").state
    lw = {}
    starts = {"S": np.log(np.clip(1 - pD, 1e-12, 1)), "D": np.log(np.clip(pD, 1e-12, 1))}
    pos = {p: i for i, p in enumerate(O.pid)}
    V = torch.zeros(G, N, sp.n, dtype=torch.float64)
    LW = torch.full((G, N), -np.inf, dtype=torch.float64)
    for letter in ("S", "D", None):
        h = hist.copy()
        if letter is None:
            rows = ~h.pid.map(first).eq("N")
            base_w = np.zeros(N)
        else:
            rows = h.pid.map(first).eq("N")
            h.loc[rows & h.pid.duplicated(keep="first").eq(False), "state"] = letter
            base_w = starts[letter]
        h = h[rows.to_numpy()]
        if len(h) == 0:
            continue
        b, _ = build(h, sp, xcols, stats, use_order=use_order)
        idx_multi = set()
        if b is not None and len(b.s0):
            idx = torch.tensor([pos[p] for p in b.pid])
            idx_multi = set(b.pid)
            for g, c in enumerate(chains):
                with torch.no_grad():
                    ll, v = forward(c, b)
                V[g, idx] = V[g, idx] + torch.exp(torch.tensor(base_w)[idx] + ll)[:, None] * v
                LW[g, idx] = torch.logaddexp(LW[g, idx], torch.tensor(base_w)[idx] + ll)

        single = [p for p in h.pid.unique() if p not in idx_multi]
        if single:
            o1 = h[h.pid.isin(single)].groupby("pid").head(1)
            o1 = o1.assign(_k=o1.pid.map(pos)).sort_values("_k")
            fake = pd.concat([o1, o1.assign(t_years=o1.t_years + 1e-6)])
            b1, _ = build(fake.sort_values(["pid", "t_years"]), sp, xcols, stats, use_order=use_order)
            idx1 = torch.tensor([pos[p] for p in b1.pid])
            for g, c in enumerate(chains):
                v0 = initial(c, b1)
                v0 = v0 @ sp.reroot
                V[g, idx1] = V[g, idx1] + torch.exp(torch.tensor(base_w)[idx1])[:, None] * v0
                LW[g, idx1] = torch.logaddexp(LW[g, idx1], torch.tensor(base_w)[idx1])
    Vn = V / V.sum(-1, keepdim=True)
    post = torch.softmax(torch.as_tensor(log_prior, dtype=torch.float64) + LW.T, 1).numpy()
    return Vn, post

def forecast_filtered(chains, Vn, post, O, xcols, stats, elapsed, use_order=True):
    O = O.reset_index(drop=True)
    x = featurize(O, xcols, stats)
    src = torch.tensor([SOURCES.index(c) if c in SOURCES else 0 for c in O.cohort])
    age = torch.tensor(O.age.to_numpy(float))
    order = torch.tensor(np.nan_to_num(O.prior_union.to_numpy(float))) if use_order else torch.zeros(len(O))
    out = 0
    for g, c in enumerate(chains):
        with torch.no_grad():
            pp = propagate_forecast(c, Vn[g], x, src, age, order, elapsed)
        out = out + post[:, [g]] * pp.numpy()
    return out

ROOT = os.environ.get("LOVE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_M = None

def _arrays():
    global _M
    if _M is None:
        z = np.load(f"{ROOT}/cache/nlsy97_monthly.npz", allow_pickle=True)
        _M = dict(pos={p: i for i, p in enumerate(z["pubid"])}, months=z["months"], state=z["state"], partner=z["partner"])
    return _M

def _same(s0, c0, s1, c1):
    if s0 == s1 == "C" and 101 <= c0 <= 111 and 101 <= c1 <= 111:
        return float(c0 == c1)
    if s0 == s1 == "M" and 201 <= c0 <= 210 and 201 <= c1 <= 210:
        return float(c0 == c1)
    return np.nan

def augment(panel, step=3):

    Z = _arrays()
    m0 = int(Z["months"][0])
    out = []
    n_added = 0
    n_skip = [0]
    for pid, g in panel.groupby("pid", sort=False):
        g = g.sort_values("t_years")
        if g.cohort.iat[0] != "NLSY97" or len(g) < 2:
            out.append(g.assign(added=0)); continue
        i = Z["pos"][int(g.pubid.iat[0])]
        st, pc = Z["state"][i], Z["partner"][i]
        code = lambda cm: (st[cm - m0], pc[cm - m0]) if 0 <= cm - m0 < len(st) else ("", np.nan)
        rows = [g.iloc[[0]].assign(added=0)]
        recs = g.to_dict("records")
        for a, b in zip(recs[:-1], recs[1:]):
            ca, cb = int(a["cmonth"]), int(b["cmonth"])
            add = [cm for cm in range(ca + 1, cb) if (cm - 1) % 12 % step == 0]
            new = []
            prev_s, prev_c = code(ca)
            if prev_s != a["state"]:
                prev_c = np.nan
            prev_s = a["state"]
            prev_obs = a["state"]
            for cm in add:
                s, c = code(cm)
                if s == "":
                    continue
                if prev_obs == "M" and s == "C":

                    n_skip[0] += 1
                    continue
                r = dict(b)
                r.update(cmonth=cm, t_years=(cm - 0.5) / 12, age=a["age"] + (cm - ca) / 12, state=s,
                         same_partner=_same(prev_s, prev_c, s, c), end_tag="", wave=a["wave"] + 0.5, year=1980 + (cm - 1) // 12)
                r["ended_here"] = (prev_obs in ("C", "M") and s == "N") or r["same_partner"] == 0
                new.append(r); prev_s, prev_c, prev_obs = s, c, s
            b = dict(b)
            if new:
                chain_same = all(r["same_partner"] == 1 for r in new) or (b["state"] in ("D", "N") and all(r["state"] == "N" for r in new))
                if not chain_same:
                    sb, cb_ = code(cb)
                    b["same_partner"] = _same(prev_s, prev_c, sb, cb_) if sb == b["state"] else np.nan
                tag = b.get("end_tag", "")
                if isinstance(tag, str) and tag in ("B", "X"):
                    hit = [r for r in new if r["ended_here"]]
                    if hit:
                        hit[0]["end_tag"] = tag; b["end_tag"] = ""
                for r in new:
                    r.pop("ended_here")
                n_added += len(new)
                rows.append(pd.DataFrame(new).assign(added=1))
            rows.append(pd.DataFrame([b]).assign(added=0))
        out.append(pd.concat(rows, ignore_index=True))
    res = pd.concat(out, ignore_index=True)
    augment.n_added = n_added
    augment.n_skipped = n_skip[0]
    return res
