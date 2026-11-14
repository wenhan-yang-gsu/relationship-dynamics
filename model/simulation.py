# Simulation on the real interview skeleton: four generating designs, refit of four structures and the inclusion rule.
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from chain import Params, generator, propagate, delayed_entry
from sequences import Batch, SOURCES
import sys, os, json, time
import common as C, cv, structures as LT
from sequences import build

DT = 1.0 / 12.0

def simulate(par: Params, panel: pd.DataFrame, b: Batch, seed=0):

    sp = par.sp
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    names = np.array(sp.names)
    real = {pid: g.reset_index(drop=True) for pid, g in panel.sort_values(["pid", "t_years"]).groupby("pid", sort=False)}
    P = len(b.s0)

    starts = []
    for pid in b.pid:
        g = real[pid]
        ok = (g.age >= 18) & g.state.isin(["S", "D", "C", "M"])
        starts.append(g.loc[ok.idxmax():].reset_index(drop=True))

    with torch.no_grad():
        state = np.zeros(P, int)
        s0 = np.array(b.s0)
        state[s0 == "S"] = sp.idx["B-S"]
        for s in "DCM":
            r = np.where(s0 == s)[0]
            if len(r) == 0:
                continue
            dist = delayed_entry(par, s, b.d0[r], b.x0[r], b.src[r], b.age0[r], b.order0[r]).numpy()
            cols = np.array(sp.phase_block(s))
            u = rng.random(len(r))[:, None]
            state[r] = cols[(dist.cumsum(1) < u).sum(1).clip(max=len(cols) - 1)]
    t_now = np.array([g.t_years.iat[0] for g in starts])
    age_now = np.array([g.age.iat[0] for g in starts])
    ptr = np.ones(P, int)
    alive = np.array([len(g) > 1 for g in starts])
    out_rows = [[dict(g.iloc[0])] for g in starts]
    for i, g in enumerate(starts):
        out_rows[i][0]["same_partner"] = np.nan
        out_rows[i][0]["end_tag"] = ""
    stopped = np.zeros(P, bool)
    with torch.no_grad():
        while alive.any():
            act = np.where(alive)[0]
            Q = generator(par, b.x0[act], b.src[act], torch.tensor(age_now[act] + DT / 2), b.order0[act])
            v = torch.zeros(len(act), sp.n)
            v[torch.arange(len(act)), torch.tensor(state[act])] = 1.0
            pr = propagate(v, Q, torch.full((len(act),), DT)).numpy().clip(min=0)
            pr = pr / pr.sum(1, keepdims=True)
            u = rng.random(len(act))[:, None]
            state[act] = (pr.cumsum(1) < u).sum(1).clip(max=sp.n - 1)
            t_now[act] += DT
            age_now[act] += DT

            for j in act:
                g = starts[j]
                while alive[j] and ptr[j] < len(g) and g.t_years.iat[ptr[j]] <= t_now[j] + 1e-9:
                    k = ptr[j]
                    real_row = g.iloc[k]
                    nm = names[state[j]]
                    blk, st = nm.split("-")
                    st = st[0]
                    prev_state = out_rows[j][-1]["state"]
                    if blk == "R":
                        obs, same, tag = st, 1.0, ""
                    else:
                        obs = st
                        same = 0.0 if st != "S" else np.nan
                        ended = prev_state in ("D", "C", "M", "U")
                        tag = blk if ended else ""

                    src = real_row.cohort
                    if src == "NLSY97" and tag and prev_state != "M":
                        tag = ""
                    if real_row.state == "N" and obs in ("S", "D"):
                        obs = "N"
                    if pd.isna(real_row.same_partner):
                        same = np.nan
                    if real_row.state == "U" and obs in ("D", "C") and same == 1:
                        obs = "U"
                    if src == "HCMST2009" and blk != "R":
                        obs, same = "E", 0.0
                        tag = blk
                    row = dict(real_row)
                    row.update(state=obs, same_partner=same, end_tag=tag)
                    out_rows[j].append(row)

                    state[j] = int(torch.argmax(sp.reroot[state[j]]))
                    ptr[j] += 1
                    if obs == "E":
                        alive[j] = False
                if ptr[j] >= len(g):
                    alive[j] = False
    return pd.DataFrame([r for rows in out_rows for r in rows])

DESIGNS = ("neither", "duration", "heterogeneity", "both")
STRUCT = ("base", "lat", "phase", "phase_lat")
SIMPLE = {"base": 0, "lat": 1, "phase": 1, "phase_lat": 2}

def load_par(path, sp, p):
    ck = torch.load(path, weights_only=False)
    par = Params(sp, p=p, n_src=3, init=ck["par"]); par.age_range = ck["age_range"]
    return par, ck

def truth_models(sp1, spP, p):
    base, ckb = load_par(f"{cv.OUT}/base_f0.pt", sp1, p)
    phase, _ = load_par(f"{cv.OUT}/phase_f0.pt", spP, p)
    lat = torch.load(f"{cv.OUT}/lat_f4.pt", weights_only=False)
    assert lat["G"] == 2
    het = LT.Mixture(base, 2, delta=lat["delta"], W=lat["W"])
    both = LT.Mixture(phase, 2, delta=lat["delta"], W=lat["W"])
    return {"neither": base, "duration": phase, "heterogeneity": het, "both": both}, ckb

def simulate_panel(truth, panel, sp, xc, stats, seed):

    b, _ = build(panel, sp, xc, stats)
    if not isinstance(truth, LT.Mixture):
        return simulate(truth, panel, b, seed=seed)
    with torch.no_grad():
        pr = torch.softmax(truth.log_prior(LT.seq_z(b, xc)), 1).numpy()
    rng = np.random.default_rng(seed + 7)
    cls = (pr.cumsum(1) < rng.random(len(pr))[:, None]).sum(1)
    out = []
    for g, chain in enumerate(truth.chains()):
        pids = set(np.array(b.pid)[cls == g])
        sub = panel[panel.pid.isin(pids)]
        if len(sub) == 0:
            continue
        bg, _ = build(sub, sp, xc, stats)
        out.append(simulate(chain, sub, bg, seed=seed + 100 * (g + 1)))
    return pd.concat(out, ignore_index=True)

def select(cells):

    keys = sorted(cells["base"])
    gain = {s: np.mean([cells["base"][k] - cells[s][k] for k in keys]) for s in STRUCT}
    worst = {s: max(cells[s][k] - cells["base"][k] for k in keys) for s in STRUCT}
    ok = [s for s in STRUCT if s == "base" or (gain[s] >= 0.003 and worst[s] <= 0.003)]
    best = min(ok, key=lambda s: -gain[s])
    simpler = [s for s in ok if SIMPLE[s] < SIMPLE[best] and gain[best] - gain[s] < 0.001]
    return (min(simpler, key=lambda s: (SIMPLE[s], -gain[s])) if simpler else best), gain, worst

def main():
    design, rep = sys.argv[1], int(sys.argv[2])
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    torch.set_num_threads(threads)
    maxiter = int(os.environ.get("SIM_MAXITER", "300"))
    t0 = time.time()
    A = C.load()
    A = A[A.split.isin(["train", "valid"])].merge(cv.folds(A), on="household", how="left")
    tr, ev = A[A.fold != 0], A[A.fold == 0]
    if os.environ.get("SIM_SUBSET"):
        n = int(os.environ["SIM_SUBSET"])
        tr = tr[tr.household.isin(tr.household.drop_duplicates().sample(n, random_state=0))]
        ev = ev[ev.household.isin(ev.household.drop_duplicates().sample(max(n // 4, 50), random_state=0))]
    cfg1, cfgP = cv.CONFIGS["base"], cv.CONFIGS["phase"]
    sp1, spP = cv.make_space(cfg1), cv.make_space(cfgP)
    ck = torch.load(f"{cv.OUT}/base_f0.pt", weights_only=False)
    xc, st = ck["xcols"], ck["stats"]
    truths, _ = truth_models(sp1, spP, len(xc) * 2)
    truth = truths[design]
    sp_t = spP if design in ("duration", "both") else sp1
    seed = 1000 * DESIGNS.index(design) + rep
    S_tr = simulate_panel(truth, tr, sp_t, xc, st, seed)
    S_ev = simulate_panel(truth, ev, sp_t, xc, st, seed + 50000)
    for S in (S_tr, S_ev):
        S["end_tag"] = S.end_tag.fillna("")
    print(f"{design} rep {rep}: simulated {len(S_tr)} + {len(S_ev)} rows, {time.time() - t0:.0f}s", flush=True)
    obsm = C.fit_obs_model(S_tr, [(cv.origins(S_tr, lm), H) for lm, H in cv.PRIMARY.items()])
    pen = cv.pen_for(cfg1, xc, None)
    fits = {}
    par_b, _, xb, bb, _ = cv.fit_config(S_tr, cfg1, threads)
    fits["base"] = cv.chain_predict(par_b, xb, st)
    m = LT.Mixture(Params(sp1, p=par_b.p, n_src=3, init=par_b.state_dict()), 2); m.par.age_range = par_b.age_range
    LT.fit_mixture(m, bb, xb, pen, maxiter=maxiter); fits["lat"] = cv.mixture_predict(m, S_ev, xb, st)
    par_p, _, xp, bp, _ = cv.fit_config(S_tr, cfgP, threads, None, par_b.state_dict())
    fits["phase"] = cv.chain_predict(par_p, xp, st)
    mp = LT.Mixture(Params(spP, p=par_p.p, n_src=3, init=par_p.state_dict()), 2); mp.par.age_range = par_p.age_range
    LT.fit_mixture(mp, bp, xp, cv.pen_for(cfgP, xp, None), maxiter=maxiter); fits["phase_lat"] = cv.mixture_predict(mp, S_ev, xp, st)
    print(f"{design} rep {rep}: four fits done, {time.time() - t0:.0f}s", flush=True)
    oracle = cv.mixture_predict(truth, S_ev, xc, st) if isinstance(truth, LT.Mixture) else cv.chain_predict(truth, xc, st)
    cells = {}
    for name, f in list(fits.items()) + [("oracle", oracle)]:
        res, _, pm = cv.score_all(f, S_ev, obsm)
        cells[name] = {f"{lm}_h{h}": res[lm][h]["person"] for lm, H in cv.PRIMARY.items() for h in H}
    chosen, gain, worst = select({s: cells[s] for s in STRUCT})
    truth_struct = {"neither": "base", "duration": "phase", "heterogeneity": "lat", "both": "phase_lat"}[design]
    out = dict(design=design, rep=rep, seed=seed, truth_structure=truth_struct, selected=chosen, correct=chosen == truth_struct,
               gain=gain, worst=worst, cells=cells, seconds=time.time() - t0)
    os.makedirs(f"{C.ROOT}/runs/sim", exist_ok=True)
    json.dump(out, open(f"{C.ROOT}/runs/sim/{design}_r{rep}.json", "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("design", "rep", "selected", "correct")}), {s: round(g, 4) for s, g in gain.items()}, flush=True)

if __name__ == "__main__":
    main()
