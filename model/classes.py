# The three latent classes: posterior membership and covariate means by class, and ten-year forecasts by origin state and class.
import sys
import torch, numpy as np, pandas as pd
import common as C, cv, structures as LT
from chain import Params
from sequences import build
import json

def profile():

    torch.set_num_threads(4)
    ck = torch.load(f"{cv.OUT}/phase_lat_f-1.pt", weights_only=False)
    sp = cv.make_space(ck["cfg"]); xc, st = ck["xcols"], ck["stats"]
    par = Params(sp, p=len(xc) * 2, n_src=3, init=ck["par"]); par.age_range = ck["age_range"]
    m = LT.Mixture(par, ck["G"], delta=ck["delta"], W=ck["W"])
    A = C.load(); A = A[A.split.isin(["train", "valid"])]
    b, _ = build(A, sp, xc, st)
    with torch.no_grad():
        post = m.posterior(b, LT.seq_z(b, xc))
    P = pd.DataFrame(post, columns=[f"p{g+1}" for g in range(ck["G"])]); P["pid"] = b.pid
    first = A[A.age >= 18].sort_values(["pid", "t_years"]).groupby("pid").head(1).set_index("pid")
    cols = ["female", "black", "hispanic", "edu_years", "log_income", "kids_any", "afqt", "mom_edu", "both_parents12", "parent_relig", "age"]
    D = P.merge(first[cols + ["cohort", "state", "prior_union"]], left_on="pid", right_index=True)
    D["modal"] = D[[f"p{g+1}" for g in range(ck["G"])]].to_numpy().argmax(1) + 1
    D.to_csv(f"{C.ROOT}/runs/class_profile.csv", index=False)
    print("persons", len(D), "modal class shares", D.modal.value_counts(normalize=True).sort_index().round(3).to_dict())
    print("mean posterior by cohort:\n", D.groupby("cohort")[[f"p{g+1}" for g in range(ck["G"])]].mean().round(3))
    print("covariate means by modal class (first adult interview):\n", D.groupby("modal")[cols].mean().round(2).T)
    print("state at first adult interview by modal class:\n", pd.crosstab(D.modal, D.state, normalize="index").round(3))
    print("posterior certainty: share with max p > 0.7:", round((D[[f"p{g+1}" for g in range(ck["G"])]].max(1) > 0.7).mean(), 3))

def forecasts():

    torch.set_num_threads(4)
    ck = torch.load(f"{cv.OUT}/phase_lat_f-1.pt", weights_only=False)
    sp = cv.make_space(ck["cfg"]); xc, st = ck["xcols"], ck["stats"]
    par = Params(sp, p=len(xc) * 2, n_src=3, init=ck["par"]); par.age_range = ck["age_range"]
    m = LT.Mixture(par, ck["G"], delta=ck["delta"], W=ck["W"])
    A = C.load(); A = A[A.split.isin(["train", "valid"])]
    H = 10.0
    rows, summary = [], {}
    for lm in ("nlsy2002", "nlsy2008"):
        O = cv.origins(A, lm).reset_index(drop=True)
        post = LT.origin_posterior(m, A, O, sp, xc, st)
        prior = LT.origin_posterior(m, A.iloc[:0], O, sp, xc, st)
        el = np.full(len(O), H)
        with torch.no_grad():
            f_post = LT.forecast_mixture(m, O, xc, st, el, post)
            f_prior = LT.forecast_mixture(m, O, xc, st, el, prior)
            f_class = [C.forecast(c, O, xc, st, el)[0].numpy() for c in m.chains()]
        D = O[["pid", "state", "age", "dur_state_yrs", "prior_union", "female"]].copy()
        D["state"] = D.state.replace({"N": "S"})
        for g in range(ck["G"]):
            D[f"post{g+1}"] = post[:, g]
        D["modal"] = post.argmax(1) + 1
        for j, lab in enumerate(["none", "cohab", "married"]):
            D[f"p_{lab}"] = f_post[:, j]; D[f"prior_{lab}"] = f_prior[:, j]
            for g in range(ck["G"]):
                D[f"class{g+1}_{lab}"] = f_class[g][:, j]
        D["abs_change"] = np.abs(f_post - f_prior).sum(1) / 2
        D["landmark"] = lm
        rows.append(D)
        summary[lm] = {}
        for s, g in D.groupby("state"):
            summary[lm][s] = dict(n=int(len(g)), modal_shares=g.modal.value_counts(normalize=True).sort_index().round(3).to_dict(),
                                  mean_abs_change=float(g.abs_change.mean()),
                                  by_modal={int(c): dict(n=int(len(gc)), prior_union=float(gc.prior_union.mean()),
                                                         p_none=float(gc.p_none.mean()), p_cohab=float(gc.p_cohab.mean()), p_married=float(gc.p_married.mean()))
                                            for c, gc in g.groupby("modal")},
                                  class_forecast_spread={lab: float(max(g[f"class{c+1}_{lab}"].mean() for c in range(ck["G"])) - min(g[f"class{c+1}_{lab}"].mean() for c in range(ck["G"])))
                                                         for lab in ("none", "cohab", "married")})
        print(lm, json.dumps(summary[lm], indent=1), flush=True)
    pd.concat(rows).to_csv(f"{C.ROOT}/runs/class_forecasts.csv", index=False)
    json.dump(summary, open(f"{C.ROOT}/runs/class_forecasts.json", "w"), indent=1)

if __name__ == "__main__":
    {"profile": profile, "forecasts": forecasts}[sys.argv[1]]()
