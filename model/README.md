# Model

The model is a continuous-time Markov chain with seven internal states. Five are the observed states, and the other two distinguish two kinds of no tie. D marks a pair in which the partner died, and it is an absorbing state. NRe marks no tie after a breakup or divorce, after which the HCMST survey stops following the pair. Both internal states are reported as NR in the results and on the simulation page.

Each transition from state i to state j has rate q = exp(α + x·β), and the state probabilities follow P(t) = expm(Q·t). We estimated α and β by maximum likelihood with an L2 penalty on β, using PyTorch and L-BFGS. Fitting and tuning use only the training people in `../dataset/relationship_five_state_dataset.csv`.

A trait enters a transition only when the training data support that effect. This requires at least 50 pairs with the trait measured, a spread of at least 0.3 standard deviations and at least 10 observed transitions. Early-stage transitions and couple or marriage transitions have separate penalties, selected by 5-fold cross-validation (λ = 30 and 10). For relationship quality, income and cohabitation in the HCMST survey, a missing wave takes the most recent earlier answer.

The HCMST survey cannot record friendship after a breakup, so one correction term adjusts the couple-to-friends and couple-to-potential transitions for HCMST rows. Prediction curves for couples and married pairs count a breakup as no tie.

`bash run_all.sh` runs `run_sensitivity.py`, `bench.py` and `final_test.py` in order. The scripts require numpy, pandas, scikit-learn and torch, and they write the following files.

| File | Contents |
|---|---|
| `results_cv.json` | cross-validation and sensitivity analysis |
| `results_bench_cv.json` | benchmark models |
| `results_test.json` | test set results, evaluated once |
| `personal_rate_multipliers.csv` | rate multiplier for a one standard deviation increase in each trait |
| `group_curves.json` | state probabilities over 0 to 10 years for the model without traits |
| `personal_model.pt` | fitted α, β and the survey correction term |
