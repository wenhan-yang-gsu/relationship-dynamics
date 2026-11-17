# Relationship dynamics

Wenhan Yang, Department of Mathematics and Statistics, Georgia State University. Advisor: Yi Jiang.

Code for a continuous-time Markov chain that forecasts a person's partnership state (single, dating, cohabiting, married) at any horizon from one fit to adjacent survey interviews. The chain carries duration phases and latent classes, is trained on NLSY97 and the How Couples Meet and Stay Together surveys of 2009 and 2017, is selected by household cross-validation against tuned direct learners under rules fixed before the results were seen, and is scored once on NLSY79.

## Files

`data_prep/nlsy97.py`, `data_prep/hcmst.py` and `data_prep/panel.py` build the training panel `cache/panel.csv` from the raw survey files; `data_prep/nlsy79.py` builds the confirmation panel `cache/nlsy79_panel.csv`. Each script runs its steps in order, or only the steps named on the command line.

`model/chain.py` defines the state space, the transition intensities and the sparse uniformization of the matrix exponential. `model/sequences.py` holds the hidden-Markov forward recursion over interview sequences with set-valued observations, the penalty and the L-BFGS fit. `model/common.py` holds covariate groups, data loading, forecast targets, forecasting and inverse-probability-weighted scoring, real income in 2020 dollars and the stage-unknown baseline. `model/structures.py` holds the latent classes, the restricted cubic splines, the origin filter for histories without a dating module and the monthly NLSY97 observations. `model/cv.py` runs the household cross-validation (`fit`), the comparison with the inclusion rule (`compare`) and the skill baseline (`null`). `model/competitors.py` trains and tunes the direct and iterated learners in the same folds (`fixed`, `tuned`, `annual`), including the truncation experiment. `model/simulation.py` simulates the four generating designs on the real interview skeleton and refits the four structures. `model/nlsy79_eval.py` refits with the NLSY79 information set, scores once at the pre-specified landmarks and horizons and applies the decision table. `model/classes.py` reports class membership and covariate means by class (`profile`) and ten-year forecasts by origin state and class (`forecasts`). `model/information.py` decomposes the forecast skill into the contributions of four information sources withheld at forecast time, out of fold (`0` to `4`, then `summary`). `model/supplement.py` holds the descriptive supplements added after review: the simpler structures scored on NLSY79 (`nlsy79`), calibration intercepts and a state-aware baseline for the cross-validation cells (`cv`), calibration tables for NLSY79 (`calib79`) and realized ten-year outcomes by latent class (`classes`). `figures.py` builds the source tables and the six figures.

## Running

```bash
pip install -r requirements.txt
python3 data_prep/nlsy97.py && python3 data_prep/hcmst.py && python3 data_prep/panel.py && python3 data_prep/nlsy79.py
cd model
for c in none base lat phase phase_lat smooth noise lat_s1 lat_s2; do for k in 0 1 2 3 4; do python3 cv.py fit $c $k; done; done
for c in trunc_base trunc_none; do for k in 0 1 2 3 4; do python3 cv.py fit $c $k; done; done
for k in 0 1 2 3 4; do python3 competitors.py tuned $k; python3 competitors.py fixed $k; python3 competitors.py annual $k; done
python3 cv.py null
python3 cv.py compare base lat phase phase_lat smooth noise latbest
python3 cv.py compare lat phase_lat
python3 cv.py compare phase_lat gbmT logitT phd itm_gbm
python3 cv.py compare trunc_base trunc_none trunc_itm_gbm trunc_itm_logit trunc_phd trunc_gbm_h5 trunc_logit_h5 trunc_dtime
for c in base lat phase phase_lat base79 lat79 phase79 phase_lat79 base79cal lat79cal phase79cal phase_lat79cal; do python3 cv.py fit $c -1; done
for d in neither duration heterogeneity both; do for r in $(seq 0 9); do python3 simulation.py $d $r; done; done
python3 nlsy79_eval.py phase_lat 79
python3 nlsy79_eval.py phase_lat full
python3 nlsy79_eval.py phase_lat 79cal
AGE_LINEAR=1 python3 nlsy79_eval.py phase_lat 79
NLSY79_WEIGHTED=1 python3 nlsy79_eval.py phase_lat 79
python3 classes.py profile && python3 classes.py forecasts
for k in 0 1 2 3 4; do python3 information.py $k; done && python3 information.py summary
python3 supplement.py classes && python3 supplement.py cv && python3 supplement.py nlsy79 && python3 supplement.py calib79 && python3 supplement.py agreement
cd .. && python3 figures.py
```

Paths are relative to the repository root, or to the directory named by the environment variable `LOVE_ROOT`; panels are read from `cache/`, results are written to `runs/` and figures to `figures/`. One cross-validation fold of the final structure needs about 200 GB of memory and several hours on one CPU node.

## Data

The raw files are not redistributed. `raw_data/NLSY97/raw/nlsy97_all_1997-2023.csv` with its `.cdb` and `.sdf` files, and `raw_data/NLSY79/raw/nlsy79_all_1979-2022.csv` with its codebook, are full-variable exports from the NLS Investigator (https://www.nlsinfo.org/investigator). `raw_data/HCMST/raw/` holds `HCMST_ver_3.04.dta`, `wave_4_supplement_v1_2.dta`, `HCMST_wave_5_supplement_ver_1.dta` and `HCMST_wave_6_public_v1.dta` from https://data.stanford.edu/hcmst; `raw_data/HCMST2017/raw/hcmst2017_2022_v2.2.dta` comes from https://data.stanford.edu/hcmst2017.

Bureau of Labor Statistics, U.S. Department of Labor. National Longitudinal Survey of Youth 1997 cohort, 1997–2023, and National Longitudinal Survey of Youth 1979 cohort, 1979–2022. Produced and distributed by the Center for Human Resource Research, The Ohio State University.

Rosenfeld, M. J., Thomas, R. J., & Falcon, M. (2018). How Couples Meet and Stay Together, Waves 1–3 (version 3.04) and Wave 4–6 supplements. Stanford University Libraries.

Rosenfeld, M. J., Thomas, R. J., & Hausen, S. (2019). How Couples Meet and Stay Together 2017 fresh sample, with 2020 and 2022 follow-up waves. Stanford University Libraries.

The code is released under the MIT License; the data remain under the terms of their sources.
