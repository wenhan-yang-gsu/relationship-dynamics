# Relationship Dynamics

Wenhan Yang and Zhiyu Liu, Math 4991, Georgia State University. Advisor: Professor Yi Jiang.

Interactive simulation: https://wenhan-yang-gsu.github.io/relationship-dynamics/

This project models how the relationship between two people changes over time. Each pair occupies one of five states: no tie (NR), friends without romantic potential (NP), romantic potential (RP), couple (CR) or married (M). Because pairs can move between these states in both directions, we treat the process as a continuous-time Markov chain. The transition from state i to state j occurs at rate q_ij(x) = exp(α_ij + Σ_k β_ijk x_k), where x contains the traits of the pair. The state probabilities therefore satisfy dP/dt = P·Q, whose solution is P(t) = P(0)·e^{Qt}. We estimated α and β by penalized maximum likelihood from 13,680 observed transitions.

The data come from two longitudinal surveys that cover different stages of a relationship. A college dating study of 208 freshmen covers the early states, from no tie to couple. The How Couples Meet and Stay Together (HCMST) survey followed 3,009 couples from 2009 to 2017 and records marriage and breakup. Training and test sets were split by person, so that no individual contributes to both sets.

On 571 held-out people with 2,685 transitions, the dynamic model with traits achieved the lowest test log loss. Its margin over multinomial logistic regression lies within the bootstrap uncertainty, whereas its margins over gradient boosting and over the dynamic model without traits do not.

| Model | Test log loss |
|---|---|
| Dynamic model with traits | 0.405 |
| Multinomial logistic regression | 0.409 |
| Gradient boosting | 0.417 |
| Dynamic model without traits | 0.456 |
| Frequency baseline | 0.466 |

## Repository contents

`dataset/` contains the code that builds the analysis table from the raw surveys, the table itself (`relationship_five_state_dataset.csv`, 25,233 rows) and a description of each variable. `model/` holds the model, the cross-validation and sensitivity analyses, the benchmark models, the final test and their results. `figures/` contains the plotting code and the result figures, and `index.html` is the simulation page. The raw survey files are not redistributed, and `raw_data/README.md` explains where to obtain them.

## Reproducing the results

```bash
pip install -r requirements.txt
python3 dataset/build_dataset.py
cd model && bash run_all.sh
cd ../figures && python3 figs.py
```

## Data sources

Rosenfeld, M. J., Thomas, R. J., & Falcon, M. (2018). How Couples Meet and Stay Together, Waves 1–3 (public version 3.04), plus Wave 4–6 supplements. Stanford University Libraries. https://data.stanford.edu/hcmst

Eastwick, P. W., Joel, S., Carswell, K. L., Molden, D. C., Finkel, E. J., & Blozis, S. A. (2023). Predicting romantic interest during early relationship development: A preregistered investigation using machine learning. European Journal of Personality, 37(3), 276–312. https://doi.org/10.1177/08902070221085877

Please cite both original sources whenever the data or the derived table are used. The code is released under the MIT License, while the data remain under the terms of their original sources.
