# Dataset

`relationship_five_state_dataset.csv` is the analysis table used by the model. Each row records one pair, a respondent and one other person, at one survey wave; the table has 25,233 rows and 182 columns. It can be rebuilt with `python3 dataset/build_dataset.py` once the raw files described in `raw_data/README.md` are in place.

## Five states

| State | Meaning | Eastwick `R_Status` | HCMST |
|---|---|---|---|
| NR | no relationship: never related, broke up, divorced, separated or partner died | 1 | breakup, divorce, separation or partner died (`nr_type` gives the reason, `partner_deceased` = 1 for death) |
| NP | acquaintance or friend, no romantic potential | 2, 4 | |
| RP | acquaintance or friend with romantic potential, including casual dating | 3, 5, 6 | |
| CR | committed relationship: serious dating or main partner, including unmarried cohabitation | 7 | unmarried and still together |
| M | married | | married |
| missing | answer could not be mapped; the row is kept and `missing_reason` says why | blank | no response, not followed after a breakup, or conflicting answers |

## Columns

- Keys and time: `row_id`, `source`, `person_id`, `pair_id`, `wave`, `obs_date`, `time_years` (years since the pair's first observation), `split` (train/test), `cv_fold` (0 to 4, training people only)
- States: `raw_status`, `raw_status_label`, `state`, one-hot `NR` to `M`, `state_missing`, `missing_reason`, `mapping_note`, `nr_type`, `partner_deceased`, `cohabiting`, `prev_state`, `next_state`, `next_wave`, `interval_years_to_next`
- Traits measured in both surveys where available: `resp_age`, `resp_gender`, `partner_age`, `partner_gender`, `partner_age_minus_resp`, `same_sex`, `known_years`, `romantic_years`, `cohab_years`, `married_years`, `resp_edu_years`, `partner_edu_years`, `hh_income_band` (HCMST, 1 to 19), `romantic_interest_self` (Eastwick, 1 to 7), `partner_interest_perceived` (Eastwick, 1 to 7), `rel_quality_1to5` (HCMST waves 1 and 4)
- `ew_*`: Eastwick participant traits and relationship ratings
- `hc_*`: HCMST background variables (education, income, race, religion, how they met)

Survey skip codes are stored as missing values, and `meta/variable_ledger.csv` describes every source variable.

## Train/test split

People from both surveys were pooled and split by person (seed 20261019). One in five people forms the test set, and the remaining training people are divided into 5 folds for cross-validation. All pairs and waves of a person stay on the same side of the split. The 993 HCMST respondents without a partner at baseline are excluded because they have no pair to follow.
