# Raw data

The original survey files are not part of this repository. After downloading them from the sources below into the listed folders, `python3 dataset/build_dataset.py` rebuilds the analysis table.

The How Couples Meet and Stay Together survey is available from Stanford Libraries after free registration (https://data.stanford.edu/hcmst). The four files `HCMST_ver_3.04.dta`, `wave_4_supplement_v1_2.dta`, `HCMST_wave_5_supplement_ver_1.dta` and `HCMST_wave_6_public_v1.dta` belong in `raw_data/HCMST/raw/`. The file `raw_data/HCMST/CHECKSUMS.sha256` identifies the exact versions used in this project. Citation: Rosenfeld, M. J., Thomas, R. J., & Falcon, M. (2018). How Couples Meet and Stay Together, Waves 1–3 (public version 3.04), plus Wave 4–6 supplements. Stanford University Libraries.

The college dating study data are available on the OSF page linked from the original article: Eastwick, P. W., Joel, S., Carswell, K. L., Molden, D. C., Finkel, E. J., & Blozis, S. A. (2023). Predicting romantic interest during early relationship development: A preregistered investigation using machine learning. European Journal of Personality, 37(3), 276–312. https://doi.org/10.1177/08902070221085877. The files `full initiation data for osf.csv`, `SamPredictors.sas7bdat`, `Rows208.sas7bdat`, `IDswithRelStatus.xlsx` and `Stage2Predictors.sas` belong in `raw_data/Eastwick/data/`.
