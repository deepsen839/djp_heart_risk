# Dataset provenance and handling

- **Source:** UCI Machine Learning Repository, Heart Disease dataset (dataset ID 45): https://archive.ics.uci.edu/dataset/45/heart+disease
- **Subset:** Cleveland subset (303 rows in the canonical source); the downloader retains the first 303 records returned by the repository package. Verify the downloaded row count and schema in `reports/data_audit.json`.
- **Retrieval:** fetched at runtime by `ucimlrepo`; the exact UTC retrieval date is recorded in the audit report. No dataset file is committed.
- **Terms/licence:** CC BY 4.0; attribution is required. Cite Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989), *Heart Disease*, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C52P4X. This repository does not redistribute the source data. Do not assume that the dataset is DJP/client data.
- **Target:** source `num` severity is transformed to binary: `0 -> 0`; `1..4 -> 1`. This is a modelling convention, not a clinical definition.
- **Patient ID:** no stable patient identifier is supplied; no patient-level split can be guaranteed and no identifier is fabricated.
- **Missingness:** source missing values are preserved as null and imputed inside the training pipeline only.
- **Limitations:** small, historical cohort; uncertain generalisability; possible measurement and selection bias; not representative of a clinic, India, or any intended deployment cohort.

Run `djp-audit` to create the machine-readable audit and data dictionary. Check all recorded counts and current source terms before submission.
