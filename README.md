# The Homecoming Problem — Predicting Which Long-Term Care Hospitals Get Patients Back to the Community

A CRISP-DM data science project on the CMS **Long-Term Care Hospital (LTCH) Provider Data** file (June 2026 refresh: 24,258 records = 311 LTCHs × 78 measure codes from the LTCH Quality Reporting Program). It trains an explainable machine learning model to flag **below-median "Discharge to Community" facilities using their clinical safety and process measures** — infections, ventilator weaning, mobility gains, pressure ulcers, readmissions, spending, and care-transition processes.

## Motivation

LTCHs treat the sickest post-acute patients (prolonged ventilation, severe wounds, multi-organ failure; 25+ day stays). For families, one outcome dominates: does the patient ever get home? CMS reports this as the risk-standardized Discharge-to-Community (DTC) rate — but it is suppressed for facilities with too few cases, and it is only one line among dozens of statistics. This project asks whether the *other* measures carry the homecoming signal, which matters for families choosing facilities, and for CMS in understanding which upstream measures travel with poor community-discharge performance. With only 311 LTCHs in the country, it is also an honest exercise in small-data ML.

Questions of interest:

1. What are the most important features, what do they mean, and how do they drive the predicted outcome?
2. What unusual or creative insights can be gathered from the data?
3. How accurate is the trained model — and how should accuracy be evaluated trustworthily at n = 291?
4. What happens in a creative predictive scenario using the trained model?

## Libraries Used

- Python 3.12
- pandas, NumPy — cleaning, long→wide facility-level reshaping
- scikit-learn — RandomForestClassifier (class-weighted), StratifiedKFold cross-validation, ROC/PR metrics
- SHAP — model explainability (TreeExplainer; summary, dependence, and local plots)
- Matplotlib, Seaborn — visualization
- Jupyter / nbconvert — notebook execution

## Files in this Repository

| File | Description |
|---|---|
| `code/The_Homecoming_Problem.ipynb` | Main Jupyter notebook: full CRISP-DM workflow — EDA, measure-code mapping, cleaning rationale, cross-validated modeling, SHAP explainability, and the predictive scenario. |
| `dataset/Long-Term_Care_Hospital-Provider_Data_Jun2026.csv` | Source data from Medicare Care Compare (data.cms.gov). |
| `README.md` | This file. |

## Summary of Results

- **Data cleaning was substantial:** the long file (facility × measure-component rows, scores as strings with "Not Available") was reshaped to a 311 × 78 wide matrix; 17 primary measures were selected and mapped to human-readable names cross-checked against CMS LTCH QRP documentation; missing features median-imputed; labels restricted to the 291 facilities with a genuinely reported DTC rate; a temporal caveat (claims-based target 2022–2024 vs assessment features 2024–2025) is stated explicitly.
- **Most important features (via SHAP):** ventilator liberation rate and mobility gain dominate — recovery-oriented care, not just harm avoidance, is what gets patients home; preventable readmissions and Medicare spending push toward low homecoming; paperwork-style process measures contribute little.
- **Creative insights:** the median LTCH successfully discharges only ~16–17% of patients to the community; recovery measures track the homecoming outcome while reporting-compliance measures barely correlate with it; the LTCH map is lopsided (Texas 39 of 311 facilities; several states have none); ~25% of LTCHs run CAUTI/CLABSI infection ratios above the national baseline while C. difficile control is broadly strong.
- **Model performance:** 5-fold cross-validated ROC AUC ≈ 0.80 ± 0.06 (the primary metric given n = 291), holdout AUC 0.82 consistent with it; precision/recall ≈ 0.75 on balanced classes — honest performance for an outcome also shaped by unmeasured social factors.
- **Scenario ("Two LTCHs, one ventilator, one choice"):** for two facilities whose DTC rate is suppressed, the model reads visible measures into homecoming risk — recovery-strong Facility A: P(below-median) = 0.25; struggling Facility B: 0.69; an all-median control lands mid-range (0.57) — with SHAP itemizing the reasons for each.

## Data Dictionary (per CMS)

| Column | Type | Description |
|---|---|---|
| `CMS Certification Number (CCN)` | Character | Facility identifier |
| `Provider Name`, address fields, `County/Parish`, `Telephone Number` | Character | Facility name and location |
| `State` | Character | Two-character state code |
| `CMS Region` | Numeric | CMS regional office (1 = Boston … 10 = Seattle) |
| `Measure Code` | Character | CMS ID prefix + variable suffix (e.g., `L_021_01_ADJ_RATE`); key measures: L_006/007/014 = CAUTI/CLABSI/CDI infection SIRs, L_011 = mobility change (ventilated), L_012 = falls with major injury, L_015 = staff flu vaccination, L_017 = potentially preventable readmissions, **L_018 = Discharge to Community (target)**, L_019 = Medicare spending per beneficiary, L_021 = pressure ulcer/injury, L_022 = breathing-trial compliance, L_023 = ventilator liberation, L_025–L_028 = health-information transfer and COVID-19 vaccination measures, L_027 = discharge function score |
| `Score` | Character | Measure score (numeric or "Not Available") |
| `Footnote` | Numeric | 1 = too few cases; 2 = data not available; 3 = shorter period; 4 = suppressed by CMS; 5 = not submitted; 6 = CI lower limit undefined (zero infections); 7 = cannot be calculated; 8 = Medicare waiver program |
| `Start/End Date`, `Measure Date Range` | Date | Reporting window (split windows separated by ";") |

## Acknowledgments

- Data: [CMS Long-Term Care Hospital Provider Data](https://data.cms.gov/provider-data/), LTCH Quality Reporting Program, Medicare Care Compare (June 2026 refresh). Measure identities cross-checked against CMS LTCH QRP public-reporting documentation.
- Project structure follows the CRISP-DM methodology, completed as part of the Udacity Data Scientist Nanodegree "Write a Data Science Blog Post" project.
