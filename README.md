# Accident Risk Prediction for Traffic Intersections

A machine-learning pipeline and Streamlit app that **scores and ranks intersection-hours by accident risk** from the conditions in that hour, so traffic teams can prioritise patrols and signal checks. It is a risk-ranking tool, not (yet) a proven advance-warning system; see [Limitations](#limitations).

> **Status: proof of concept.** On this small dataset the model ranks risk about **2–3x better than random**, but most alerts are still false alarms (see [Results](#results) and [Limitations](#limitations)). The goal of this repo is a correct, leakage-free pipeline with honest evaluation, not an inflated score.

**Demo:** run locally with `streamlit run Accident_app.py` (see [Run it](#run-it))  |  **Author:** [Arshia Kauser](https://www.linkedin.com/in/arshia-kauser-50086732/)

## App preview

![Risk predictions table](images/app_predictions.png)<img width="1366" height="693" alt="Accident_app_p" src="https://github.com/user-attachments/assets/95db7274-a515-4414-ba08-ebf745c86729" />

![Summary metrics and risk map](images/app_map.png)<img width="1366" height="689" alt="Accident_app_p1" src="https://github.com/user-attachments/assets/1f385cf4-865b-47c5-ac29-74a8f4afda1e" />


## Business problem

Accidents at intersections and on highways are driven by unpredictable traffic, poorly timed signals and road conditions, and there is no system that warns about accident-prone areas in advance.

| | |
|---|---|
| Objective | Warn about accident-prone times and places in advance (this repo scores risk per hour; advance warning is not yet demonstrated) |
| Constraint | Keep false alarms low |
| ML success criterion | Original target "85% accuracy" is misleading here: a model that never predicts an accident already scores about 95% (only 4.6% of rows are accidents). This project uses **precision among the top-k riskiest rows** and **PR-AUC** instead. |
| Business success criterion | 20% fewer accidents at flagged zones. This cannot be shown from historical data and needs a field pilot (flagged vs unflagged locations). |

## Data

- 2,016 hourly records for **12 locations over 7 days** (1-7 July 2025): traffic volume and speed, signal timings, weather, lighting, enforcement level, location attributes.
- **93 accident hours (4.6%).** `severity`, `cause`, `vehicles_involved` and `veh_count_at_accident` exist only after an accident happens, so they are excluded as inputs (target leakage).
- The dataset is not included here unless you add it to the repo root as `accidents_dataset.csv`. `sample_data/new_data.csv` has 91 rows without target columns for trying the app.

## Approach

1. **EDA** (`EDA_Num_Cat_Uni_Bi_Analysis.py`): class balance, effect sizes (AUC, Cohen's d, Cramer's V) instead of p-values alone, redundancy and leakage checks.
2. **Feature engineering** (`preprocessing8.py`): signal ratios, congestion density, overspeeding index, weather x lighting interactions, KMeans traffic profiles. Missing values are kept as information (indicator columns / "missing" category). Feature selection by SHAP / random-forest importance.
3. **Honest validation** (`Model_Train_Evaluate_Predict.py`):
   - **Time-based split:** train on the earliest 80% of hours, test on the latest 20%.
   - **Class imbalance handled inside cross-validation** (SMOTE only ever sees training folds); one balancing strategy per model.
   - **Alert threshold from out-of-fold predictions** on the training data, never from the test set. Mode used: flag the **top 5% riskiest rows** (`alert_budget`), which matches the "minimize false alarms" constraint.
   - **Best model chosen by cross-validated precision** among the top 5% of rows; the test set is used once, with bootstrap 95% intervals because it contains only 15 accidents.
4. **App** (`Accident_app.py`): upload a CSV, get a risk level (Low / Medium / High / Critical), a recommended action and a risk map. High / Critical means "above the alert threshold".

## Results

Time-based test set: 404 hourly rows, **15 accidents (3.7% base rate)**. Cross-validation on the training set (1,612 rows, 4.8% base rate).

| Model | CV precision, top 10% | Test precision, top 5% | Test accidents caught, top 5% | Test lift, top 10% |
|---|---|---|---|---|
| CatBoost (selected by CV) | 9.9% | 15% | 20% | 2.0x |
| BalancedRandomForest | 8.7% | 15% | 20% | 2.7x |
| RandomForest | 9.9% | 20% | 27% | 3.4x |
| GradientBoost | 9.3% | 10% | 13% | 2.7x |
| LightGBM | 8.1% | 10% | 13% | 2.0x |
| _Random flagging_ | _4.8%_ | _3.7%_ | _5%_ | _1.0x_ |

![Model performance comparison](images/model_performance_comparison.png)
<img width="3600" height="1800" alt="model_performance_comparison" src="https://github.com/user-attachments/assets/02cdb561-e541-4dbe-852d-894449ff3dea" />



**How to read this:** flagging the riskiest 5-10% of hours finds accidents at roughly 2-3x the rate of random flagging. The test set has only 15 accidents, so the models are statistically very close; one extra hit changes a ranking. Cross-validated numbers are more reliable than the test numbers.

## Limitations

- **Not proven as advance warning:** the model scores risk from same-hour measurements (traffic volume, speed), which may partly reflect an accident that is already happening. In a quick check with earlier-hour data only, the signal fell to about random level on this 7-day dataset. Advance warning needs much more history.
- **Small data:** 93 accidents, 12 locations, 7 days. This limits any model; results should not be generalised to other cities or seasons.
- **Most alerts are false:** even the best operating point leaves roughly 8-9 false alerts per real accident.
- **Cannot claim performance at new locations:** only 12 sites; the model learns risk patterns at those sites.
- **Probabilities are not calibrated** (training uses resampling); use scores for ranking and thresholds, not as literal probabilities.
- No evidence yet of accident reduction: that requires a pilot.

## Run it

**Try the app (needs Python 3.10):**

```bash
git clone https://github.com/<arshiakauser>/accident-risk-prediction.git
cd accident-risk-prediction
pip install -r requirements.txt
streamlit run Accident_app.py
```

The app opens at http://localhost:8501. Upload `sample_data/new_data.csv` to see the predictions, risk levels, recommended actions and the risk map. The trained model files (`best_model.pkl`, `preprocessing_artifacts.pkl`) are included in the repo. They were saved with the library versions pinned in `requirements.txt`, so keep those versions.

**Re-run the analysis and training (optional):**

```bash
pip install -r requirements-train.txt

# put accidents_dataset.csv in this folder, then:
python EDA_Num_Cat_Uni_Bi_Analysis.py                 # plots + tables -> eda_outputs/
python Model_Train_Evaluate_Predict.py                # trains, evaluates, saves best_model.pkl
python Model_Train_Evaluate_Predict.py predict --new-data sample_data/new_data.csv
```

Options: `--threshold-mode f1|precision|alert_budget`, `--alert-budget 0.05`, `--selection-metric CV_P@5%` (see the config block at the top of the training script). The PostgreSQL upload in the app is hidden by default; for private use set the environment variable `ENABLE_DB_UPLOAD=1`.

## Repository layout

```
Accident_app.py                      Streamlit app
Model_Train_Evaluate_Predict.py      train / evaluate / predict (CLI)
preprocessing8.py                     feature engineering, preprocessing, shared helpers
EDA_Num_Cat_Uni_Bi_Analysis.py        exploratory analysis
best_model.pkl, preprocessing_artifacts.pkl   trained model files used by the app
requirements.txt                     pinned app dependencies (deployment)
requirements-train.txt               full dependencies (training, unpinned)
sample_data/new_data.csv             91 rows to try the app
```

## Next steps

More weeks, months and locations of data (the main lever); a "next 1-3 hours" target built only from earlier data; history features (accidents at a location in previous days, recent speed and traffic trends); predicting risk per location per day instead of per hour; a field pilot to measure real accident reduction.
