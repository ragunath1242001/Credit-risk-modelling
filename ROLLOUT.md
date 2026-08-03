# CreditRiskLab Rollout Record

Last updated: 2026-08-04

## Project purpose

CreditRiskLab is an educational credit-risk modelling platform built around the public South German Credit dataset. It demonstrates reproducible data ingestion, probability-of-bad-outcome modelling, model comparison, calibration, explainability, synthetic LGD/EAD/ECL extensions, monitoring, API serving, and Streamlit presentation.

It is not a live lending system, bank-approved model, regulatory capital model, fairness assessment, or IFRS 9-compliant implementation.

## Completed work

### Data and provenance

- Added UCI South German Credit ingestion.
- Added SHA-256 checksum and dataset version metadata.
- Added target mapping: original `kredit=0` is represented as `target=1` bad outcome.
- Added English feature labels for the Streamlit UI.
- Added native validation and an executable Great Expectations-style suite.
- Kept empirical and synthetic datasets separate.

### PD modelling

- Added leakage-safe preprocessing with imputation, scaling, and one-hot encoding.
- Added logistic-regression baseline.
- Added XGBoost challenger with scikit-learn fallback for hosted environments.
- Added bounded Optuna tuning.
- Added stratified train/test evaluation.
- Added ROC AUC, Gini, KS, Brier/probability-accuracy score, confusion matrix, and calibration metrics.
- Added calibrated logistic model artifact.
- Added low/medium/high risk bands.
- Added scorecard bins, threshold sensitivity, and segment evidence.

### Explainability and monitoring

- Added SHAP explanations using a representative background sample.
- Added English SHAP feature labels.
- Added partial-dependence artifact generation.
- Added PSI drift monitoring.
- Added Evidently report generation when the optional dependency is available.
- Added deterministic synthetic longitudinal performance monitoring.

### Synthetic risk extensions

- Added seeded synthetic recovery/LGD data.
- Added seeded synthetic revolving EAD/CCF data.
- Added trained synthetic LGD and EAD regression artifacts.
- Added scenario-based ECL calculation for upside, base, and downside cases.
- Added explicit synthetic flags, seeds, versions, and configuration hashes.

### API and persistence

- Added FastAPI health and readiness endpoints.
- Added single and batch PD prediction endpoints.
- Added selectable model names in API requests.
- Added LGD, EAD, and ECL endpoints.
- Added registry and monitoring endpoints.
- Added SQLite local persistence and optional PostgreSQL support.
- Added prediction, dataset, model, validation review, monitoring, and audit tables.
- Added MLflow run and artifact logging when MLflow is available.

### Streamlit application

- Added sidebar navigation pages:
  - Overview
  - Portfolio explorer
  - PD model lab
  - Single prediction
  - Expected loss
  - Monitoring
  - Model registry
  - Documentation
- Added model dropdowns for:
  - Logistic regression
  - Calibrated logistic regression
  - XGBoost challenger
- Added formatted English labels instead of raw metadata JSON.
- Added readable provenance, validation, SHAP, drift, registry, and performance tables.
- Added Streamlit Cloud runtime and secrets templates.
- Removed Docker because the project is intended to run directly on Streamlit Cloud and locally with Python.

### Testing and documentation

- Added pytest coverage for core modelling, synthetic generators, contracts, challenger training, and longitudinal data.
- Current test status: `6 passed`.
- Added comprehensive README.
- Added architecture, data dictionary, model card, validation, limitations, and demo documentation.
- Added MIT license.
- Added pinned full-environment requirements and lightweight Streamlit Cloud requirements.
- Added a plain-English project guide PDF locally; it was intentionally not committed to Git.

## Current deployment model

### Streamlit Cloud

The intended hosted path is Streamlit Cloud using:

```text
streamlit_app.py
requirements.txt
runtime.txt
```

The hosted app runs in local demo mode. It does not require Docker, PostgreSQL, FastAPI, or MLflow to display the platform.

### Local development

Lightweight Streamlit demo:

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Full modelling environment:

```powershell
python -m pip install -r requirements-full.txt
python scripts/train_pd.py
python scripts/generate_synthetic.py
python scripts/generate_reports.py
uvicorn credit_risk_lab.api:app --reload
```

## Current model results

The reproducible holdout run currently reports approximately:

| Model | ROC AUC | Gini | KS | Brier score |
|---|---:|---:|---:|---:|
| Logistic regression | 0.754 | 0.508 | 0.377 | 0.175 |
| XGBoost challenger | 0.773 | 0.546 | 0.394 | 0.166 |

These values are demonstration results on a small historical dataset and are not production performance evidence.

## Known limitations

- The public dataset has no reliable time dimension, so no empirical temporal backtest is claimed.
- LGD, EAD, recovery, longitudinal, macro, and scenario data are synthetic.
- The hosted demo may use the scikit-learn challenger fallback if XGBoost is unavailable.
- Streamlit Cloud does not provide the full MLflow/PostgreSQL serving stack.
- Model registry records are educational metadata, not formal governance approval.
- The dashboard is a portfolio demonstration, not a production monitoring control.
- Feature attribution is not causal evidence or a legal fairness assessment.

## Next implementation plan

### Priority 1: Stabilise hosted demo

1. Confirm Streamlit Cloud redeploys successfully with Python 3.12.
2. Verify all sidebar pages load without raw metadata or missing optional artifacts.
3. Verify all three model dropdown choices work in hosted mode.
4. Add a small hosted smoke-test checklist for future changes.

### Priority 2: Improve model selection evidence

1. Display side-by-side baseline, calibrated, and XGBoost metrics.
2. Make the selected model's metric table update dynamically.
3. Add a visible explanation of why a model is marked `approved-for-demo`.
4. Persist the selected model name in prediction audit records.

### Priority 3: Improve synthetic risk workflows

1. Add explicit LGD and EAD model dropdowns where formula and trained predictions can be compared.
2. Add amount-weighted LGD errors and CCF error metrics.
3. Add Stage 1, Stage 2, and Stage 3 educational labels to the ECL page.
4. Add scenario assumptions and discounting details to the UI.

### Priority 4: Improve monitoring and governance

1. Add a dedicated performance-monitoring artifact with period-by-period metrics.
2. Populate monitoring-run and audit-event records from the reporting script.
3. Add validation-review notes and decision history to the registry page.
4. Add artifact links for SHAP, calibration, partial dependence, and Evidently reports.

### Priority 5: Portfolio polish

1. Add screenshots to the README.
2. Add a short demo video or GIF if useful.
3. Add a release tag and changelog.
4. Add a Streamlit Cloud URL to the README after deployment is stable.

## Resume instructions for the next session

Start by reading this file, then run:

```powershell
git status
python -m pytest -q
streamlit run streamlit_app.py
```

If the hosted app fails, inspect Streamlit Cloud logs first and compare the failure with `requirements.txt`, `runtime.txt`, and the fallback paths in `src/credit_risk_lab/core.py`.
