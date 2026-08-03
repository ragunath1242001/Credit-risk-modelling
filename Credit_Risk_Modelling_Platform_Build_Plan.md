# CreditRiskLab — Streamlit Credit Risk Modelling Platform Blueprint

## 1. Vision

Build one production-style credit-risk analytics platform, not a collection of notebooks. The platform will ingest the public **South German Credit** dataset, validate and engineer features, develop and calibrate a PD model, register and serve it, and expose the evidence through a Streamlit application.

LGD, EAD, expected loss, longitudinal monitoring and scenario analysis will be explicit synthetic extensions. This is necessary because South German Credit is a small historical classification dataset and does not contain recovery cash flows, utilisation histories, repeated observations or reliable time-series outcomes.

```text
South German Credit → validation → features → PD training → validation
                                      ↓
                         MLflow registry → FastAPI inference
                                      ↓
                   Streamlit app → monitoring and model evidence

Synthetic recovery/EAD/longitudinal data → LGD + EAD + ECL demonstrations
```

The finished project should demonstrate modelling discipline, software engineering, MLOps and model-risk awareness without claiming to be a live lending system, a regulatory model or an IFRS 9-compliant implementation.

## 2. Portfolio positioning

Project 1, the European Banking Risk & Governance Lab, demonstrates banking context, IFRS 9, Basel, BCBS 239, governance and reporting. CreditRiskLab demonstrates that the candidate can build, validate, deploy and monitor credit-risk models.

Target roles:

- Junior Credit Risk Analyst or Modeller
- Risk Analytics Analyst
- Model Validation Analyst
- Credit Risk Data Scientist
- Financial Machine Learning Engineer
- Banking Data Scientist
- Credit Risk or IFRS 9 Analytics Consultant

Recommended portfolio headline:

> Built a reproducible credit-risk modelling platform using South German Credit for PD development, synthetic extensions for LGD/EAD/ECL, MLflow, FastAPI, Streamlit, PostgreSQL, Docker, SHAP, Evidently, pytest and GitHub Actions.

## 3. Scope

### In scope

- public South German Credit ingestion and provenance;
- schema and data-quality validation with Great Expectations;
- leakage-safe feature engineering;
- logistic-regression PD baseline;
- XGBoost or LightGBM PD challenger;
- probability calibration and scorecard-style evidence;
- synthetic LGD, EAD, longitudinal and scenario generators;
- PD, LGD, EAD and IFRS 9-inspired expected-loss workflow;
- ROC AUC, Gini, KS, Brier, calibration, confusion matrix and PSI;
- SHAP explanations and partial-dependence plots where appropriate;
- MLflow experiment tracking and model registry;
- FastAPI inference service;
- PostgreSQL metadata and prediction storage;
- Streamlit portfolio application;
- Evidently drift and quality reports;
- Docker Compose local deployment;
- pytest tests and GitHub Actions CI.

### Non-goals

- real customer or confidential data;
- live credit approval, pricing or underwriting;
- regulatory capital use or model approval;
- claiming compliance with IFRS 9, Basel, EBA, ECB, PRA or other frameworks;
- pretending that the public dataset supports reliable LGD, EAD or temporal backtesting;
- building cloud infrastructure before the local vertical slice works;
- a large dashboard with no reproducible modelling pipeline.

## 4. Data blueprint

### 4.1 Primary empirical dataset

Use **South German Credit** from the UCI Machine Learning Repository as the primary PD dataset.

Source: [UCI South German Credit](https://archive-beta.ics.uci.edu/dataset/522/south%2Bgerman%2Bcredit)

The UCI record describes 1,000 German loan applicants with 20 categorical/integer predictors and a binary good/bad credit-risk target. It is appropriate for a reproducible European PD demonstration because it is public, compact, documented and easy for a reviewer to run locally.

Record in the repository:

- source URL and citation;
- licence;
- download date;
- file checksum;
- dataset version;
- fields used and fields excluded;
- target definition;
- transformation history.

Use South German Credit for PD, feature engineering, scorecard work, calibration, SHAP, discrimination metrics and reference-population monitoring.

### 4.2 Limitations

Do not present it as a complete European bank portfolio. It is historical, small, single-outcome and not a longitudinal loan-performance dataset. It lacks reliable observation dates, recovery cash flows, collateral outcomes, contractual limits, drawdowns, macroeconomic histories and representative population coverage.

Therefore, do not claim a true temporal backtest on the public records. Use a stratified holdout for model development and describe any later-period or drift exercise as synthetic.

### 4.3 Synthetic extensions

Create separate deterministic generators. Never silently merge synthetic records with the empirical dataset.

```text
south_german_credit_v1      empirical PD development data
synthetic_recovery_v1       LGD training data
synthetic_revolving_ead_v1  EAD/CCF training data
synthetic_longitudinal_v1   monitoring, delayed outcomes and scenarios
```

Every synthetic dataset must contain `synthetic = true`, generator version, seed and configuration hash. Use plausible but explicitly illustrative relationships; document that they are not observed bank behaviour.

## 5. End-to-end architecture

```text
                 ┌───────────────────────────────┐
                 │ UCI South German Credit       │
                 └──────────────┬────────────────┘
                                ↓
                 ┌───────────────────────────────┐
                 │ Ingestion + provenance         │
                 └──────────────┬────────────────┘
                                ↓
                 ┌───────────────────────────────┐
                 │ Great Expectations validation  │
                 └──────────────┬────────────────┘
                                ↓
                 ┌───────────────────────────────┐
                 │ Feature pipeline + PD models   │
                 └──────────────┬────────────────┘
                                ↓
                 ┌───────────────────────────────┐
                 │ Metrics + calibration + SHAP  │
                 └──────────────┬────────────────┘
                                ↓
                 ┌───────────────────────────────┐
                 │ MLflow registry                │
                 └──────────────┬────────────────┘
                                ↓
                 ┌───────────────────────────────┐
                 │ FastAPI model-serving API       │
                 └──────────────┬────────────────┘
                                ↓
                 ┌───────────────────────────────┐
                 │ PostgreSQL metadata/predictions│
                 └──────────────┬────────────────┘
                                ↓
                 ┌───────────────────────────────┐
                 │ Streamlit hosted application   │
                 │ metrics · prediction · ECL     │
                 │ drift · registry · limitations │
                 └───────────────────────────────┘
```

Keep the boundaries clear:

- Streamlit is the user-facing application and hosting target.
- FastAPI remains the serving boundary for the full local stack.
- Streamlit Cloud demo mode may load a registered local model directly when external services are unavailable, but must label that mode clearly.
- PostgreSQL and MLflow are infrastructure services, not Streamlit substitutes.

## 6. Build order

1. Create the Python package, configuration, logging and health check.
2. Add South German Credit ingestion, checksum and provenance.
3. Add schema, type, category, range and missingness expectations.
4. Build the PD preprocessing pipeline and deterministic stratified split.
5. Train and evaluate the logistic baseline; log it to MLflow.
6. Train the XGBoost or LightGBM challenger with bounded Optuna tuning.
7. Calibrate probabilities and generate validation, scorecard and SHAP artefacts.
8. Add synthetic recovery, EAD and longitudinal generators.
9. Train bounded LGD/EAD models and calculate expected loss.
10. Register the demo champion and expose versioned FastAPI endpoints.
11. Build the Streamlit app against the API, with a local demo fallback.
12. Add PostgreSQL persistence, Evidently monitoring, Docker Compose, tests and CI.

Do not start with the dashboard. The first milestone is a clean command that ingests, validates and trains PD on South German Credit and records the run in MLflow.

## 7. Repository structure

```text
credit-risk-lab/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock                         # or requirements.txt; choose one
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── streamlit_app.py                # Streamlit entry point
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── configs/
│   ├── data.yaml
│   ├── pd.yaml
│   ├── lgd.yaml
│   ├── ead.yaml
│   └── scenarios.yaml
├── data/
│   ├── raw/south_german_credit/.gitkeep
│   ├── raw/synthetic/.gitkeep
│   ├── processed/pd_training/.gitkeep
│   ├── processed/synthetic_recovery/.gitkeep
│   ├── processed/synthetic_revolving_ead/.gitkeep
│   ├── processed/synthetic_longitudinal/.gitkeep
│   └── README.md
├── src/credit_risk_lab/
│   ├── config.py
│   ├── logging.py
│   ├── data/ingest.py
│   ├── data/validate.py
│   ├── data/synthetic.py
│   ├── features/build.py
│   ├── models/pd.py
│   ├── models/lgd.py
│   ├── models/ead.py
│   ├── models/metrics.py
│   ├── models/calibration.py
│   ├── models/explainability.py
│   ├── loss/expected_loss.py
│   ├── registry/mlflow_client.py
│   ├── monitoring/drift.py
│   ├── monitoring/performance.py
│   ├── db/models.py
│   ├── db/session.py
│   ├── api/main.py
│   └── ui/api_client.py
├── scripts/
│   ├── ingest_data.py
│   ├── train_pd.py
│   ├── generate_synthetic_recovery.py
│   ├── generate_synthetic_ead.py
│   ├── generate_synthetic_longitudinal.py
│   ├── train_lgd.py
│   ├── train_ead.py
│   ├── calculate_ecl.py
│   └── generate_monitoring_report.py
├── tests/
│   ├── test_data_validation.py
│   ├── test_features.py
│   ├── test_expected_loss.py
│   ├── test_synthetic_generators.py
│   ├── test_api.py
│   └── test_model_metrics.py
└── docs/
    ├── architecture.md
    ├── data_dictionary.md
    ├── model_development.md
    ├── validation_report.md
    ├── model_card.md
    ├── limitations.md
    └── demo_script.md
```

## 8. Technology stack

| Area | Choice | Use |
|---|---|---|
| Language | Python | Pipeline, models and services |
| Data | Pandas, NumPy | Cleaning, transformation and calculations |
| Modelling | scikit-learn | Baselines, preprocessing, metrics and calibration |
| Challenger | XGBoost or LightGBM | Non-linear PD model |
| Tuning | Optuna | Bounded reproducible search |
| Tracking | MLflow | Runs, metrics, artefacts and registry |
| API | FastAPI + Pydantic | Typed model-serving boundary |
| UI/hosting | Streamlit | Hosted portfolio application |
| Database | PostgreSQL + SQLAlchemy/Alembic | Metadata, predictions and audit events |
| Quality | Great Expectations | Dataset contracts |
| Monitoring | Evidently + PSI | Drift and quality reports |
| Explainability | SHAP | Global/local attribution |
| Runtime | Docker Compose | Reproducible local stack |
| Testing | pytest | Unit, integration and API tests |
| CI | GitHub Actions | Test, build and smoke checks |

## 9. PD modelling

Define the public target from the UCI documentation and record the mapping from good/bad to the model target. Do not call it a 12-month PD unless a documented performance horizon exists; call it a binary credit-risk outcome probability.

Use:

- baseline: logistic regression with imputation and one-hot encoding;
- challenger: XGBoost or LightGBM;
- calibration: Platt scaling or isotonic regression on a calibration split;
- optional scorecard evidence: bins, coefficients, odds direction and risk bands.

Fit every transformer on training data only. Exclude post-outcome information. Persist the feature manifest, target definition, random seed, split method and model version.

Outputs: raw probability, calibrated probability, risk band, model version, dataset version, feature-set version and request ID.

## 10. LGD modelling — synthetic

Generate defaulted synthetic records containing EAD, recoveries, recovery costs, collateral indicator, product type, seniority and time to recovery.

```text
LGD = clip((EAD - recoveries + recovery_costs) / EAD, 0, 1)
```

Use a mean/segment baseline and a regularised or boosting regressor challenger. Evaluate on defaulted synthetic observations, report row-weighted and amount-weighted errors, and label every model and report `synthetic = true`.

## 11. EAD modelling — synthetic

Generate synthetic revolving exposures with credit limit, drawn balance and balance at default.

```text
CCF = (EAD_at_default - drawn_balance) / (credit_limit - drawn_balance)
EAD = drawn_balance + CCF × (credit_limit - drawn_balance)
```

Handle fully drawn accounts, zero denominators and contractual bounds explicitly. Compare an observed-balance or segment baseline with a regression challenger. Never imply the EAD model was learned from South German Credit.

## 12. IFRS 9-inspired expected loss

Use the empirically developed PD plus synthetic LGD/EAD outputs:

```text
ECL = Σ_t Σ_s scenario_weight_s × PD_t,s × LGD_t,s × EAD_t,s × discount_factor_t
```

Demonstrate Stage 1, Stage 2 and Stage 3 labels only as educational workflow states. Keep scenario weights, horizon, discount rate, lifetime assumptions and synthetic macro factors in configuration.

Call it **IFRS 9-inspired**, not IFRS 9-compliant. The platform cannot establish accounting compliance or a bank’s policy treatment.

## 13. Validation and explainability

For PD, report ROC AUC, Gini (`2 × AUC - 1`), KS, Brier score, calibration curve, reliability by risk band, confusion matrix, precision, recall and F1 at an explicitly chosen threshold.

Use a stratified holdout for South German Credit. Do not claim temporal backtesting on it. Use `synthetic_longitudinal_v1` for an explicitly labelled demonstration of delayed outcomes, period comparison and monitoring.

Also report:

- segment performance;
- missingness sensitivity;
- threshold sensitivity;
- feature stability and PSI;
- baseline versus challenger comparison;
- model limitations and recommendation.

SHAP outputs:

- global importance bar chart;
- beeswarm summary;
- local explanation for a selected applicant-like synthetic observation;
- comparison with logistic coefficients;
- optional partial-dependence plots for a small number of meaningful features.

Explain that attribution is not causality and does not establish legal fairness.

## 14. Champion–challenger and registry

Register every run with parameters, seed, dataset version, feature version, metrics, calibration, SHAP artefacts and environment metadata.

Lifecycle:

```text
Candidate → Validated → Staging → Approved-for-demo → Retired
```

`Approved-for-demo` means approved for this portfolio demonstration only. Require a recorded human review decision; metrics alone do not constitute model governance.

## 15. FastAPI endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/ready` | Model/database readiness |
| GET | `/model-info` | Model, version and provenance |
| POST | `/v1/pd/predict` | Single PD prediction |
| POST | `/v1/pd/batch` | Batch PD predictions |
| POST | `/v1/lgd/predict` | Synthetic LGD prediction |
| POST | `/v1/ead/predict` | Synthetic EAD prediction |
| POST | `/v1/ecl/calculate` | Expected-loss calculation |
| GET | `/v1/monitoring/summary` | Latest quality/drift summary |

Validate payloads, return request IDs and model versions, reject malformed values, avoid logging raw payloads and return safe errors.

## 16. Streamlit application blueprint

### Pages

Use Streamlit navigation with these pages:

1. **Overview:** project purpose, architecture, dataset provenance and limitations.
2. **Portfolio explorer:** South German Credit class balance, feature distributions and data-quality results.
3. **PD model lab:** baseline/challenger metrics, ROC, calibration, KS/Gini and risk-band tables.
4. **Single prediction:** form inputs, PD output, risk band, model version and SHAP explanation.
5. **Expected loss:** synthetic LGD/EAD inputs, scenario selector, Stage 1/2/3 educational result and calculation breakdown.
6. **Monitoring:** Evidently report summary, PSI, feature drift and synthetic period comparison.
7. **Model registry:** current champion, challenger, validation decision, run ID and artefact links.
8. **Documentation:** model card, data dictionary, validation report and limitations.

### Streamlit rules

- use `st.cache_data` for immutable dataset loading and `st.cache_resource` for model/API clients;
- keep model loading out of every widget rerun;
- use `st.session_state` only for small UI state;
- read configuration from environment variables and `st.secrets`, never hard-code credentials;
- default to a local demo mode that uses bundled sample artefacts;
- display a visible “educational demonstration — not for lending decisions” notice;
- make charts readable without requiring a large dashboard framework;
- keep all business calculations in `src/`, not inside page scripts;
- use API calls in hosted full-stack mode and expose the active mode in the UI.

### Streamlit entry point

Support:

```text
streamlit run streamlit_app.py
```

The entry point should select `DEMO_MODE=local` or `DEMO_MODE=api`. Local mode loads registered artefacts from a configured path. API mode calls FastAPI using `FASTAPI_BASE_URL`. Both modes must return the same response schema.

## 17. Docker Compose

Services:

- `streamlit`: user-facing app on port 8501;
- `api`: FastAPI service on port 8000;
- `postgres`: metadata and predictions;
- `mlflow`: tracking and registry.

Include health checks, named volumes, `.env.example`, non-secret local defaults and service dependency conditions.

```text
docker compose up --build
docker compose exec api python scripts/ingest_data.py
docker compose exec api python scripts/train_pd.py
docker compose exec api pytest
```

## 18. Streamlit hosting plan

Host the Streamlit application on **Streamlit Community Cloud** or an equivalent Streamlit-compatible service.

### Recommended first deployment

Use Streamlit Cloud in local demo mode:

- keep South German Credit ingestion reproducible;
- commit only small, licence-permitted sample artefacts;
- store model artefact references or small demo artefacts safely;
- put non-public values in Streamlit secrets;
- do not expose PostgreSQL or MLflow credentials;
- show the app mode and dataset provenance visibly.

### Full-stack deployment

For API mode, host FastAPI, PostgreSQL and MLflow separately using protected services, then configure `FASTAPI_BASE_URL` and any registry credentials through Streamlit secrets. Streamlit Cloud is the UI host; it is not a replacement for the API, database or MLflow server.

If external services are unavailable, retain local Docker Compose as the canonical reproducibility environment and use Streamlit Cloud only for the read-only/demo path.

## 19. Database schema

Tables:

- `datasets`: source, version, checksum, licence, empirical/synthetic flag and ingestion time;
- `features`: name, type, source, transformation and version;
- `models`: target, family, MLflow URI, status and intended use;
- `training_runs`: run ID, parameters, metrics and artefacts;
- `validation_reviews`: decision, evidence, notes and timestamp;
- `prediction_requests`: request ID, endpoint, mode and status;
- `predictions`: request ID, model version, PD/LGD/EAD/ECL and timestamp;
- `monitoring_runs`: report, reference/current dataset versions, PSI and status;
- `audit_events`: actor, action, object and timestamp.

Never store names, addresses, account numbers or raw personal identifiers. Use synthetic IDs or irreversible hashes.

## 20. Testing and CI

Test that:

- South German Credit ingestion records provenance;
- validation rejects malformed columns and ranges;
- no target leakage enters features;
- PD probabilities, LGD and CCF stay within bounds;
- synthetic generators are deterministic for a fixed seed;
- ECL is non-negative and changes predictably;
- MLflow models can be loaded;
- API schemas, request IDs and safe errors work;
- Streamlit data/model loading functions work in demo mode;
- Evidently produces a monitoring artefact.

GitHub Actions should install the pinned environment, run lint/format checks, execute pytest, build the API and Streamlit images, and run a health smoke test.

## 21. Security

- public/synthetic data only;
- no secrets in Git;
- Streamlit secrets and environment variables for credentials;
- least-privilege database access;
- no raw payload logging;
- protected MLflow and API endpoints outside localhost;
- dependency and container updates;
- safe error messages;
- visible warning that the app is not a credit-decision system.

## 22. Ten-week roadmap

### Week 1 — repository and Streamlit shell

Create the package, Docker Compose, Streamlit entry point, navigation shell, README, architecture diagram and limitations notice. Acceptance: `streamlit run streamlit_app.py` opens locally.

### Week 2 — South German Credit ingestion

Implement download/loading, checksum, provenance, data dictionary and Great Expectations checks. Acceptance: the dataset is versioned and invalid input fails clearly.

### Week 3 — PD baseline

Implement leakage-safe preprocessing, target mapping, stratified split, logistic regression, metrics and MLflow logging. Acceptance: one command trains the baseline.

### Week 4 — challenger and calibration

Add XGBoost or LightGBM, bounded Optuna tuning, calibration, ROC/KS/Gini/Brier and Streamlit comparison charts. Acceptance: baseline and challenger are compared in the app.

### Week 5 — synthetic LGD/EAD

Add deterministic recovery and revolving-exposure generators, bounded regressors and provenance labels. Acceptance: synthetic status is visible in reports and API responses.

### Week 6 — expected loss

Implement configurable scenarios, Stage 1/2/3 educational states, discounting and ECL breakdown. Acceptance: the Streamlit page explains every calculation component.

### Week 7 — explainability and validation

Add SHAP, optional partial dependence, model card, validation report and human champion decision. Acceptance: a reviewer can trace the selected model to evidence.

### Week 8 — FastAPI and PostgreSQL

Implement versioned endpoints, migrations, prediction persistence and Streamlit API client. Acceptance: local full-stack mode works through Docker Compose.

### Week 9 — monitoring and CI

Add Evidently, PSI, synthetic longitudinal monitoring, GitHub Actions and container smoke tests. Acceptance: a shifted synthetic sample creates a visible warning.

### Week 10 — hosting and portfolio release

Deploy Streamlit demo, configure secrets, add screenshots/demo script, complete limitations and tag a release. Acceptance: an unfamiliar reviewer can open, understand and reproduce the demo.

## 23. Demo scenarios

1. Compare logistic PD baseline with boosting challenger.
2. Submit one applicant-like observation and show calibrated PD, risk band and SHAP contributors.
3. Calculate synthetic expected loss under baseline, upside and downside scenarios.
4. Show that synthetic period drift triggers Evidently/PSI monitoring.
5. Open the registry page and explain the champion–challenger decision.

## 24. CV wording

> Built and hosted a Streamlit credit-risk modelling platform using South German Credit for PD development, with synthetic LGD/EAD/ECL extensions, calibrated logistic and gradient-boosting models, SHAP validation, MLflow registry, FastAPI inference, PostgreSQL, Docker, Evidently, pytest and GitHub Actions.

Use “production-style” or “production-grade portfolio architecture”, not “production lending system”.

## 25. Acceptance criteria

- South German Credit source, UCI citation, licence, checksum and target mapping are documented;
- empirical and synthetic data are physically and logically separated;
- synthetic generators are seeded, versioned and labelled;
- PD baseline and challenger are reproducible;
- validation includes ROC AUC, Gini, KS, Brier, calibration and confusion matrix;
- no temporal backtest is claimed for South German Credit;
- synthetic monitoring/backtesting is explicitly labelled;
- SHAP evidence exists;
- MLflow stores runs and model versions;
- FastAPI returns versioned predictions;
- Streamlit runs locally and in the hosted demo path;
- Docker Compose starts the full local stack;
- PostgreSQL stores metadata and predictions without personal identifiers;
- pytest and CI pass;
- the app contains clear public/synthetic-data and non-regulatory disclaimers.

## 26. Future enhancements

Only after the core platform is stable: survival PD, transition-matrix lifetime PD, macroeconomic scenario generation, vintage analysis, subgroup stability, automated challenger promotion, feature-store integration, role-based access control, cloud infrastructure as code and delayed-outcome performance monitoring.

## 27. Final guardrail

CreditRiskLab is an educational portfolio project. South German Credit is the primary empirical PD source. LGD, EAD, recovery, utilisation, longitudinal outcomes, macro scenarios and lifetime paths are synthetic extensions. The application, models, expected-loss calculations, validation results, APIs and monitoring reports must never be presented as a production lending system, bank-approved model, IFRS 9-compliant implementation, regulatory capital model or evidence of regulatory approval.
