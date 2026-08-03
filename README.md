# CreditRiskLab

CreditRiskLab is a reproducible, portfolio-grade credit-risk modelling platform built with the public [South German Credit dataset](https://archive-beta.ics.uci.edu/dataset/522/south%2Bgerman%2Bcredit).

It demonstrates the end-to-end workflow for binary credit-risk outcome modelling:

```text
UCI data → provenance and quality checks → PD models → calibration and evidence
                                      ↓
                         MLflow tracking and model artifacts
                                      ↓
                   FastAPI inference → Streamlit exploration
                                      ↓
                 synthetic LGD/EAD/ECL and drift monitoring
```

> Educational demonstration only. This project is not a lending decision system, regulatory capital model, bank-approved model, or IFRS 9-compliant implementation.

## What is included

- Reproducible South German Credit ingestion with SHA-256 provenance.
- Leakage-safe scikit-learn preprocessing and logistic-regression PD baseline.
- XGBoost challenger with bounded Optuna tuning.
- Stratified holdout metrics: ROC AUC, Gini, KS, Brier score, confusion matrix, calibration, threshold, segment, and scorecard evidence.
- Calibrated PD artifact and SHAP explanation artifact.
- Seeded synthetic LGD, revolving EAD, and longitudinal monitoring data.
- Trained synthetic LGD/EAD regression artifacts and formula-based ECL calculation.
- Great Expectations-compatible data contract and Evidently drift report.
- MLflow run/artifact tracking with a local registry record.
- FastAPI single, batch, LGD, EAD, ECL, registry, and monitoring endpoints.
- Streamlit local demo with sidebar navigation.
- SQLite local persistence with optional PostgreSQL support.
- GitHub Actions test workflow, pinned `requirements.txt`, and documentation.

## Quick start

Python 3.12 is recommended.

```powershell
git clone https://github.com/ragunath1242001/Credit-risk-modelling.git
cd Credit-risk-modelling
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train the models and generate evidence artifacts:

```powershell
python scripts/train_pd.py
python scripts/generate_synthetic.py
python scripts/generate_reports.py
```

Start the Streamlit application:

```powershell
streamlit run streamlit_app.py
```

The app runs in local demo mode and does not require Docker, PostgreSQL, or an externally hosted API.

## API

Start the FastAPI service in a second terminal:

```powershell
uvicorn credit_risk_lab.api:app --reload
```

Useful endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /ready` | Model readiness check |
| `GET /model-info` | Model and dataset metadata |
| `POST /v1/pd/predict` | Single PD prediction |
| `POST /v1/pd/batch` | Batch PD predictions |
| `POST /v1/lgd/predict` | Synthetic LGD prediction |
| `POST /v1/ead/predict` | Synthetic EAD prediction |
| `POST /v1/ecl/calculate` | Expected-loss calculation |
| `GET /v1/monitoring/summary` | PSI drift summary |
| `GET /v1/monitoring/performance` | Synthetic longitudinal performance |
| `GET /v1/registry/latest` | Latest tracked run |

Interactive API documentation is available at `http://127.0.0.1:8000/docs` after starting FastAPI.

## Example PD request

```powershell
$body = @{ features = @{ laufkont = 1; laufzeit = 18; moral = 4; verw = 2; hoehe = 1049; sparkont = 1; beszeit = 2; rate = 4; famges = 2; buerge = 1; wohnzeit = 4; verm = 2; alter = 21; weitkred = 3; wohn = 1; bishkred = 1; beruf = 3; pers = 2; telef = 1; gastarb = 2 } } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/pd/predict -Method Post -ContentType 'application/json' -Body $body
```

## Model results

The reproducible baseline run currently reports approximately:

| Model | ROC AUC | Gini | KS | Brier |
|---|---:|---:|---:|---:|
| Logistic regression | 0.754 | 0.508 | 0.377 | 0.175 |
| XGBoost challenger | 0.773 | 0.546 | 0.394 | 0.166 |

These results are holdout results on a small historical public dataset. They are not evidence of production performance.

## Project layout

```text
configs/                 Dataset, model, scenario, and quality-suite configuration
data/                    Downloaded empirical and generated synthetic data
docs/                    Architecture, model card, dictionary, validation, limitations
scripts/                 Training, generation, and report commands
src/credit_risk_lab/    Core modelling, API, registry, database, quality, and monitoring code
tests/                   Unit and integration smoke tests
streamlit_app.py        Streamlit entry point
requirements.txt        Pinned runtime environment
```

## Generated artifacts

Running the training/report commands creates local files under `artifacts/`, including:

- `pd_model.joblib` and `pd_calibrated.joblib`
- `lgd_model.joblib` and `ead_model.joblib`
- `registry.json` and MLflow run data
- `ge_validation.json` and `data_quality.json`
- `monitoring_report.html`
- `shap.json`, `partial_dependence.png`, and validation evidence

Artifacts, local databases, and build output are excluded from Git by `.gitignore`.

## Streamlit Cloud

The app can be deployed directly from this repository using `streamlit_app.py` as the entry point. Use the repository root as the app directory and configure only non-public values through Streamlit Cloud secrets. A template is provided at `.streamlit/secrets.toml.example`.

The hosted local-demo path does not need PostgreSQL, FastAPI, MLflow, or Docker. Those services are optional for fuller local integration and model-serving workflows.

## Testing

```powershell
python -m pytest -q
```

The test suite covers deterministic synthetic generation, bounds, ECL, data contracts, challenger training, and longitudinal generation.

## Data and modelling limitations

South German Credit is small, historical, single-outcome, and lacks reliable observation dates, recovery cash flows, collateral outcomes, utilisation histories, and macroeconomic histories. Therefore:

- the empirical target is a binary credit-risk outcome, not a documented 12-month PD;
- no temporal backtest is claimed;
- LGD, EAD, recovery, longitudinal, macro, and monitoring examples are synthetic;
- synthetic results do not represent observed bank behaviour;
- model attribution is not causal and does not establish legal fairness;
- expected loss is IFRS 9-inspired education, not accounting compliance.

See [docs/limitations.md](docs/limitations.md), [docs/model_card.md](docs/model_card.md), and [docs/validation_report.md](docs/validation_report.md) for more detail.

## License

This project is released under the MIT License. Dataset terms and attribution remain subject to the original UCI source and its documentation.
