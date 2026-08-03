import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import joblib
from credit_risk_lab.core import load_data, train_pd, train_challenger, tune_challenger, save_model, calibrate_model, explain_sample, scorecard_bins
from credit_risk_lab.quality import write_quality_report, execute_expectation_suite
from credit_risk_lab.registry import log_run
from credit_risk_lab.extensions import generate_ead, generate_lgd, train_ead_model, train_lgd_model
from credit_risk_lab.db import save_metadata
from credit_risk_lab.db import save_review

data, provenance = load_data(); model, metrics = train_pd(data); _, challenger_metrics = train_challenger(data)
write_quality_report(data); execute_expectation_suite(data); Path("artifacts").mkdir(exist_ok=True)
joblib.dump(calibrate_model(model, data), "artifacts/pd_calibrated.joblib")
Path("artifacts/shap.json").write_text(str(explain_sample(model, data)))
lgd, _ = generate_lgd(); ead, _ = generate_ead(); lgd_model, _ = train_lgd_model(lgd); ead_model, _ = train_ead_model(ead)
joblib.dump(lgd_model, "artifacts/lgd_model.joblib"); joblib.dump(ead_model, "artifacts/ead_model.joblib")
metadata = {**provenance, **metrics, "model_family": "logistic_regression", "status": "approved-for-demo", "challenger_roc_auc": challenger_metrics["roc_auc"], "calibrated": True, "shap_artifact": "artifacts/shap.json"}
metadata["optuna"] = tune_challenger(data, trials=3); metadata["scorecard_bins"] = scorecard_bins(data, "hoehe")
save_model(model, metadata); save_metadata(provenance["version"], provenance["source"], provenance["sha256"], provenance["version"], "logistic_regression", metadata)
save_review("approved-for-demo", {"baseline": metrics, "challenger": challenger_metrics, "optuna": metadata["optuna"]})
log_run({**metadata, "artifact_path": "artifacts/pd_model.joblib"})
print({"baseline": metrics, "challenger": challenger_metrics})
