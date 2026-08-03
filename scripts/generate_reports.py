import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from credit_risk_lab.core import load_data, load_model, explain_sample, scorecard_bins, threshold_metrics
from credit_risk_lab.quality import write_quality_report
from credit_risk_lab.longitudinal import generate_longitudinal
from credit_risk_lab.extensions import write_monitoring_report

data, meta = load_data(); pack = load_model(); write_quality_report(data)
Path("artifacts").mkdir(exist_ok=True); Path("artifacts/model_card.md").write_text(f"# Model card\n\nDataset: {meta['version']}\n\nROC AUC: {pack['metadata'].get('roc_auc')}\n\nThis educational model is not for lending decisions.\n")
Path("artifacts/shap.json").write_text(str(explain_sample(pack["model"], data)))
Path("artifacts/scorecard_bins.json").write_text(__import__('json').dumps(scorecard_bins(data, "hoehe"), indent=2))
probabilities = pack["model"].predict_proba(data.drop(columns="target"))[:, 1]
evidence = {"thresholds": threshold_metrics(data.target.to_numpy(), probabilities), "segments": {str(k): {"rows": int(len(g)), "bad_rate": float(g.target.mean())} for k, g in data.groupby("laufkont")}}
Path("artifacts/validation_evidence.json").write_text(__import__('json').dumps(evidence, indent=2))
try:
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay
    PartialDependenceDisplay.from_estimator(pack["model"], data.drop(columns="target"), ["hoehe", "alter"])
    plt.savefig("artifacts/partial_dependence.png", dpi=120, bbox_inches="tight"); plt.close()
except Exception as exc:
    Path("artifacts/partial_dependence.txt").write_text(str(exc))
generate_longitudinal().to_csv("data/processed/synthetic_longitudinal_v1.csv", index=False)
write_monitoring_report(data.drop(columns="target"), current=data.drop(columns="target") * 1.15)
print(pack["metadata"])
