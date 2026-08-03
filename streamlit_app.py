import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import json
from credit_risk_lab.core import load_data, load_model, explain_sample
from credit_risk_lab.extensions import calculate_portfolio_ecl, generate_ead, generate_lgd, monitoring_summary, performance_summary
from credit_risk_lab.longitudinal import generate_longitudinal
from credit_risk_lab.registry import latest_run

st.set_page_config(page_title="CreditRiskLab", layout="wide")
st.warning("Educational demonstration — not for lending decisions, regulatory capital, or IFRS 9 compliance.")
st.title("CreditRiskLab")
st.caption("Local demo · empirical PD · synthetic LGD/EAD/ECL/monitoring")

@st.cache_data
def dataset(): return load_data()

@st.cache_resource
def model(): return load_model()

data, provenance = dataset(); pack = model()
page = st.sidebar.radio("Navigate", ["Overview", "Portfolio explorer", "PD model lab", "Single prediction", "Expected loss", "Monitoring", "Model registry", "Documentation"])

if page == "Overview":
    st.write("A reproducible credit-risk modelling platform built from public data.")
    overview = st.columns(4); overview[0].metric("Dataset", provenance.get("version", "unknown")); overview[1].metric("Rows", provenance.get("rows", len(data))); overview[2].metric("Target", "Binary bad outcome"); overview[3].metric("Source", "UCI")
    st.subheader("Data provenance")
    st.table({"Field": ["Source", "Dataset version", "SHA-256", "Target mapping"], "Value": [provenance.get("source"), provenance.get("version"), provenance.get("sha256"), provenance.get("target")]})
elif page == "Portfolio explorer":
    st.metric("Rows", len(data)); st.metric("Bad-outcome rate", f"{data.target.mean():.1%}"); st.bar_chart(data.target.value_counts())
    st.dataframe(data.head(10), use_container_width=True)
elif page == "PD model lab":
    cols = st.columns(4)
    for col, key in zip(cols, ["roc_auc", "gini", "ks", "brier"]): col.metric(key.upper(), f"{pack['metadata'][key]:.3f}")
    st.subheader("Validation summary")
    st.table({"Metric": ["ROC AUC", "Gini", "KS", "Brier", "Training rows", "Test rows", "Seed"], "Value": [pack["metadata"].get("roc_auc"), pack["metadata"].get("gini"), pack["metadata"].get("ks"), pack["metadata"].get("brier"), pack["metadata"].get("n_train"), pack["metadata"].get("n_test"), pack["metadata"].get("seed")]})
    matrix = pack["metadata"].get("confusion_matrix")
    if matrix: st.subheader("Confusion matrix"); st.dataframe({"Actual good": matrix[0], "Actual bad": matrix[1]}, use_container_width=True)
    if pack["metadata"].get("calibration"):
        st.line_chart({"observed": pack["metadata"]["calibration"]["observed"], "predicted": pack["metadata"]["calibration"]["predicted"]})
    if Path("artifacts/validation_evidence.json").exists():
        evidence = json.loads(Path("artifacts/validation_evidence.json").read_text()); st.subheader("Threshold sensitivity"); st.dataframe(evidence["thresholds"], use_container_width=True, hide_index=True); st.subheader("Segment evidence"); st.dataframe([{"Segment": k, **v} for k, v in evidence["segments"].items()], use_container_width=True, hide_index=True)
elif page == "Single prediction":
    row = data.drop(columns="target").iloc[[0]]
    if st.button("Score sample applicant"):
        p = float(pack["model"].predict_proba(row)[0, 1]); st.metric("Probability of bad outcome", f"{p:.1%}"); st.write("Risk band:", "high" if p >= .5 else "medium" if p >= .2 else "low")
    explanation = explain_sample(pack["model"], data)
    st.subheader("Applicant features"); st.dataframe(row, use_container_width=True, hide_index=True)
    st.subheader("SHAP contributors")
    if explanation.get("values"):
        contributors = sorted(({"Feature": k, "Contribution": v} for k, v in explanation["values"].items()), key=lambda item: abs(item["Contribution"]), reverse=True)
        st.dataframe(contributors, use_container_width=True, hide_index=True)
    else: st.info("SHAP explanation is unavailable in this hosted runtime.")
elif page == "Expected loss":
    lgd, _ = generate_lgd(100); ead, _ = generate_ead(100); rows = lgd[["lgd"]].join(ead[["ead_at_default"]].rename(columns={"ead_at_default": "ead"})); rows["pd"] = .2
    scenario = st.selectbox("Scenario", ["upside", "base", "downside"]); st.metric("Synthetic ECL", f"{calculate_portfolio_ecl(rows, scenario)['ecl']:,.2f}")
elif page == "Monitoring":
    reference = data.drop(columns="target"); current = reference * 1.15; summary = monitoring_summary(reference, current)
    status = summary["status"].upper(); st.metric("Drift status", status); st.caption("PSI is illustrative synthetic monitoring; it is not a temporal backtest of South German Credit.")
    st.subheader("Feature drift")
    drift = [{"Feature": feature, "PSI": value, "Interpretation": "warning" if value >= .25 else "stable"} for feature, value in summary["features"].items()]
    st.dataframe(drift, use_container_width=True, hide_index=True)
    st.subheader("Synthetic longitudinal performance")
    performance = performance_summary(generate_longitudinal())
    performance_rows = [{"Period": period, **values} for period, values in performance["periods"].items()]
    st.dataframe(performance_rows, use_container_width=True, hide_index=True)
    st.info("A warning means the illustrative shifted sample exceeded the PSI threshold of 0.25. Investigate before treating any monitoring result as model evidence.")
elif page == "Model registry":
    metadata = latest_run() or pack["metadata"]
    st.subheader(metadata.get("model_name", "pd_logistic_baseline"))
    identity = st.columns(4)
    identity[0].metric("Version", metadata.get("model_version", "pd-v1"))
    identity[1].metric("Family", metadata.get("model_family", "logistic_regression"))
    identity[2].metric("Status", metadata.get("status", "approved-for-demo"))
    identity[3].metric("Dataset", metadata.get("dataset_version", "south_german_credit_v1"))
    st.caption("Approved for this educational demonstration only.")
    st.subheader("Validation metrics")
    st.table({"Metric": ["ROC AUC", "Gini", "KS", "Brier", "Train rows", "Test rows"], "Value": [metadata.get("roc_auc"), metadata.get("gini"), metadata.get("ks"), metadata.get("brier"), metadata.get("n_train"), metadata.get("n_test")]})
    st.subheader("Evidence")
    evidence = {"Calibrated probabilities": metadata.get("calibrated", False), "SHAP artifact": Path(metadata.get("shap_artifact", "artifacts/shap.json")).exists(), "Optuna tuning": bool(metadata.get("optuna")), "Challenger ROC AUC": metadata.get("challenger_roc_auc", "not available in hosted fallback")}
    st.table({"Item": list(evidence), "Value": list(evidence.values())})
    st.caption("Detailed run metadata is stored in the local registry/MLflow artifacts.")
else:
    st.markdown("South German Credit is historical, small, and not longitudinal. LGD, EAD, recovery, monitoring and expected-loss examples are synthetic. No temporal backtest or regulatory/IFRS 9 claim is made.")
