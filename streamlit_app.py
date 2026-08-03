import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import json
from credit_risk_lab.core import load_data, load_model, explain_sample
from credit_risk_lab.extensions import calculate_portfolio_ecl, generate_ead, generate_lgd, monitoring_summary
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
    st.write("A reproducible credit-risk modelling platform built from public data."); st.json(provenance)
elif page == "Portfolio explorer":
    st.metric("Rows", len(data)); st.metric("Bad-outcome rate", f"{data.target.mean():.1%}"); st.bar_chart(data.target.value_counts())
    st.dataframe(data.head(10), use_container_width=True)
elif page == "PD model lab":
    cols = st.columns(4)
    for col, key in zip(cols, ["roc_auc", "gini", "ks", "brier"]): col.metric(key.upper(), f"{pack['metadata'][key]:.3f}")
    st.json(pack["metadata"])
    if pack["metadata"].get("calibration"):
        st.line_chart({"observed": pack["metadata"]["calibration"]["observed"], "predicted": pack["metadata"]["calibration"]["predicted"]})
    if Path("artifacts/validation_evidence.json").exists():
        evidence = json.loads(Path("artifacts/validation_evidence.json").read_text()); st.subheader("Threshold sensitivity"); st.dataframe(evidence["thresholds"]); st.subheader("Segment evidence"); st.json(evidence["segments"])
elif page == "Single prediction":
    row = data.drop(columns="target").iloc[[0]]
    if st.button("Score sample applicant"):
        p = float(pack["model"].predict_proba(row)[0, 1]); st.metric("Probability of bad outcome", f"{p:.1%}"); st.write("Risk band:", "high" if p >= .5 else "medium" if p >= .2 else "low")
    st.json(explain_sample(pack["model"], data))
elif page == "Expected loss":
    lgd, _ = generate_lgd(100); ead, _ = generate_ead(100); rows = lgd[["lgd"]].join(ead[["ead_at_default"]].rename(columns={"ead_at_default": "ead"})); rows["pd"] = .2
    scenario = st.selectbox("Scenario", ["upside", "base", "downside"]); st.metric("Synthetic ECL", f"{calculate_portfolio_ecl(rows, scenario)['ecl']:,.2f}")
elif page == "Monitoring":
    reference = data.drop(columns="target"); current = reference * 1.15; st.json(monitoring_summary(reference, current))
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
    with st.expander("Raw run metadata"):
        st.json(metadata)
else:
    st.markdown("South German Credit is historical, small, and not longitudinal. LGD, EAD, recovery, monitoring and expected-loss examples are synthetic. No temporal backtest or regulatory/IFRS 9 claim is made.")
