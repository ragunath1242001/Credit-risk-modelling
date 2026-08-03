from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, confusion_matrix, roc_auc_score,
                             roc_curve)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://archive.ics.uci.edu/static/public/522/south+german+credit.zip"
DATA_VERSION = "south_german_credit_v1"
ARTIFACT = Path("artifacts/pd_model.joblib")
CHALLENGER_ARTIFACT = Path("artifacts/pd_xgboost.joblib")
FEATURE_LABELS = {
    "laufkont": "Checking account status", "laufzeit": "Loan duration (months)", "moral": "Credit history", "verw": "Loan purpose", "hoehe": "Loan amount", "sparkont": "Savings account", "beszeit": "Employment duration", "rate": "Installment rate", "famges": "Personal status and sex", "buerge": "Guarantor or other debtor", "wohnzeit": "Residence duration", "verm": "Property", "alter": "Age", "weitkred": "Other installment plans", "wohn": "Housing", "bishkred": "Existing credits", "beruf": "Job status", "pers": "People financially liable", "telef": "Telephone", "gastarb": "Foreign worker"
}


def load_data(cache: Path | None = None) -> tuple[pd.DataFrame, dict]:
    cache = cache or Path("data/raw/south_german_credit/SouthGermanCredit.asc")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        raw = cache.read_bytes()
    else:
        blob = urlopen(DATA_URL, timeout=30).read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            raw = z.read("SouthGermanCredit.asc")
        cache.write_bytes(raw)
    checksum = hashlib.sha256(raw).hexdigest()
    frame = pd.read_csv(io.BytesIO(raw), sep=r"\s+")
    if "kredit" not in frame:
        raise ValueError("South German Credit target 'kredit' is missing")
    frame["target"] = (frame.pop("kredit") == 0).astype(int)  # UCI: 0 bad, 1 good
    meta = {"source": DATA_URL, "version": DATA_VERSION, "sha256": checksum,
            "rows": len(frame), "target": "kredit: 0 bad, 1 good; target=1 means bad"}
    return frame, meta


def validate(frame: pd.DataFrame) -> None:
    required = {"target", "laufzeit", "hoehe", "alter"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if frame.empty or not set(frame.target.unique()) <= {0, 1}:
        raise ValueError("target must contain binary values")
    if frame.isna().sum().sum() > 0:
        raise ValueError("unexpected missing values")


def train_pd(frame: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, dict]:
    validate(frame)
    x, y = frame.drop(columns="target"), frame.target
    cat = [c for c in x if x[c].nunique() <= 10]
    num = [c for c in x if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    model = Pipeline([("features", pre), ("model", LogisticRegression(max_iter=1000, random_state=seed))])
    xt, xv, yt, yv = train_test_split(x, y, test_size=.25, stratify=y, random_state=seed)
    model.fit(xt, yt)
    p = model.predict_proba(xv)[:, 1]
    auc = roc_auc_score(yv, p)
    fpr, tpr, _ = roc_curve(yv, p)
    observed, predicted = calibration_curve(yv, p, n_bins=10, strategy="quantile")
    metrics = {"roc_auc": float(auc), "gini": float(2 * auc - 1),
               "brier": float(brier_score_loss(yv, p)),
               "ks": float(np.max(tpr - fpr)),
               "confusion_matrix": confusion_matrix(yv, p >= .5).tolist(),
               "calibration": {"observed": observed.tolist(), "predicted": predicted.tolist()},
               "n_train": len(xt), "n_test": len(xv), "seed": seed,
               "dataset_version": DATA_VERSION}
    return model, metrics


def train_challenger(frame: pd.DataFrame, seed: int = 42, params: dict | None = None):
    """Bounded XGBoost challenger."""
    validate(frame); x, y = frame.drop(columns="target"), frame.target
    cat = [c for c in x if x[c].nunique() <= 10]; num = [c for c in x if c not in cat]
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num), ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
    try:
        from xgboost import XGBClassifier
        xgb_params = {"n_estimators": 120, "max_depth": 3, "learning_rate": .05, "subsample": .8, "colsample_bytree": .8, "eval_metric": "logloss", "random_state": seed, "n_jobs": 1}; xgb_params.update(params or {})
        estimator = XGBClassifier(**xgb_params)
        family = "xgboost"
    except ImportError:
        estimator = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=12, learning_rate=.05, random_state=seed)
        family = "hist_gradient_boosting_fallback"
    model = Pipeline([("features", pre), ("model", estimator)])
    xt, xv, yt, yv = train_test_split(x, y, test_size=.25, stratify=y, random_state=seed); model.fit(xt, yt); p = model.predict_proba(xv)[:, 1]
    auc = roc_auc_score(yv, p); fpr, tpr, _ = roc_curve(yv, p)
    return model, {"roc_auc": float(auc), "gini": float(2 * auc - 1), "brier": float(brier_score_loss(yv, p)), "ks": float(np.max(tpr - fpr)), "model_family": family, "seed": seed}

def tune_challenger(frame: pd.DataFrame, seed: int = 42, trials: int = 5) -> dict:
    """Small reproducible Optuna search; returns the best bounded parameters."""
    try:
        import optuna
    except ImportError:
        return {"framework": "optuna-unavailable", "best_params": {}}
    def objective(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 60, 160), "max_depth": trial.suggest_int("max_depth", 2, 4), "learning_rate": trial.suggest_float("learning_rate", .03, .15)}
        _, metrics = train_challenger(frame, seed=seed, params=params)
        return metrics["roc_auc"]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)); study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return {"framework": "optuna", "trials": trials, "best_value": study.best_value, "best_params": study.best_params}

def scorecard_bins(frame: pd.DataFrame, feature: str, bins: int = 5) -> list[dict]:
    groups = pd.qcut(frame[feature], q=bins, duplicates="drop"); result = frame.assign(_bin=groups).groupby("_bin", observed=True).target.agg(["count", "mean"]).reset_index().to_dict("records")
    return [{**row, "_bin": str(row["_bin"]), "count": int(row["count"]), "mean": float(row["mean"])} for row in result]

def threshold_metrics(y_true, probabilities, thresholds=(.2, .5, .8)) -> list[dict]:
    from sklearn.metrics import precision_score, recall_score, f1_score
    return [{"threshold": t, "precision": precision_score(y_true, probabilities >= t, zero_division=0), "recall": recall_score(y_true, probabilities >= t, zero_division=0), "f1": f1_score(y_true, probabilities >= t, zero_division=0)} for t in thresholds]


def calibrate_model(model, frame: pd.DataFrame, seed: int = 42):
    x, y = frame.drop(columns="target"), frame.target
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3); calibrated.fit(x, y)
    return calibrated


def risk_bands(probabilities: np.ndarray) -> list[str]:
    return ["low" if p < .2 else "medium" if p < .5 else "high" for p in probabilities]


def explain_sample(model, frame: pd.DataFrame, row: int = 0) -> dict:
    try:
        import shap
        features = frame.drop(columns="target")
        x = features.iloc[[row]]
        background = features.sample(min(100, len(features)), random_state=42)
        values = shap.Explainer(model.predict_proba, background)(x).values[0]
        values = values[:, 1] if values.ndim == 2 else values
        return {"method": "SHAP", "baseline": "average prediction over 100 representative applicants", "values": {c: float(v) for c, v in zip(x.columns, values)}}
    except Exception as exc:
        return {"method": "unavailable", "reason": str(exc)}


def ecl(pd_value: float, lgd: float, ead: float, scenario_weight: float = 1.0,
        discount_factor: float = 1.0) -> float:
    if not all(0 <= v <= 1 for v in (pd_value, lgd, scenario_weight, discount_factor)) or ead < 0:
        raise ValueError("PD, LGD, weights and discount must be in [0, 1]; EAD must be non-negative")
    return float(pd_value * lgd * ead * scenario_weight * discount_factor)


def save_model(model, metadata: dict) -> None:
    import joblib
    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, ARTIFACT)


def load_model():
    import joblib
    if not ARTIFACT.exists():
        model, metrics = train_pd(load_data()[0])
        save_model(model, {**metrics, "model_name": "pd_logistic_baseline", "model_family": "logistic_regression", "model_version": "pd-v1", "status": "approved-for-demo", "calibrated": False})
    return joblib.load(ARTIFACT)

def load_models() -> dict:
    import joblib
    data, _ = load_data(); baseline = load_model(); models = {"Logistic regression": baseline["model"]}
    if Path("artifacts/pd_calibrated.joblib").exists(): models["Calibrated logistic regression"] = joblib.load("artifacts/pd_calibrated.joblib")
    if CHALLENGER_ARTIFACT.exists():
        models["XGBoost challenger"] = joblib.load(CHALLENGER_ARTIFACT)
    else:
        challenger, _ = train_challenger(data); models["XGBoost challenger"] = challenger
    return models


if __name__ == "__main__":
    data, provenance = load_data(); model, metrics = train_pd(data)
    save_model(model, {**provenance, **metrics})
    print(json.dumps({**provenance, **metrics}, indent=2))
