from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .core import ecl

def _meta(name: str, seed: int, config: dict) -> dict:
    return {"dataset_version": name, "synthetic": True, "seed": seed,
            "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]}

def generate_lgd(n: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed); ead = rng.uniform(500, 50000, n); collateral = rng.binomial(1, .45, n)
    recoveries = ead * np.clip(.25 + .35 * collateral + rng.normal(0, .12, n), 0, 1); costs = ead * rng.uniform(.01, .08, n)
    df = pd.DataFrame({"ead": ead, "recoveries": recoveries, "recovery_costs": costs, "collateral": collateral,
                       "product_type": rng.integers(0, 3, n), "seniority": rng.integers(1, 3, n), "time_to_recovery": rng.uniform(1, 36, n)})
    df["lgd"] = np.clip((df.ead - df.recoveries + df.recovery_costs) / df.ead, 0, 1)
    return df, _meta("synthetic_recovery_v1", seed, {"n": n})

def generate_ead(n: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed); limit = rng.uniform(1000, 100000, n); drawn = limit * rng.uniform(.1, .95, n)
    ccf = np.clip(rng.beta(4, 3, n), 0, 1); ead = drawn + ccf * (limit - drawn)
    return pd.DataFrame({"credit_limit": limit, "drawn_balance": drawn, "ead_at_default": ead, "ccf": ccf}), _meta("synthetic_revolving_ead_v1", seed, {"n": n})

def calculate_portfolio_ecl(rows: pd.DataFrame, scenario: str = "base") -> dict:
    multipliers = {"upside": .8, "base": 1.0, "downside": 1.25}
    if scenario not in multipliers: raise ValueError("scenario must be upside, base, or downside")
    total = sum(ecl(float(r.pd) * multipliers[scenario], float(r.lgd), float(r.ead)) for r in rows.itertuples())
    return {"scenario": scenario, "ecl": total, "rows": len(rows), "synthetic": True}

def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    a = np.histogram(reference, edges)[0] / len(reference); b = np.histogram(current, edges)[0] / len(current)
    a, b = np.clip(a, 1e-6, 1), np.clip(b, 1e-6, 1)
    return float(np.sum((b - a) * np.log(b / a)))

def monitoring_summary(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    common = [c for c in reference.select_dtypes("number") if c in current]
    values = {c: psi(reference[c], current[c]) for c in common}
    return {"synthetic": True, "features": values, "status": "warning" if any(v >= .25 for v in values.values()) else "ok"}

def train_lgd_model(frame: pd.DataFrame, seed: int = 42):
    features = ["ead", "recoveries", "recovery_costs", "collateral", "product_type", "seniority", "time_to_recovery"]
    model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=seed, n_jobs=1).fit(frame[features], frame["lgd"])
    return model, {"model_family": "random_forest_regressor", "synthetic": True, "target": "lgd"}

def train_ead_model(frame: pd.DataFrame, seed: int = 42):
    features = ["credit_limit", "drawn_balance"]
    model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=seed, n_jobs=1).fit(frame[features], frame["ccf"])
    return model, {"model_family": "random_forest_regressor", "synthetic": True, "target": "ccf"}

def performance_summary(frame: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    return {"synthetic": True, "periods": {str(p): {"rows": int(len(g)), "roc_auc": float(roc_auc_score(g.outcome, g.pd)) if g.outcome.nunique() > 1 else None, "brier": float(brier_score_loss(g.outcome, g.pd))} for p, g in frame.groupby("period")}}

def write_monitoring_report(reference: pd.DataFrame, current: pd.DataFrame, path: str = "artifacts/monitoring_report.html") -> dict:
    result = monitoring_summary(reference, current)
    rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in result["features"].items())
    from pathlib import Path
    Path(path).parent.mkdir(exist_ok=True)
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        Report([DataDriftPreset()]).run(current_data=current, reference_data=reference).save_html(path)
        result["framework"] = "evidently"
    except Exception as exc:
        result["framework"] = "native_psi"; result["framework_error"] = str(exc)
        Path(path).write_text(f"<html><body><h1>CreditRiskLab monitoring</h1><p>Status: {result['status']}; synthetic={result['synthetic']}</p><table><tr><th>Feature</th><th>PSI</th></tr>{rows}</table></body></html>")
    return result
