from uuid import uuid4
from pathlib import Path
import joblib

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core import ecl, load_data, load_model, risk_bands, train_challenger
from .extensions import performance_summary
from .longitudinal import generate_longitudinal
from .db import save_prediction
from .extensions import calculate_portfolio_ecl, generate_ead, generate_lgd, monitoring_summary
from .registry import latest_run

app = FastAPI(title="CreditRiskLab API", version="0.1.0")


class Prediction(BaseModel):
    features: dict[str, float | int] = Field(min_length=1)

class BatchPrediction(BaseModel):
    rows: list[dict[str, float | int]] = Field(min_length=1, max_length=1000)


class ECL(BaseModel):
    pd: float = Field(ge=0, le=1)
    lgd: float = Field(ge=0, le=1)
    ead: float = Field(ge=0)
    scenario_weight: float = Field(default=1, ge=0, le=1)
    discount_factor: float = Field(default=1, ge=0, le=1)


class LGDRequest(BaseModel):
    ead: float = Field(gt=0)
    recoveries: float = Field(ge=0)
    recovery_costs: float = Field(ge=0)


class EADRequest(BaseModel):
    credit_limit: float = Field(gt=0)
    drawn_balance: float = Field(ge=0)
    ccf: float = Field(ge=0, le=1)


@app.get("/health")
def health(): return {"status": "ok"}


@app.get("/model-info")
def model_info(): return load_model()["metadata"]


@app.post("/v1/pd/predict")
def predict(request: Prediction):
    try:
        pack = load_model(); scorer = joblib.load("artifacts/pd_calibrated.joblib") if Path("artifacts/pd_calibrated.joblib").exists() else pack["model"]
        probability = float(scorer.predict_proba(pd.DataFrame([request.features]))[0, 1])
        result = {"request_id": str(uuid4()), "pd": probability, "model_version": pack["metadata"].get("dataset_version", "demo")}
        save_prediction(result["request_id"], "/v1/pd/predict", request.features, result)
        return result
    except Exception as exc:
        raise HTTPException(400, "invalid feature payload") from exc

@app.post("/v1/pd/batch")
def batch_predict(request: BatchPrediction):
    try:
        pack = load_model(); scorer = joblib.load("artifacts/pd_calibrated.joblib") if Path("artifacts/pd_calibrated.joblib").exists() else pack["model"]; probabilities = scorer.predict_proba(pd.DataFrame(request.rows))[:, 1]
        return {"request_id": str(uuid4()), "predictions": [{"pd": float(p), "risk_band": b} for p, b in zip(probabilities, risk_bands(probabilities))], "model_version": pack["metadata"].get("dataset_version", "demo")}
    except Exception as exc:
        raise HTTPException(400, "invalid batch feature payload") from exc


@app.post("/v1/ecl/calculate")
def calculate(request: ECL):
    result = {"request_id": str(uuid4()), "ecl": ecl(request.pd, request.lgd, request.ead, request.scenario_weight, request.discount_factor)}
    save_prediction(result["request_id"], "/v1/ecl/calculate", request.model_dump(), result)
    return result


@app.post("/v1/lgd/predict")
def lgd_predict(request: LGDRequest):
    if request.recoveries > request.ead + request.recovery_costs:
        raise HTTPException(400, "recoveries exceed exposure")
    try:
        model = joblib.load("artifacts/lgd_model.joblib"); value = float(model.predict([[request.ead, request.recoveries, request.recovery_costs, 0, 0, 1, 12]])[0])
    except Exception:
        value = (request.ead - request.recoveries + request.recovery_costs) / request.ead
    value = min(1.0, max(0.0, value))
    return {"request_id": str(uuid4()), "lgd": value, "synthetic": True}


@app.post("/v1/ead/predict")
def ead_predict(request: EADRequest):
    if request.drawn_balance > request.credit_limit:
        raise HTTPException(400, "drawn balance exceeds credit limit")
    try:
        model = joblib.load("artifacts/ead_model.joblib"); ccf = float(model.predict([[request.credit_limit, request.drawn_balance]])[0]); value = request.drawn_balance + max(0, min(1, ccf)) * (request.credit_limit - request.drawn_balance)
    except Exception:
        value = request.drawn_balance + request.ccf * (request.credit_limit - request.drawn_balance)
    return {"request_id": str(uuid4()), "ead": value, "ccf": request.ccf, "synthetic": True}


@app.get("/ready")
def ready():
    load_model()
    return {"status": "ready"}


@app.get("/v1/registry/latest")
def registry_latest(): return latest_run()

@app.get("/v1/pd/challenger")
def challenger_info():
    data, _ = load_data(); _, metrics = train_challenger(data); return metrics

@app.get("/v1/monitoring/performance")
def performance_monitoring(): return performance_summary(generate_longitudinal())


@app.get("/v1/synthetic/summary")
def synthetic_summary():
    lgd, _ = generate_lgd(100); ead, _ = generate_ead(100)
    return {"lgd_rows": len(lgd), "ead_rows": len(ead), "synthetic": True}


@app.get("/v1/monitoring/summary")
def monitoring():
    from .core import load_data
    data, _ = load_data(); reference = data.drop(columns="target"); current = reference.copy()
    for c in current.select_dtypes("number"): current[c] *= 1.15
    return monitoring_summary(reference, current)
