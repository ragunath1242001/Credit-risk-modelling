from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
REGISTRY = Path("artifacts/registry.json")

def log_run(metadata: dict) -> dict:
    REGISTRY.parent.mkdir(exist_ok=True); runs = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    run = {"run_id": f"local-{len(runs)+1:04d}", "created_at": datetime.now(timezone.utc).isoformat(), **metadata}; runs.append(run)
    try:
        import mlflow
        with mlflow.start_run(run_name=run["run_id"]):
            for k, v in metadata.items():
                if isinstance(v, (int, float)): mlflow.log_metric(k, v)
            mlflow.set_tag("dataset_version", metadata.get("dataset_version", "unknown"))
            if metadata.get("artifact_path"):
                mlflow.log_artifact(metadata["artifact_path"])
        run["mlflow"] = True
    except Exception as exc: run["mlflow_error"] = str(exc)
    runs[-1] = run
    REGISTRY.write_text(json.dumps(runs, indent=2))
    return run

def latest_run() -> dict:
    return json.loads(REGISTRY.read_text())[-1] if REGISTRY.exists() else {}
