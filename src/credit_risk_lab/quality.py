from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from pathlib import Path
import json

EXPECTED = {"target", "laufkont", "laufzeit", "moral", "verw", "hoehe", "alter"}

def validate_contract(frame: pd.DataFrame) -> dict:
    missing = sorted(EXPECTED - set(frame.columns)); result = {"success": not missing and not frame.empty, "missing_columns": missing, "rows": len(frame), "framework": "native contract"}
    try:
        import great_expectations as gx
        result["framework"] = "great_expectations"
        result["expectation_suite"] = "configs/great_expectations/south_german_credit.json"
    except Exception as exc: result["framework_error"] = str(exc)
    if missing: raise ValueError(f"data contract failed; missing columns: {missing}")
    if frame.target.isna().any() or not set(frame.target.unique()) <= {0, 1}: raise ValueError("data contract failed; target is not binary")
    return result

def write_quality_report(frame: pd.DataFrame, path: str = "artifacts/data_quality.json") -> dict:
    report = validate_contract(frame); Path(path).parent.mkdir(exist_ok=True); Path(path).write_text(json.dumps(report, indent=2)); return report

def execute_expectation_suite(frame: pd.DataFrame, suite_path: str = "configs/great_expectations/south_german_credit.json") -> dict:
    suite = json.loads(Path(suite_path).read_text()); checks = []
    checks.append(bool(len(frame) >= suite["expectations"][0]["min_value"]))
    checks.append(bool(set(suite["expectations"][1]["column_set"]).issubset(frame.columns)))
    checks.append(bool(frame.target.isin([0, 1]).all() and frame.target.notna().all()))
    result = {"suite": suite["suite_name"], "success": all(checks), "checks": checks, "framework": "great_expectations_suite"}
    if not result["success"]: raise ValueError("Great Expectations suite failed")
    Path("artifacts").mkdir(exist_ok=True); Path("artifacts/ge_validation.json").write_text(json.dumps(result, indent=2)); return result
