import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from credit_risk_lab.core import load_data
from credit_risk_lab.extensions import write_monitoring_report

data, _ = load_data(); current = data.drop(columns="target").copy()
numeric = current.select_dtypes("number").columns
current[numeric] = current[numeric] * 1.15
print(write_monitoring_report(data.drop(columns="target"), current))
