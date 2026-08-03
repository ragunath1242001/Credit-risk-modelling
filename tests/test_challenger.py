import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from credit_risk_lab.core import load_data, train_challenger
from credit_risk_lab.longitudinal import generate_longitudinal
from credit_risk_lab.quality import validate_contract

def test_challenger_and_contract():
    data, _ = load_data(); _, metrics = train_challenger(data)
    assert 0 <= metrics["roc_auc"] <= 1 and validate_contract(data)["success"]

def test_longitudinal_is_seeded():
    assert generate_longitudinal(5, 2).equals(generate_longitudinal(5, 2))
