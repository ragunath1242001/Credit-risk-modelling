import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from credit_risk_lab.extensions import generate_ead, generate_lgd, monitoring_summary

def test_synthetic_generators_are_deterministic_and_bounded():
    a, ma = generate_lgd(20); b, mb = generate_lgd(20)
    assert a.equals(b) and ma == mb and a.lgd.between(0, 1).all()
    ead, _ = generate_ead(20); assert (ead.ccf.between(0, 1)).all()

def test_monitoring_detects_shift():
    a, _ = generate_lgd(100); b = a.copy(); b.ead *= 2
    assert monitoring_summary(a, b)["status"] == "warning"
