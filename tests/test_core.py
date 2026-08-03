import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
from credit_risk_lab.core import ecl, load_data, train_pd


def test_training_and_metrics():
    data, meta = load_data()
    model, metrics = train_pd(data)
    assert meta["version"] == "south_german_credit_v1"
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= model.predict_proba(data.drop(columns="target").iloc[:2])[:, 1].min() <= 1


def test_ecl_bounds():
    assert ecl(.2, .4, 100) == pytest.approx(8)
    with pytest.raises(ValueError): ecl(1.1, .4, 100)

