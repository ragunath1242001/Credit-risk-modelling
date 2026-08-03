import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from credit_risk_lab.extensions import generate_ead, generate_lgd

Path("data/processed").mkdir(parents=True, exist_ok=True)
lgd, lgd_meta = generate_lgd(); ead, ead_meta = generate_ead()
lgd.to_csv("data/processed/synthetic_recovery_v1.csv", index=False)
ead.to_csv("data/processed/synthetic_revolving_ead_v1.csv", index=False)
print(lgd_meta); print(ead_meta)
