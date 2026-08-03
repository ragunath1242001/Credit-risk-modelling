# CreditRiskLab

Educational credit-risk modelling platform based on the public [South German Credit](https://archive-beta.ics.uci.edu/dataset/522/south%2Bgerman%2Bcredit) dataset.

```powershell
python scripts/train_pd.py
uvicorn credit_risk_lab.api:app --reload
streamlit run streamlit_app.py
```

The empirical dataset is used only for binary PD demonstration. LGD, EAD, ECL and monitoring examples are synthetic and explicitly labelled. This is not a lending, regulatory, or IFRS 9-compliant system.

