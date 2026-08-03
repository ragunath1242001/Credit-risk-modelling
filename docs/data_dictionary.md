# Data dictionary

The 20 original predictors are preserved internally as documented integer-coded UCI fields, but the Streamlit UI displays English labels:

| Original field | English label |
|---|---|
| `laufkont` | Checking account status |
| `laufzeit` | Loan duration (months) |
| `moral` | Credit history |
| `verw` | Loan purpose |
| `hoehe` | Loan amount |
| `sparkont` | Savings account |
| `beszeit` | Employment duration |
| `rate` | Installment rate |
| `famges` | Personal status and sex |
| `buerge` | Guarantor or other debtor |
| `wohnzeit` | Residence duration |
| `verm` | Property |
| `alter` | Age |
| `weitkred` | Other installment plans |
| `wohn` | Housing |
| `bishkred` | Existing credits |
| `beruf` | Job status |
| `pers` | People financially liable |
| `telef` | Telephone |
| `gastarb` | Foreign worker |

`target=1` means the original `kredit=0` bad outcome. Synthetic LGD, EAD, and longitudinal data are stored separately and always carry `synthetic=true`.
