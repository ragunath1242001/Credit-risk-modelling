import numpy as np
import pandas as pd

def generate_longitudinal(n: int = 1000, periods: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed); rows = []
    for period in range(periods):
        risk = rng.normal(.2 + period * .03, .05, n).clip(.01, .95)
        rows.append(pd.DataFrame({"period": period, "synthetic": True, "pd": risk, "outcome": rng.binomial(1, risk)}))
    return pd.concat(rows, ignore_index=True)
