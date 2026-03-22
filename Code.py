"""
STEP 1: CLEANING DATA 
"""

import pandas as pd

df = pd.read_excel("data/Filtered_RI_T_USD_M_2025.xlsx")
print(df.head())
monthly_returns = pd.read_excel("data/Filtered_RI_T_USD_M_2025.xlsx")
monthly_returns = monthly_returns.dropna(how='all')
monthly_returns.iloc[:, 2:] = monthly_returns.iloc[:, 2:].where(monthly_returns.iloc[:, 2:] >= 0.5)
monthly_returns.iloc[:, 2:] = monthly_returns.iloc[:, 2:].ffill(axis=1)
price_data = monthly_returns.drop(columns=["NAME", "ISIN"])
zero_returns = (price_data.iloc[:, 2:] == 0).T.rolling(window=120).mean().T
stale_mask = zero_returns > 0.5
monthly_returns_clean = monthly_returns.loc[:, :2].copy()
monthly_returns_clean.to_csv("data/clean_returns.csv", index=False)