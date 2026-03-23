"""
STEP 1: CLEANING DATA 
Cleaning monthly returns per project guidelines
"""

import pandas as pd
import numpy as np

# ── 1. LOAD ──────────────────────────────────────────────────────────────────
monthly_returns = pd.read_excel("data/Filtered_RI_T_USD_M_2025.xlsx")

# Columns : NAME, ISIN, then one column per month from 1999 to 2026

# ── 2. DROP FULLY EMPTY ROWS ─────────────────────────────────────────────────
monthly_returns = monthly_returns.dropna(how="all")

# ── 3. IGNORE LOW PRICES (< 0.5) ────────────────────────────────────────────
# Per guidelines prices below 0.5 are treated as missing
price_cols = monthly_returns.columns[2:]
monthly_returns[price_cols] = monthly_returns[price_cols].where(
    monthly_returns[price_cols] >= 0.5
)

# ── 4. HANDLE DELISTINGS (→ price goes to 0) ───────────────────
# Distinguish trailing empty values (delisting) from mid-sample ones (gaps).
def mark_delisting(row):
    """After last valid observation, set price to 0 (return will be -100%)."""
    last_valid = row.last_valid_index()
    if last_valid is None:
        return row
    last_pos = row.index.get_loc(last_valid)
    row.iloc[last_pos + 1:] = 0  # price goes to zero at delisting
    return row

price_data = monthly_returns[price_cols].apply(mark_delisting, axis=1)
monthly_returns[price_cols] = price_data

# ── 5. FILL MID-SAMPLE GAPS ──────────────────────────────────────────
# Only fill gaps between valid values
monthly_returns[price_cols] = monthly_returns[price_cols].ffill(axis=1)

# ── 6. COMPUTE SIMPLE MONTHLY RETURNS ────────────────────────────────────────
# R_t = RI_t / RI_{t-1} - 1
prices = monthly_returns[price_cols].values.astype(float)
returns_matrix = np.full_like(prices, np.nan)
returns_matrix[:, 1:] = prices[:, 1:] / prices[:, :-1] - 1

returns_df = pd.DataFrame(
    returns_matrix,
    index=monthly_returns.index,
    columns=price_cols
)
returns_df = pd.concat([monthly_returns[["NAME", "ISIN"]], returns_df], axis=1)

# ── 7. FLAG STALE SECURITIES ──────────────────────────────────────────────────
# For each year-end:
# If proportion of zero returns > 50%, exclude from that year's investment set.
# This is applied dynamically for every investment year.
# Here we pre-compute the full zero-return indicator matrix for convenience.

ret_only = returns_df[price_cols]
is_zero = (ret_only == 0).astype(float)

# ── 8. BUILD YEARLY FEASIBLE INVESTMENT SET ─────────────────────────────────────
# At each year-end Y, a firm is investable if:
#   (a) its last RI at end-Y is not NaN (i.e., not already delisted/missing)
#   (b) it has at least 36 valid return observations in the trailing 120 months
#   (c) zero-return proportion in trailing 120 months ≤ 50%
#   (d) [Part II] it has carbon data available — to be merged later

# Map column names to positions for easy indexing
date_cols = list(price_cols)

year_ends = {y: f"{y}-12" for y in range(2013, 2025)}  # adjust to your col format

investable_universe = {}  # dict: year → list of valid ISINs

for year, end_col in year_ends.items():
    if end_col not in date_cols:
        continue
    end_idx = date_cols.index(end_col)
    window_start = max(0, end_idx - 119)
    window = ret_only.iloc[:, window_start:end_idx + 1]

    # (a) Price must exist at year-end in RI
    ri_at_year_end = monthly_returns[end_col]
    has_price = ri_at_year_end.notna() & (ri_at_year_end > 0)

    # (b) Minimum 36 valid return observations in window
    valid_obs = window.notna().sum(axis=1)
    enough_history = valid_obs >= 36

    # (c) Stale price filter
    zero_proportion = (window == 0).sum(axis=1) / window.shape[1]
    not_stale = zero_proportion <= 0.5

    mask = has_price & enough_history & not_stale
    investable_universe[year] = monthly_returns.loc[mask, "ISIN"].tolist()
    print(f"{year}: {mask.sum()} investable firms")

# ── 9. SAVE OUTPUTS ───────────────────────────────────────────────────────────
returns_df.to_csv("data/clean_returns.csv", index=False)
monthly_returns.to_csv("data/clean_prices.csv", index=False)

import json
with open("data/investable_universe.json", "w") as f:
    json.dump(investable_universe, f, indent=2)

print("Cleaning complete.")
