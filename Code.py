"""
STEP 1: CLEANING DATA 
Cleaning monthly and yearly RI per project guidelines
"""

import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD FILES
# ═══════════════════════════════════════════════════════════════════════════════
df_ri_m = pd.read_excel("data/Filtered_RI_T_USD_M_2025.xlsx")
df_ri_y = pd.read_excel("data/Filtered_RI_T_USD_Y_2025.xlsx")

print("Files loaded.")
print(f"  Monthly RI: {df_ri_m.shape}")
print(f"  Yearly RI:  {df_ri_y.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. IDENTIFY COLUMN TYPES
# ═══════════════════════════════════════════════════════════════════════════════
# Monthly RI: date columns are timestamp objects after NAME, ISIN
price_cols_m = df_ri_m.columns[2:]
price_cols_m_dt = pd.to_datetime(price_cols_m)

# Yearly RI: date columns are integers after NAME, ISIN
year_cols_ri = [c for c in df_ri_y.columns if isinstance(c, int)]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLEANING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def clean_price_series(df, value_cols, min_price=0.5):
    """
    Cleans a price dataframe according to project spec rules:
      - Nullify prices below min_price
      - Mark delistings (trailing NaNs → 0)
      - Forward-fill mid-sample gaps only
    """
    df = df.copy()

    # Step 1: nullify low prices
    if min_price > 0:
        df[value_cols] = df[value_cols].where(df[value_cols] >= min_price)

    # Step 2: mark delistings (trailing NaNs → 0)
    # Must come before ffill so zeros are not overwritten
    def mark_delisting(row):
        last_valid = row.last_valid_index()
        if last_valid is None:
            return row  # entirely missing — leave as NaN, exclude from universe
        last_pos = row.index.get_loc(last_valid)
        if last_pos < len(row) - 1:
            row.iloc[last_pos + 1:] = 0
        return row

    df[value_cols] = df[value_cols].apply(mark_delisting, axis=1)

    # Step 3: forward-fill mid-sample gaps only
    # ffill does not propagate zeros, only NaNs — so post-delisting zeros
    # are preserved correctly
    df[value_cols] = df[value_cols].ffill(axis=1)

    return df

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLEAN RI FILES
# ═══════════════════════════════════════════════════════════════════════════════
df_ri_m_clean = clean_price_series(df_ri_m, price_cols_m)
df_ri_y_clean = clean_price_series(df_ri_y, year_cols_ri)
print("Price series cleaned.")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. COMPUTE RETURNS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Monthly returns ──────────────────────────────────────────────────────────
prices_m = df_ri_m_clean[price_cols_m].values.astype(float)
returns_m = np.full_like(prices_m, np.nan)
with np.errstate(divide='ignore', invalid='ignore'):
    returns_m[:, 1:] = prices_m[:, 1:] / prices_m[:, :-1] - 1

returns_m_df = pd.DataFrame(returns_m, columns=price_cols_m)
returns_m_df = pd.concat([df_ri_m_clean[["NAME", "ISIN"]].reset_index(drop=True),
                           returns_m_df], axis=1)

# ── Annual returns ───────────────────────────────────────────────────────────
prices_y = df_ri_y_clean[year_cols_ri].values.astype(float)
returns_y = np.full_like(prices_y, np.nan)
with np.errstate(divide='ignore', invalid='ignore'):
    returns_y[:, 1:] = prices_y[:, 1:] / prices_y[:, :-1] - 1

return_year_cols = year_cols_ri[1:]
returns_y_df = pd.DataFrame(returns_y[:, 1:], columns=return_year_cols)
returns_y_df = pd.concat([df_ri_y_clean[["NAME", "ISIN"]].reset_index(drop=True),
                           returns_y_df], axis=1)

print("Returns computed.")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. BUILD INVESTABLE UNIVERSE PER YEAR
# ═══════════════════════════════════════════════════════════════════════════════
ret_only_m = returns_m_df[price_cols_m]

# Map each rebalancing year to its December monthly column
year_ends = {}
for year in range(2013, 2025):
    dec_mask = (price_cols_m_dt.month == 12) & (price_cols_m_dt.year == year)
    matches = price_cols_m[dec_mask]
    if len(matches) > 0:
        year_ends[year] = matches[0]
    else:
        print(f"Warning: no December column found for {year}")

date_cols_m = list(price_cols_m)
investable_universe = {}

for year, end_col in year_ends.items():
    end_idx = date_cols_m.index(end_col)
    window_start = max(0, end_idx - 119)
    window = ret_only_m.iloc[:, window_start:end_idx + 1]

    # Filter 1: valid monthly RI at year-end
    ri_m_at_year_end = df_ri_m_clean[end_col]
    has_monthly_price = ri_m_at_year_end.notna() & (ri_m_at_year_end > 0)

    # Filter 2: valid yearly RI at year-end
    ri_y_at_year_end = df_ri_y_clean[year] if year in year_cols_ri else None
    if ri_y_at_year_end is not None:
        has_yearly_price = ri_y_at_year_end.notna() & (ri_y_at_year_end > 0)
    else:
        has_yearly_price = pd.Series(True, index=df_ri_y_clean.index)

    # Filter 3: sufficient return history (≥36 months in trailing 120m window)
    valid_obs = window.notna().sum(axis=1)
    enough_history = valid_obs >= 36

    # Filter 4: stale price filter (≤50% zero returns in trailing 120m window)
    zero_proportion = (window == 0).sum(axis=1) / window.shape[1]
    not_stale = zero_proportion <= 0.5

    # Intersection of all filters
    # Match yearly RI filter to monthly df via ISIN
    monthly_isins = df_ri_m_clean["ISIN"]
    yearly_valid_isins = set(df_ri_y_clean.loc[has_yearly_price, "ISIN"].dropna())

    base_mask = has_monthly_price & enough_history & not_stale
    final_mask = base_mask & monthly_isins.isin(yearly_valid_isins)

    investable_universe[year] = df_ri_m_clean.loc[final_mask, "ISIN"].tolist()
    print(f"{year}: {final_mask.sum()} investable firms")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
returns_m_df.to_csv("data/clean_returns_monthly.csv", index=False)
returns_y_df.to_csv("data/clean_returns_yearly.csv", index=False)
df_ri_m_clean.to_csv("data/clean_prices_monthly.csv", index=False)
df_ri_y_clean.to_csv("data/clean_prices_yearly.csv", index=False)

import json
with open("data/investable_universe.json", "w") as f:
    json.dump(investable_universe, f, indent=2)

print("\nAll outputs saved.")
