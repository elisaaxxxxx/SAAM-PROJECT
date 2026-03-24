"""
STEP 1: CLEANING RETURNS DATA 
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
print(f"Zeros remaining in monthly RI: {(df_ri_m_clean[price_cols_m] == 0).sum().sum()}")
print(f"Zeros remaining in yearly RI:  {(df_ri_y_clean[year_cols_ri] == 0).sum().sum()}")

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
        has_yearly_price = pd.Series(True, index=df_ri_m_clean.index)

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

"""
STEP 2: CLEANING MARKET VALUE DATA 
Cleaning monthly and yearly MV per project guidelines
"""

import pandas as pd
import numpy as np

df_mv_m = pd.read_excel("data/Filtered_MV_T_USD_M_2025.xlsx", sheet_name=0)
df_mv_y = pd.read_excel("data/Filtered_MV_T_USD_Y_2025.xlsx", sheet_name=0)

mv_cols_m = df_mv_m.columns[2:]
mv_cols_m_dt = pd.to_datetime(mv_cols_m)
year_cols_mv = [c for c in df_mv_y.columns if isinstance(c, int)]

# ── Clean monthly MV ──────────────────────────────────────────────────────────
# Zeros mean not yet listed or delisted — replace with NaN first
df_mv_m_clean = df_mv_m.copy()
df_mv_m_clean[mv_cols_m] = df_mv_m_clean[mv_cols_m].replace(0, np.nan)

# Mark delistings (trailing NaNs → 0)
def mark_delisting(row):
    last_valid = row.last_valid_index()
    if last_valid is None:
        return row
    last_pos = row.index.get_loc(last_valid)
    if last_pos < len(row) - 1:
        row.iloc[last_pos + 1:] = 0
    return row

df_mv_m_clean[mv_cols_m] = df_mv_m_clean[mv_cols_m].apply(mark_delisting, axis=1)

# Forward-fill mid-sample gaps (NaNs between valid values only)
df_mv_m_clean[mv_cols_m] = df_mv_m_clean[mv_cols_m].ffill(axis=1)

# ── Clean yearly MV ───────────────────────────────────────────────────────────
df_mv_y_clean = df_mv_y.copy()
df_mv_y_clean[year_cols_mv] = df_mv_y_clean[year_cols_mv].replace(0, np.nan)
df_mv_y_clean[year_cols_mv] = df_mv_y_clean[year_cols_mv].apply(mark_delisting, axis=1)
df_mv_y_clean[year_cols_mv] = df_mv_y_clean[year_cols_mv].ffill(axis=1)

# ── Sanity checks ─────────────────────────────────────────────────────────────
print(f"Monthly MV zeros remaining:  {(df_mv_m_clean[mv_cols_m] == 0).sum().sum()}")
print(f"Monthly MV NaNs remaining:   {df_mv_m_clean[mv_cols_m].isna().sum().sum()}")
print(f"Yearly MV zeros remaining:   {(df_mv_y_clean[year_cols_mv] == 0).sum().sum()}")
print(f"Yearly MV NaNs remaining:    {df_mv_y_clean[year_cols_mv].isna().sum().sum()}")

# ── Save ──────────────────────────────────────────────────────────────────────
df_mv_m_clean.to_csv("data/clean_mv_monthly.csv", index=False)
df_mv_y_clean.to_csv("data/clean_mv_yearly.csv", index=False)
print("\nMV files saved.")

"""
STEP 3: CLEANING CARBON AND REVENUE DATA 
Cleaning CO2 and Revenue per project guidelines
"""

import pandas as pd
import numpy as np

df_co2 = pd.read_excel("data/Filtered_CO2_SCOPE_1_Y_2025.xlsx")
df_rev = pd.read_excel("data/Filtered_REV_Y_2025.xlsx")

year_cols_co2 = [c for c in df_co2.columns if isinstance(c, int)]
year_cols_rev = [c for c in df_rev.columns if isinstance(c, int)]

# ── Force numeric (handles scientific notation in revenue) ────────────────────
df_co2[year_cols_co2] = df_co2[year_cols_co2].apply(pd.to_numeric, errors='coerce')
df_rev[year_cols_rev] = df_rev[year_cols_rev].apply(pd.to_numeric, errors='coerce')

# ── CO2: replace zeros with NaN (not yet reporting) ──────────────────────────
df_co2[year_cols_co2] = df_co2[year_cols_co2].replace(0, np.nan)

# ── Revenue: replace zeros and negatives with NaN ────────────────────────────
df_rev[year_cols_rev] = df_rev[year_cols_rev].replace(0, np.nan)
df_rev[year_cols_rev] = df_rev[year_cols_rev].where(df_rev[year_cols_rev] > 0)

# ── Forward-fill mid-sample and trailing gaps (per spec) ─────────────────────
# Leading NaNs are preserved — can't invest until data appears
# Note: no delisting logic needed for carbon/revenue — these are reported
# annually and a trailing gap just means latest data not yet available,
# not that the firm ceased to exist
df_co2[year_cols_co2] = df_co2[year_cols_co2].ffill(axis=1)
df_rev[year_cols_rev] = df_rev[year_cols_rev].ffill(axis=1)

# ── Compute carbon intensity ──────────────────────────────────────────────────
# CI = CO2 (tonnes) / Revenue (thousands USD / 1000) = CO2 / (Rev / 1000)
# i.e. tonnes CO2 per million USD revenue
# Both dataframes are aligned by row (same 634 firms, same order) — verify:
assert (df_co2["ISIN"].values == df_rev["ISIN"].values).all(), \
    "ISIN order mismatch between CO2 and Revenue files"

ci_values = np.full((len(df_co2), len(year_cols_co2)), np.nan)
for i, year in enumerate(year_cols_co2):
    if year in year_cols_rev:
        co2 = df_co2[year].values.astype(float)
        rev = df_rev[year].values.astype(float) / 1000  # convert to millions
        with np.errstate(divide='ignore', invalid='ignore'):
            ci_values[:, i] = np.where(rev > 0, co2 / rev, np.nan)

df_ci = pd.DataFrame(ci_values, columns=year_cols_co2)
df_ci = pd.concat([df_co2[["NAME", "ISIN"]].reset_index(drop=True), df_ci], axis=1)

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("=== After cleaning ===")
print(f"CO2 NaNs:     {df_co2[year_cols_co2].isna().sum().sum()}")
print(f"Revenue NaNs: {df_rev[year_cols_rev].isna().sum().sum()}")
print(f"CI NaNs:      {df_ci[year_cols_co2].isna().sum().sum()}")
print(f"CI negatives: {(df_ci[year_cols_co2] < 0).sum().sum()}")

# Check how many firms have carbon data available by year
print("\n=== Firms with valid CI per year ===")
for year in year_cols_co2:
    valid = df_ci[year].notna().sum()
    print(f"  {year}: {valid} firms")

# ── Save ──────────────────────────────────────────────────────────────────────
df_co2.to_csv("data/clean_co2.csv", index=False)
df_rev.to_csv("data/clean_revenue.csv", index=False)
df_ci.to_csv("data/clean_carbon_intensity.csv", index=False)

print("\nCarbon files saved.")
