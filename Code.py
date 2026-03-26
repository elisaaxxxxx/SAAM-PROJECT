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

"""
STEP 4: BUILD FINAL INVESTABLE UNIVERSE
Combines all cleaned data sources and applies all exclusion criteria
to produce the final per-year investable universe.
"""
import pandas as pd
import numpy as np
import json

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL CLEANED FILES
# ═══════════════════════════════════════════════════════════════════════════════
returns_m_df  = pd.read_csv("data/cleaned data/clean_returns_monthly.csv")
prices_m_df   = pd.read_csv("data/cleaned data/clean_prices_monthly.csv")
prices_y_df   = pd.read_csv("data/cleaned data/clean_prices_yearly.csv")
mv_y_df       = pd.read_csv("data/cleaned data/clean_mv_yearly.csv")
ci_df         = pd.read_csv("data/cleaned data/clean_carbon_intensity.csv")

print("Files loaded.")
print(f"  Monthly returns: {returns_m_df.shape}")
print(f"  Monthly prices:  {prices_m_df.shape}")
print(f"  Yearly prices:   {prices_y_df.shape}")
print(f"  Yearly MV:       {mv_y_df.shape}")
print(f"  Carbon intensity:{ci_df.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. IDENTIFY COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════
# Monthly: timestamp columns after NAME, ISIN
ret_cols_m   = [c for c in returns_m_df.columns  if c not in ["NAME", "ISIN"]]
price_cols_m = [c for c in prices_m_df.columns   if c not in ["NAME", "ISIN"]]

# Yearly: integer columns after NAME, ISIN
year_cols_py = [c for c in prices_y_df.columns   if c not in ["NAME", "ISIN"]]
year_cols_mv = [c for c in mv_y_df.columns       if c not in ["NAME", "ISIN"]]
year_cols_ci = [c for c in ci_df.columns         if c not in ["NAME", "ISIN"]]

# Convert yearly cols to int for consistent comparison
year_cols_py = [int(c) for c in year_cols_py]
year_cols_mv = [int(c) for c in year_cols_mv]
year_cols_ci = [int(c) for c in year_cols_ci]

# Convert monthly columns to datetime for December lookup
price_cols_m_dt = pd.to_datetime(price_cols_m)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MAP REBALANCING YEARS TO DECEMBER MONTHLY COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════
year_ends = {}
for year in range(2013, 2025):
    dec_mask = (price_cols_m_dt.month == 12) & (price_cols_m_dt.year == year)
    matches = [c for c, m in zip(price_cols_m, dec_mask) if m]
    if matches:
        year_ends[year] = matches[0]
    else:
        print(f"Warning: no December column found for {year}")

print(f"\nRebalancing years mapped: {list(year_ends.keys())}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. BUILD UNIVERSE YEAR BY YEAR
# ═══════════════════════════════════════════════════════════════════════════════
all_isins = returns_m_df["ISIN"].dropna().tolist()
ret_matrix = returns_m_df[ret_cols_m].values.astype(float)

investable_universe = {}
exclusion_log = {}

for year, end_col in year_ends.items():

    # ── Monthly window (trailing 120 months) ─────────────────────────────────
    end_idx = price_cols_m.index(end_col)
    window_start = max(0, end_idx - 119)
    window = ret_matrix[:, window_start:end_idx + 1]

    # ── Filter 1: valid monthly price at year-end ─────────────────────────────
    mp_at_end = prices_m_df[end_col].values.astype(float)
    f1_monthly_price = (mp_at_end > 0) & (~np.isnan(mp_at_end))

    # ── Filter 2: valid yearly RI at year-end ─────────────────────────────────
    if year in year_cols_py:
        yp_at_end = prices_y_df[str(year)].values.astype(float)
        f2_yearly_price = (yp_at_end > 0) & (~np.isnan(yp_at_end))
    else:
        f2_yearly_price = np.ones(len(all_isins), dtype=bool)

    # ── Filter 3: sufficient return history (≥36 months in window) ───────────
    valid_obs = (~np.isnan(window)).sum(axis=1)
    f3_history = valid_obs >= 36

    # ── Filter 4: stale price filter (≤50% zero returns in window) ───────────
    zero_prop = (window == 0).sum(axis=1) / window.shape[1]
    f4_not_stale = zero_prop <= 0.5

    # ── Filter 5: valid MV at year-end ────────────────────────────────────────
    if year in year_cols_mv:
        mv_at_end = mv_y_df[str(year)].values.astype(float)
        f5_mv = (mv_at_end > 0) & (~np.isnan(mv_at_end))
    else:
        f5_mv = np.ones(len(all_isins), dtype=bool)

    # ── Filter 6: valid carbon intensity at year-end ──────────────────────────
    if year in year_cols_ci:
        ci_at_end = ci_df[str(year)].values.astype(float)
        f6_carbon = (~np.isnan(ci_at_end)) & (ci_at_end > 0)
    else:
        f6_carbon = np.zeros(len(all_isins), dtype=bool)

    # ── Combine all filters ───────────────────────────────────────────────────
    final_mask = f1_monthly_price & f2_yearly_price & f3_history & \
                 f4_not_stale & f5_mv & f6_carbon

    investable_isins = [isin for isin, m in zip(all_isins, final_mask) if m]
    investable_universe[year] = investable_isins

    # ── Exclusion log ─────────────────────────────────────────────────────────
    n = len(all_isins)
    exclusion_log[year] = {
        "start":               n,
        "fail_monthly_price":  int((~f1_monthly_price).sum()),
        "fail_yearly_price":   int((~f2_yearly_price).sum()),
        "fail_history":        int((~f3_history).sum()),
        "fail_stale":          int((~f4_not_stale).sum()),
        "fail_mv":             int((~f5_mv).sum()),
        "fail_carbon":         int((~f6_carbon).sum()),
        "final":               int(final_mask.sum())
    }

    print(f"{year}: {final_mask.sum():3d} investable firms  "
          f"(price:{(~f1_monthly_price).sum()} "
          f"yprice:{(~f2_yearly_price).sum()} "
          f"hist:{(~f3_history).sum()} "
          f"stale:{(~f4_not_stale).sum()} "
          f"mv:{(~f5_mv).sum()} "
          f"carbon:{(~f6_carbon).sum()} excluded)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
# Save universe as JSON
with open("data/investable_universe.json", "w") as f:
    json.dump(investable_universe, f, indent=2)

# Save exclusion log as CSV for report
exclusion_df = pd.DataFrame(exclusion_log).T
exclusion_df.index.name = "year"
exclusion_df.to_csv("data/exclusion_log.csv")

# Save a flat CSV of all investable firms per year (useful for inspection)
rows = []
for year, isins in investable_universe.items():
    firm_names = returns_m_df.set_index("ISIN")["NAME"]
    for isin in isins:
        rows.append({
            "year": year,
            "ISIN": isin,
            "NAME": firm_names.get(isin, "")
        })
universe_flat_df = pd.DataFrame(rows)
universe_flat_df.to_csv("data/investable_universe_flat.csv", index=False)

print("\n=== Exclusion log ===")
print(exclusion_df.to_string())
print("\nAll outputs saved.")
print(f"Files created:")
print(f"  data/investable_universe.json      — per-year ISIN lists")
print(f"  data/exclusion_log.csv             — filter breakdown per year")
print(f"  data/investable_universe_flat.csv  — flat list for inspection")

"""
STEP 5: EXCLUSION AUDIT
Produces a clear summary of which firms were excluded at each cleaning step
and a final master file showing inclusion/exclusion status per firm per year.
"""
import pandas as pd
import numpy as np
import json

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL CLEANED FILES
# ═══════════════════════════════════════════════════════════════════════════════
returns_m_df = pd.read_csv("data/cleaned data/clean_returns_monthly.csv")
prices_m_df  = pd.read_csv("data/cleaned data/clean_prices_monthly.csv")
prices_y_df  = pd.read_csv("data/cleaned data/clean_prices_yearly.csv")
mv_y_df      = pd.read_csv("data/cleaned data/clean_mv_yearly.csv")
ci_df        = pd.read_csv("data/cleaned data/clean_carbon_intensity.csv")

with open("data/investable_set.json") as f:
    investable_set = json.load(f)

# Master firm list
firms = returns_m_df[["NAME", "ISIN"]].copy()
all_isins = firms["ISIN"].tolist()

# Column identification
price_cols_m    = [c for c in prices_m_df.columns  if c not in ["NAME", "ISIN"]]
ret_cols_m      = [c for c in returns_m_df.columns if c not in ["NAME", "ISIN"]]
year_cols_py    = [int(c) for c in prices_y_df.columns if c not in ["NAME", "ISIN"]]
year_cols_mv    = [int(c) for c in mv_y_df.columns    if c not in ["NAME", "ISIN"]]
year_cols_ci    = [int(c) for c in ci_df.columns      if c not in ["NAME", "ISIN"]]
price_cols_m_dt = pd.to_datetime(price_cols_m)
ret_matrix      = returns_m_df[ret_cols_m].values.astype(float)

# Map rebalancing years to December columns
year_ends = {}
for year in range(2013, 2025):
    dec_mask = (price_cols_m_dt.month == 12) & (price_cols_m_dt.year == year)
    matches = [c for c, m in zip(price_cols_m, dec_mask) if m]
    if matches:
        year_ends[year] = matches[0]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE FILTER STATUS FOR EVERY FIRM × YEAR
# ═══════════════════════════════════════════════════════════════════════════════
records = []

for year, end_col in year_ends.items():

    end_idx      = price_cols_m.index(end_col)
    window_start = max(0, end_idx - 119)
    window       = ret_matrix[:, window_start:end_idx + 1]

    # Each filter as boolean array (True = passes)
    mp_at_end  = prices_m_df[end_col].values.astype(float)
    f1 = (mp_at_end > 0) & (~np.isnan(mp_at_end))

    yp_at_end  = prices_y_df[str(year)].values.astype(float) \
                 if year in year_cols_py else np.ones(len(all_isins), dtype=bool)
    f2 = (yp_at_end > 0) & (~np.isnan(yp_at_end))

    valid_obs  = (~np.isnan(window)).sum(axis=1)
    f3 = valid_obs >= 36

    zero_prop  = (window == 0).sum(axis=1) / window.shape[1]
    f4 = zero_prop <= 0.5

    mv_at_end  = mv_y_df[str(year)].values.astype(float) \
                 if year in year_cols_mv else np.ones(len(all_isins), dtype=bool)
    f5 = (mv_at_end > 0) & (~np.isnan(mv_at_end))

    ci_at_end  = ci_df[str(year)].values.astype(float) \
                 if year in year_cols_ci else np.zeros(len(all_isins), dtype=bool)
    f6 = (~np.isnan(ci_at_end)) & (ci_at_end > 0)

    final = f1 & f2 & f3 & f4 & f5 & f6

    for i, isin in enumerate(all_isins):
        # Determine exclusion reason (first failing filter wins)
        if final[i]:
            reason = "included"
        elif not f1[i]:
            reason = "no monthly price"
        elif not f2[i]:
            reason = "no yearly price"
        elif not f3[i]:
            reason = "insufficient history"
        elif not f4[i]:
            reason = "stale prices"
        elif not f5[i]:
            reason = "no market value"
        elif not f6[i]:
            reason = "no carbon data"
        else:
            reason = "unknown"

        records.append({
            "year":       year,
            "ISIN":       isin,
            "NAME":       firms.loc[i, "NAME"],
            "f1_monthly_price":  bool(f1[i]),
            "f2_yearly_price":   bool(f2[i]),
            "f3_history":        bool(f3[i]),
            "f4_not_stale":      bool(f4[i]),
            "f5_mv":             bool(f5[i]),
            "f6_carbon":         bool(f6[i]),
            "investable":        bool(final[i]),
            "exclusion_reason":  reason
        })

audit_df = pd.DataFrame(records)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SUMMARY: FIRMS NEVER INVESTABLE IN ANY YEAR
# ═══════════════════════════════════════════════════════════════════════════════
ever_investable = audit_df.groupby("ISIN")["investable"].any()
never_investable_isins = ever_investable[~ever_investable].index.tolist()
never_investable_df = firms[firms["ISIN"].isin(never_investable_isins)].copy()

# Why was each never-investable firm excluded (most common reason)
never_reasons = audit_df[audit_df["ISIN"].isin(never_investable_isins)] \
    .groupby("ISIN")["exclusion_reason"].agg(lambda x: x.value_counts().index[0])
never_investable_df["primary_reason"] = never_investable_df["ISIN"].map(never_reasons)

print(f"=== Firms NEVER investable in any year: {len(never_investable_df)} ===")
print(never_investable_df.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SUMMARY: FIRMS INVESTABLE IN SOME BUT NOT ALL YEARS
# ═══════════════════════════════════════════════════════════════════════════════
years_investable = audit_df.groupby("ISIN")["investable"].sum().astype(int)
sometimes_df = firms[
    firms["ISIN"].isin(years_investable[
        (years_investable > 0) & (years_investable < 11)
    ].index)
].copy()
sometimes_df["years_investable"] = sometimes_df["ISIN"].map(years_investable)

print(f"\n=== Firms investable in SOME years only: {len(sometimes_df)} ===")
print(sometimes_df.sort_values("years_investable").to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 5. PIVOT: ONE ROW PER FIRM, ONE COLUMN PER YEAR
# ═══════════════════════════════════════════════════════════════════════════════
pivot_investable = audit_df.pivot(
    index="ISIN", columns="year", values="investable"
).astype(int)  # 1 = investable, 0 = excluded

pivot_reason = audit_df.pivot(
    index="ISIN", columns="year", values="exclusion_reason"
)

# Merge with firm names
pivot_investable = firms.set_index("ISIN")[["NAME"]].join(pivot_investable)
pivot_reason     = firms.set_index("ISIN")[["NAME"]].join(pivot_reason)

# Add summary columns
pivot_investable["total_years_investable"] = \
    pivot_investable[[y for y in range(2013, 2025)]].sum(axis=1)
pivot_investable["ever_investable"] = \
    pivot_investable["total_years_investable"] > 0

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
# Full audit log (long format — one row per firm per year)
audit_df.to_csv("data/exclusion_audit_long.csv", index=False)

# Pivot: 1/0 investable per firm per year
pivot_investable.to_csv("data/exclusion_audit_pivot.csv")

# Pivot: exclusion reason per firm per year
pivot_reason.to_csv("data/exclusion_audit_reasons.csv")

# Never investable firms
never_investable_df.to_csv("data/firms_never_investable.csv", index=False)

print(f"\n=== Per-year investable counts ===")
for year in range(2013, 2025):
    n = audit_df[audit_df["year"] == year]["investable"].sum()
    print(f"  {year}: {n} firms")

print("\nFiles saved:")
print("  data/exclusion_audit_long.csv      — full filter status per firm per year")
print("  data/exclusion_audit_pivot.csv     — 1/0 investable matrix")
print("  data/exclusion_audit_reasons.csv   — exclusion reason matrix")
print("  data/firms_never_investable.csv    — firms excluded in every year")
