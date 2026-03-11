"""
analysis/cleaning.py
────────────────────
Loads raw CSVs, cleans them, merges them into one
master DataFrame, and exports to data/merged.csv.
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")


def load_and_clean():
    # ── Load raw files ────────────────────────────────────────────────
    ls  = pd.read_csv(os.path.join(DATA_DIR, "loadshedding.csv"))
    rev = pd.read_csv(os.path.join(DATA_DIR, "revenue.csv"))

    # ── Parse dates ───────────────────────────────────────────────────
    ls["date"]  = pd.to_datetime(ls["date"])
    rev["date"] = pd.to_datetime(rev["date"])

    # ── Fill missing values ───────────────────────────────────────────
    # Days with no load shedding have blank start/end — fill with "None"
    ls["scheduled_start"] = ls["scheduled_start"].fillna("None")
    ls["scheduled_end"]   = ls["scheduled_end"].fillna("None")
    ls["stage"]           = ls["stage"].fillna(0).astype(int)
    ls["hours_affected"]  = ls["hours_affected"].fillna(0)

    # ── Derive useful columns ─────────────────────────────────────────
    ls["stage_category"] = ls["stage"].apply(categorise_stage)
    ls["is_loadshedding"] = ls["stage"] > 0

    rev["month"]      = rev["date"].dt.month_name()
    rev["month_num"]  = rev["date"].dt.month
    rev["week"]       = rev["date"].dt.to_period("W").astype(str)
    rev["day_name"]   = rev["date"].dt.day_name()

    # ── Merge on date ─────────────────────────────────────────────────
    merged = pd.merge(rev, ls[["date", "stage", "hours_affected",
                                "stage_category", "is_loadshedding"]],
                      on="date", how="left")

    # Fill any remaining NaN stages (public holidays not in LS data)
    merged["stage"]          = merged["stage"].fillna(0).astype(int)
    merged["hours_affected"] = merged["hours_affected"].fillna(0)
    merged["stage_category"] = merged["stage_category"].fillna("None")
    merged["is_loadshedding"]= merged["is_loadshedding"].fillna(False)

    # ── Baseline revenue ──────────────────────────────────────────────
    # Calculate baseline as average revenue on days with NO load shedding
    # and NO public holidays — this is our "normal" trading day
    normal_days = merged[(merged["stage"] == 0) &
                         (merged["is_public_holiday"] == False) &
                         (merged["revenue"] > 0)]
    baseline = normal_days["revenue"].mean()
    merged["baseline_revenue"] = baseline

    # Revenue loss = baseline - actual (only meaningful on trading days)
    merged["revenue_loss"] = merged.apply(
        lambda r: max(0, baseline - r["revenue"])
        if r["is_loadshedding"] and not r["is_public_holiday"]
        else 0,
        axis=1
    )

    # Percentage of baseline revenue achieved
    merged["pct_of_baseline"] = merged.apply(
        lambda r: (r["revenue"] / baseline) * 100
        if r["revenue"] > 0 and not r["is_public_holiday"]
        else None,
        axis=1
    )

    # ── Export cleaned data ───────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, "merged.csv")
    merged.to_csv(out_path, index=False)
    print(f"✅ Cleaned data saved to {out_path}")
    print(f"   Rows: {len(merged)}")
    print(f"   Date range: {merged['date'].min().date()} → {merged['date'].max().date()}")
    print(f"   Baseline daily revenue (no load shedding): R {baseline:,.2f}")
    print(f"   Load shedding days: {merged['is_loadshedding'].sum()}")
    print(f"   Public holidays: {merged['is_public_holiday'].sum()}")

    return merged, baseline


def categorise_stage(stage):
    if stage == 0:
        return "None"
    elif stage <= 2:
        return "Low (1-2)"
    elif stage <= 4:
        return "Medium (3-4)"
    else:
        return "High (5-6)"


if __name__ == "__main__":
    df, baseline = load_and_clean()
    print("\nSample of merged data:")
    print(df[["date", "revenue", "stage", "hours_affected",
              "stage_category", "revenue_loss"]].head(10).to_string())
