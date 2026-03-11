"""
analysis/anomaly_detection.py
──────────────────────────────
Flags days where revenue dropped more than 30% below
the rolling average — likely caused by load shedding.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")


def load_data():
    path = os.path.join(DATA_DIR, "merged.csv")
    return pd.read_csv(path, parse_dates=["date"])


def detect_anomalies(df, threshold=0.30):
    """
    Flag days where revenue is more than `threshold`
    percent below the 7-day rolling average.
    Only runs on actual trading days.
    """
    trading = df[(df["is_public_holiday"] == False) &
                 (df["revenue"] > 0)].copy()

    # 7-day rolling average revenue
    trading = trading.sort_values("date")
    trading["rolling_avg"] = (
        trading["revenue"]
        .rolling(window=7, min_periods=3)
        .mean()
        .shift(1)   # shift so we don't use the current day in its own average
    )

    # Drop rows where rolling avg couldn't be calculated
    trading = trading.dropna(subset=["rolling_avg"])

    # Flag anomalies
    trading["pct_below_avg"] = (
        (trading["rolling_avg"] - trading["revenue"]) /
        trading["rolling_avg"] * 100
    )

    trading["is_anomaly"] = trading["pct_below_avg"] > (threshold * 100)

    anomalies = trading[trading["is_anomaly"]].copy()
    anomalies = anomalies.sort_values("pct_below_avg", ascending=False)

    return trading, anomalies


def correlation_analysis(df):
    """
    Pearson correlation between load shedding hours
    and revenue on trading days.
    """
    trading = df[(df["is_public_holiday"] == False) &
                 (df["revenue"] > 0)].copy()

    corr = trading["hours_affected"].corr(trading["revenue"])
    stage_corr = trading["stage"].corr(trading["revenue"])

    print("── Correlation Analysis ────────────────────────────────────")
    print(f"   Hours of load shedding vs Revenue : {corr:.3f}")
    print(f"   Load shedding stage vs Revenue    : {stage_corr:.3f}")
    print()
    print("   Interpretation:")
    if corr < -0.5:
        print("   ✅ Strong NEGATIVE correlation — more load shedding hours")
        print("      clearly linked to lower revenue.")
    elif corr < -0.3:
        print("   ⚠️  Moderate negative correlation — load shedding has")
        print("      a noticeable but not dominant effect on revenue.")
    else:
        print("   ℹ️  Weak correlation — other factors also affect revenue.")

    return corr, stage_corr


def total_revenue_loss(df):
    """
    Estimate total revenue lost to load shedding
    across the 3-month period.
    """
    total_loss  = df["revenue_loss"].sum()
    days_affected = (df["is_loadshedding"] & (df["revenue"] > 0) &
                     (df["is_public_holiday"] == False)).sum()
    avg_loss_per_day = total_loss / days_affected if days_affected > 0 else 0

    print("── Revenue Loss Summary ────────────────────────────────────")
    print(f"   Total estimated revenue lost    : R {total_loss:,.2f}")
    print(f"   Load shedding trading days      : {days_affected}")
    print(f"   Average loss per affected day   : R {avg_loss_per_day:,.2f}")

    # By month
    monthly = (
        df.groupby("month_num")["revenue_loss"]
        .sum()
        .reset_index()
    )
    month_names = {1: "January", 2: "February", 3: "March"}
    monthly["month"] = monthly["month_num"].map(month_names)
    print("\n   Loss by Month:")
    for _, row in monthly.iterrows():
        print(f"   {row['month']:<12} : R {row['revenue_loss']:,.2f}")

    return total_loss, days_affected, avg_loss_per_day


if __name__ == "__main__":
    df = load_data()
    trading_df, anomalies = detect_anomalies(df)

    print(f"── Anomaly Detection (30% below rolling average) ───────────")
    print(f"   Anomalous days detected: {len(anomalies)}")
    print()

    if len(anomalies) > 0:
        print("   Top 10 worst days:")
        cols = ["date", "revenue", "stage", "hours_affected", "pct_below_avg"]
        top10 = anomalies[cols].head(10).copy()
        top10["date"] = top10["date"].dt.strftime("%Y-%m-%d")
        top10["revenue"] = top10["revenue"].map(lambda x: f"R {x:,.0f}")
        top10["pct_below_avg"] = top10["pct_below_avg"].map(lambda x: f"{x:.1f}%")
        print(top10.to_string(index=False))
        print()

    corr, stage_corr = correlation_analysis(df)
    print()
    total_revenue_loss(df)
