"""
analysis/forecasting.py
───────────────────────
Uses linear regression to predict next month's revenue
based on load shedding hours and historical patterns.
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")


def load_data():
    path = os.path.join(DATA_DIR, "merged.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def run_forecast():
    df = load_data()

    # ── Prepare features ──────────────────────────────────────────────
    # Only use actual trading days (exclude public holidays)
    trading = df[(df["is_public_holiday"] == False) &
                 (df["revenue"] > 0)].copy()

    # Features: hours of load shedding, is_weekend, month number
    trading["is_weekend_int"] = trading["is_weekend"].astype(int)

    X = trading[["hours_affected", "is_weekend_int", "month_num"]].values
    y = trading["revenue"].values

    # ── Train model ───────────────────────────────────────────────────
    model = LinearRegression()
    model.fit(X, y)

    # ── Evaluate model ────────────────────────────────────────────────
    y_pred = model.predict(X)
    r2  = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print("── Linear Regression Model ─────────────────────────────────")
    print(f"   R² Score          : {r2:.3f}  (1.0 = perfect, 0 = no fit)")
    print(f"   Mean Abs Error    : R {mae:,.2f} per day")
    print(f"\n   Coefficients:")
    print(f"   Hours of LS       : R {model.coef_[0]:,.2f} per hour")
    print(f"   Weekend effect    : R {model.coef_[1]:,.2f}")
    print(f"   Month effect      : R {model.coef_[2]:,.2f} per month")
    print(f"   Base intercept    : R {model.intercept_:,.2f}")

    # ── Predict next month (April) scenarios ─────────────────────────
    print("\n── April 2024 Scenario Forecasts ───────────────────────────")
    scenarios = {
        "No Load Shedding (Stage 0)":     [0,  0, 4],
        "Low Load Shedding (Stage 2)":    [4,  0, 4],
        "Medium Load Shedding (Stage 4)": [8,  0, 4],
        "High Load Shedding (Stage 6)":   [12, 0, 4],
    }

    forecasts = {}
    for name, features in scenarios.items():
        pred = model.predict([features])[0]
        forecasts[name] = max(0, pred)
        print(f"   {name:<40} → R {max(0,pred):,.2f} / day")

    # ── Monthly revenue forecast ──────────────────────────────────────
    # Assume 22 trading weekdays in April
    print("\n── Estimated Monthly Revenue (22 trading days) ─────────────")
    for name, daily in forecasts.items():
        monthly = daily * 22
        print(f"   {name:<40} → R {monthly:,.2f}")

    return model, forecasts, r2, mae, trading


def stage_impact_summary(df):
    """Average revenue by load shedding stage."""
    trading = df[(df["is_public_holiday"] == False) &
                 (df["revenue"] > 0)].copy()

    summary = (
        trading.groupby("stage")["revenue"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    summary.columns = ["stage", "avg_revenue", "days", "std_dev"]
    summary["avg_revenue"] = summary["avg_revenue"].round(2)
    summary["std_dev"]     = summary["std_dev"].round(2)
    return summary


if __name__ == "__main__":
    model, forecasts, r2, mae, trading = run_forecast()
    df = load_data()
    print("\n── Average Revenue by Stage ────────────────────────────────")
    print(stage_impact_summary(df).to_string(index=False))
