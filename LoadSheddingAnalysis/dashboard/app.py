"""
dashboard/app.py
────────────────
Load Shedding Impact Dashboard
Multi-page Dash app — neon orange and black theme

Pages:
  /            → Home
  /dashboard   → Full analytics view
  /forecast    → April 2024 forecast scenarios

Run: python app.py
Open: http://127.0.0.1:8050
"""

import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
TODAY    = date.today().isoformat()

# ── LOAD DATA ─────────────────────────────────────────────────────────────

def load_data():
    merged_path = os.path.join(DATA_DIR, "merged.csv")
    if not os.path.exists(merged_path):
        print("⚠️  merged.csv not found. Running cleaning script first...")
        sys.path.append(os.path.join(BASE_DIR, "../analysis"))
        from cleaning import load_and_clean
        df, _ = load_and_clean()
    else:
        df = pd.read_csv(merged_path, parse_dates=["date"])
    return df

df = load_data()

# ── ANALYSIS ──────────────────────────────────────────────────────────────

trading = df[(df["is_public_holiday"] == False) & (df["revenue"] > 0)].copy()
baseline = df["baseline_revenue"].iloc[0]

# Correlation
corr_hours = trading["hours_affected"].corr(trading["revenue"])
corr_stage = trading["stage"].corr(trading["revenue"])

# Linear Regression
trading["is_weekend_int"] = trading["is_weekend"].astype(int)
X = trading[["hours_affected", "is_weekend_int", "month_num"]].values
y = trading["revenue"].values
model = LinearRegression()
model.fit(X, y)
y_pred    = model.predict(X)
r2        = r2_score(y, y_pred)
mae       = mean_absolute_error(y, y_pred)
ls_cost   = abs(model.coef_[0])

# KPI numbers
total_revenue   = trading["revenue"].sum()
total_loss      = df["revenue_loss"].sum()
ls_days         = int((df["is_loadshedding"] & (df["revenue"] > 0) &
                       (df["is_public_holiday"] == False)).sum())
avg_loss_day    = total_loss / ls_days if ls_days > 0 else 0
stage6_avg      = trading[trading["stage"] == 6]["revenue"].mean() if len(trading[trading["stage"] == 6]) > 0 else 0
stage0_avg      = trading[trading["stage"] == 0]["revenue"].mean()
stage6_drop_pct = ((stage0_avg - stage6_avg) / stage0_avg * 100) if stage0_avg > 0 else 0

# ── CHARTS ────────────────────────────────────────────────────────────────

STAGE_COLORS = {
    "None":         "#2ecc71",
    "Low (1-2)":    "#f39c12",
    "Medium (3-4)": "#e67e22",
    "High (5-6)":   "#e74c3c"
}

# 1. Daily revenue scatter coloured by stage
fig_daily = px.scatter(
    df[df["revenue"] > 0], x="date", y="revenue",
    color="stage_category",
    title="Daily Revenue by Load Shedding Stage",
    labels={"revenue": "Revenue (R)", "date": "Date", "stage_category": "Stage"},
    color_discrete_map=STAGE_COLORS,
    template="plotly_dark"
)
fig_daily.add_hline(y=baseline, line_dash="dash", line_color="white",
                    annotation_text=f"Baseline R {baseline:,.0f}")

# 2. Average revenue by stage bar
stage_rev = trading.groupby("stage")["revenue"].mean().reset_index()
fig_stage = px.bar(
    stage_rev, x="stage", y="revenue",
    title="Average Daily Revenue by Load Shedding Stage",
    labels={"revenue": "Avg Revenue (R)", "stage": "Stage"},
    color="revenue", color_continuous_scale="RdYlGn",
    template="plotly_dark"
)
fig_stage.add_hline(y=baseline, line_dash="dash", line_color="white",
                    annotation_text="Baseline")

# 3. Monthly revenue vs loss
month_order = ["January", "February", "March"]
monthly = df[df["revenue"] > 0].groupby("month").agg(
    total_revenue=("revenue", "sum"),
    total_loss=("revenue_loss", "sum"),
    ls_days=("is_loadshedding", "sum")
).reset_index()
monthly["month"] = pd.Categorical(monthly["month"], categories=month_order, ordered=True)
monthly = monthly.sort_values("month")

fig_monthly = px.bar(
    monthly, x="month", y=["total_revenue", "total_loss"],
    title="Monthly Revenue vs Revenue Lost to Load Shedding",
    labels={"value": "Amount (R)", "variable": ""},
    barmode="group",
    color_discrete_sequence=["#3498db", "#e74c3c"],
    template="plotly_dark"
)

# 4. Scatter: hours vs revenue with trendline
fig_scatter = px.scatter(
    trading, x="hours_affected", y="revenue",
    color="stage_category",
    trendline="ols",
    title=f"Hours of Load Shedding vs Revenue  (Correlation: {corr_hours:.2f})",
    labels={"hours_affected": "Hours Affected", "revenue": "Revenue (R)"},
    color_discrete_map=STAGE_COLORS,
    template="plotly_dark"
)

# 5. Revenue box by stage category
stage_order = ["None", "Low (1-2)", "Medium (3-4)", "High (5-6)"]
fig_box = px.box(
    trading, x="stage_category", y="revenue",
    category_orders={"stage_category": stage_order},
    title="Revenue Distribution by Load Shedding Category",
    labels={"revenue": "Revenue (R)", "stage_category": "Stage Category"},
    color="stage_category", color_discrete_map=STAGE_COLORS,
    template="plotly_dark"
)

# 6. Weekly revenue trend
weekly = df[df["revenue"] > 0].groupby("week").agg(
    revenue=("revenue", "sum"),
    loss=("revenue_loss", "sum")
).reset_index()
fig_weekly = px.line(
    weekly, x="week", y=["revenue", "loss"],
    title="Weekly Revenue vs Revenue Lost",
    labels={"value": "R", "week": "Week", "variable": ""},
    template="plotly_dark",
    color_discrete_sequence=["#3498db", "#e74c3c"]
)

# 7. Anomaly chart
trading_sorted = trading.copy().sort_values("date")
trading_sorted["rolling_avg"] = (
    trading_sorted["revenue"].rolling(7, min_periods=3).mean().shift(1)
)
trading_sorted = trading_sorted.dropna(subset=["rolling_avg"])
trading_sorted["pct_below"] = (
    (trading_sorted["rolling_avg"] - trading_sorted["revenue"]) /
    trading_sorted["rolling_avg"] * 100
)
anomalies = trading_sorted[trading_sorted["pct_below"] > 30]

fig_anomaly = go.Figure()
fig_anomaly.add_trace(go.Scatter(
    x=trading_sorted["date"], y=trading_sorted["revenue"],
    mode="lines+markers", name="Revenue", line=dict(color="#3498db")
))
fig_anomaly.add_trace(go.Scatter(
    x=anomalies["date"], y=anomalies["revenue"],
    mode="markers", name="Anomaly (30%+ drop)",
    marker=dict(color="red", size=12, symbol="x")
))
fig_anomaly.add_hline(y=baseline, line_dash="dash", line_color="#2ecc71",
                      annotation_text=f"Baseline R {baseline:,.0f}")
fig_anomaly.update_layout(
    title="Revenue Anomaly Detection",
    xaxis_title="Date", yaxis_title="Revenue (R)",
    template="plotly_dark"
)

# 8. Forecast chart
scenarios = {
    "No LS (Stage 0)":    [0,  0, 4],
    "Low (Stage 2)":      [4,  0, 4],
    "Medium (Stage 4)":   [8,  0, 4],
    "High (Stage 6)":     [12, 0, 4],
}
forecast_results = []
for name, features in scenarios.items():
    daily = max(0, model.predict([features])[0])
    forecast_results.append({"Scenario": name, "Daily Revenue": daily, "Monthly": daily * 22})

fdf = pd.DataFrame(forecast_results)
fig_forecast = px.bar(
    fdf, x="Scenario", y="Daily Revenue",
    title="April 2024 Revenue Forecast by Scenario",
    labels={"Daily Revenue": "Predicted Daily Revenue (R)"},
    color="Daily Revenue", color_continuous_scale="RdYlGn",
    template="plotly_dark"
)
fig_forecast.add_hline(y=baseline, line_dash="dash", line_color="white",
                       annotation_text=f"Baseline R {baseline:,.0f}")

# ── THEME ─────────────────────────────────────────────────────────────────

BG       = "#0a0a0a"
BG_CARD  = "#111111"
BG_CHART = "#1a1a1a"
ACCENT   = "#ff6b00"
TEXT     = "#ffffff"
MUTED    = "#888888"

CARD = {
    "background": BG_CARD, "borderRadius": "12px",
    "padding": "18px 22px", "textAlign": "center",
    "flex": "1", "margin": "6px", "minWidth": "130px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.5)",
    "border": f"1px solid #222"
}
CHART_CARD = {
    "background": BG_CHART, "borderRadius": "12px",
    "padding": "10px", "flex": "1", "minWidth": "320px", "margin": "8px"
}
NAV_LINK = {
    "color": MUTED, "textDecoration": "none",
    "padding": "8px 18px", "borderRadius": "8px",
    "fontSize": "14px", "fontWeight": "500"
}
NAV_ACTIVE = {**NAV_LINK, "color": TEXT, "background": ACCENT}

# ── COMPONENTS ────────────────────────────────────────────────────────────

def kpi(label, value, color=TEXT):
    return html.Div([
        html.P(label, style={"color": MUTED, "margin": "0 0 4px 0",
               "fontSize": "10px", "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.H3(value, style={"color": color, "margin": "0", "fontSize": "20px", "fontWeight": "700"}),
    ], style=CARD)


def navbar(current):
    pages = [("/", "🏠 Home"), ("/dashboard", "📊 Dashboard"), ("/forecast", "🔮 Forecast")]
    links = [html.A(label, href=path, style=NAV_ACTIVE if current == path else NAV_LINK)
             for path, label in pages]
    return html.Div(style={
        "background": BG_CARD, "padding": "0 28px",
        "display": "flex", "alignItems": "center",
        "justifyContent": "space-between", "height": "58px",
        "boxShadow": "0 2px 12px rgba(0,0,0,0.6)",
        "position": "sticky", "top": "0", "zIndex": "1000"
    }, children=[
        html.Div([
            html.Span("⚡", style={"fontSize": "22px", "marginRight": "10px"}),
            html.Span("Load Shedding Impact", style={"color": TEXT, "fontWeight": "700", "fontSize": "16px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(links, style={"display": "flex", "gap": "6px", "alignItems": "center"}),
    ])


def feature_card(icon, title, desc):
    return html.Div([
        html.Div(icon, style={"fontSize": "28px", "marginBottom": "10px"}),
        html.H3(title, style={"color": TEXT, "margin": "0 0 8px 0", "fontSize": "15px"}),
        html.P(desc, style={"color": MUTED, "margin": "0", "fontSize": "13px", "lineHeight": "1.6"}),
    ], style={**CHART_CARD, "minWidth": "200px"})


# ── PAGES ─────────────────────────────────────────────────────────────────

def page_home():
    return html.Div([
        navbar("/"),
        html.Div(style={"maxWidth": "860px", "margin": "60px auto", "padding": "0 24px"}, children=[
            html.Div(style={"textAlign": "center", "marginBottom": "52px"}, children=[
                html.Div("⚡", style={"fontSize": "72px"}),
                html.H1("Load Shedding Impact Analysis",
                        style={"color": TEXT, "fontSize": "32px", "margin": "12px 0", "fontWeight": "800"}),
                html.P("A data analytics project measuring how Eskom load shedding "
                       "affected the daily revenue of a Johannesburg retail shop "
                       "across January, February, and March 2024.",
                       style={"color": MUTED, "fontSize": "16px", "lineHeight": "1.7",
                              "maxWidth": "600px", "margin": "0 auto"}),
                html.Div(style={"marginTop": "20px", "display": "flex",
                                "gap": "12px", "justifyContent": "center"}, children=[
                    html.A("View Dashboard →", href="/dashboard", style={
                        **NAV_ACTIVE, "fontSize": "15px", "padding": "10px 28px"
                    }),
                    html.A("View Forecast →", href="/forecast", style={
                        **NAV_LINK, "fontSize": "15px", "padding": "10px 28px",
                        "border": f"1px solid {ACCENT}"
                    }),
                ]),
            ]),
            html.H2("What This Project Analyses", style={"color": TEXT, "fontSize": "20px", "marginBottom": "16px"}),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "14px", "marginBottom": "48px"}, children=[
                feature_card("📉", "Revenue vs Stage",
                    "How each load shedding stage (1–6) directly affected average daily revenue."),
                feature_card("🔢", "Correlation Analysis",
                    "Pearson correlation coefficient measuring the statistical strength of the relationship."),
                feature_card("🤖", "Linear Regression",
                    "Machine learning model predicting revenue from load shedding hours. Includes R² score."),
                feature_card("🚨", "Anomaly Detection",
                    "Rolling average method flagging days where revenue dropped 30%+ below normal."),
                feature_card("🔮", "Scenario Forecasting",
                    "Predicted revenue for April 2024 under Stage 0, 2, 4, and 6 scenarios."),
                feature_card("💸", "Financial Loss",
                    "Total estimated rand value lost to load shedding across the 3-month period."),
            ]),
            html.H2("Tech Stack", style={"color": TEXT, "fontSize": "20px", "marginBottom": "16px"}),
            html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "marginBottom": "48px"},
                     children=[
                html.Span(t, style={"background": ACCENT, "color": TEXT, "padding": "6px 16px",
                                    "borderRadius": "20px", "fontSize": "13px", "fontWeight": "600"})
                for t in ["Python", "pandas", "scikit-learn", "Plotly", "Dash", "Jupyter Notebooks"]
            ]),
            html.P(f"Data period: Jan–Mar 2024  •  Built: {TODAY}",
                   style={"color": MUTED, "textAlign": "center", "fontSize": "12px"}),
        ])
    ])


def page_dashboard():
    return html.Div([
        navbar("/dashboard"),
        html.Div(style={"padding": "24px 28px", "background": BG, "minHeight": "100vh"}, children=[
            html.H2("📊 Impact Dashboard", style={"color": TEXT, "margin": "0 0 4px 0"}),
            html.P("Jan–Mar 2024  •  Johannesburg Retail Shop  •  All values in ZAR",
                   style={"color": MUTED, "margin": "0 0 22px 0", "fontSize": "12px"}),

            # KPI cards
            html.Div(style={"display": "flex", "flexWrap": "wrap", "marginBottom": "20px"}, children=[
                kpi("Total Revenue (3 months)",    f"R {total_revenue:,.0f}",    "#3498db"),
                kpi("Total Revenue Lost",          f"R {total_loss:,.0f}",       "#e74c3c"),
                kpi("Load Shedding Days",          str(ls_days),                  TEXT),
                kpi("Avg Loss per LS Day",         f"R {avg_loss_day:,.0f}",     "#e67e22"),
                kpi("Stage 6 Revenue Drop",        f"{stage6_drop_pct:.1f}%",    "#e74c3c"),
                kpi("Baseline Daily Revenue",      f"R {baseline:,.0f}",         "#2ecc71"),
                kpi("LS Hours Cost",               f"R {ls_cost:,.0f}/hr",       "#f39c12"),
            ]),

            # Row 1
            html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                html.Div(dcc.Graph(figure=fig_daily, config={"displayModeBar": False}),
                         style={**CHART_CARD, "flex": "2"}),
                html.Div(dcc.Graph(figure=fig_stage, config={"displayModeBar": False}),
                         style={**CHART_CARD, "flex": "1"}),
            ]),
            # Row 2
            html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                html.Div(dcc.Graph(figure=fig_monthly, config={"displayModeBar": False}),
                         style={**CHART_CARD, "flex": "1"}),
                html.Div(dcc.Graph(figure=fig_box, config={"displayModeBar": False}),
                         style={**CHART_CARD, "flex": "1"}),
            ]),
            # Row 3
            html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                html.Div(dcc.Graph(figure=fig_scatter, config={"displayModeBar": False}),
                         style={**CHART_CARD, "flex": "1"}),
                html.Div(dcc.Graph(figure=fig_weekly, config={"displayModeBar": False}),
                         style={**CHART_CARD, "flex": "1"}),
            ]),
            # Anomaly
            html.Div(dcc.Graph(figure=fig_anomaly, config={"displayModeBar": False}),
                     style={**CHART_CARD}),
        ])
    ])


def page_forecast():
    # Findings table rows
    stage_rev_data = trading.groupby("stage")["revenue"].mean().reset_index()
    table_rows = []
    for _, row in stage_rev_data.iterrows():
        drop = ((stage0_avg - row["revenue"]) / stage0_avg * 100)
        table_rows.append(html.Tr([
            html.Td(f"Stage {int(row['stage'])}",
                    style={"padding": "10px 14px", "color": TEXT, "fontWeight": "600"}),
            html.Td(f"R {row['revenue']:,.0f}",
                    style={"padding": "10px 14px", "color": "#2ecc71"}),
            html.Td(f"{drop:+.1f}%",
                    style={"padding": "10px 14px",
                           "color": "#2ecc71" if drop >= 0 else "#e74c3c",
                           "fontWeight": "700"}),
        ], style={"borderBottom": "1px solid #222"}))

    TH = {"padding": "10px 14px", "color": MUTED, "fontSize": "11px",
          "textTransform": "uppercase", "borderBottom": f"2px solid {ACCENT}"}

    return html.Div([
        navbar("/forecast"),
        html.Div(style={"padding": "24px 28px", "background": BG, "minHeight": "100vh"}, children=[
            html.H2("🔮 Forecast & Findings", style={"color": TEXT, "margin": "0 0 4px 0"}),
            html.P("Linear regression model trained on Jan–Mar 2024 data",
                   style={"color": MUTED, "margin": "0 0 22px 0", "fontSize": "12px"}),

            # Model stats cards
            html.Div(style={"display": "flex", "flexWrap": "wrap", "marginBottom": "20px"}, children=[
                kpi("R² Score",               f"{r2:.3f}",               "#f39c12"),
                kpi("Mean Absolute Error",    f"R {mae:,.0f}/day",       TEXT),
                kpi("Corr: Hours vs Revenue", f"{corr_hours:.3f}",       "#e74c3c"),
                kpi("Corr: Stage vs Revenue", f"{corr_stage:.3f}",       "#e74c3c"),
                kpi("Cost per LS Hour",       f"R {ls_cost:,.0f}",       "#e67e22"),
            ]),

            # Forecast chart
            html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                html.Div(dcc.Graph(figure=fig_forecast, config={"displayModeBar": False}),
                         style={**CHART_CARD, "flex": "2"}),

                # Monthly forecast table
                html.Div(style={**CHART_CARD, "flex": "1"}, children=[
                    html.H3("April 2024 Monthly Estimates",
                            style={"color": TEXT, "margin": "8px 8px 16px 8px", "fontSize": "14px"}),
                    html.P("Based on 22 trading weekdays",
                           style={"color": MUTED, "margin": "0 8px 12px 8px", "fontSize": "12px"}),
                    *[html.Div(style={
                        "padding": "12px 14px", "borderBottom": "1px solid #222",
                        "display": "flex", "justifyContent": "space-between"
                    }, children=[
                        html.Span(row["Scenario"], style={"color": MUTED, "fontSize": "13px"}),
                        html.Span(f"R {row['Monthly']:,.0f}",
                                  style={"color": TEXT, "fontWeight": "700", "fontSize": "14px"}),
                    ]) for _, row in fdf.iterrows()],
                ]),
            ]),

            # Stage impact table
            html.Div(style={**CHART_CARD, "marginTop": "8px"}, children=[
                html.H3("Average Revenue by Load Shedding Stage",
                        style={"color": TEXT, "margin": "8px 8px 16px 8px", "fontSize": "14px"}),
                html.Table(style={"width": "100%", "borderCollapse": "collapse"}, children=[
                    html.Thead(html.Tr([
                        html.Th("Stage",           style=TH),
                        html.Th("Avg Daily Revenue", style=TH),
                        html.Th("vs Stage 0",       style=TH),
                    ])),
                    html.Tbody(table_rows),
                ]),
            ]),

            # Key findings box
            html.Div(style={
                **CHART_CARD, "marginTop": "8px",
                "border": f"1px solid {ACCENT}", "padding": "20px 24px"
            }, children=[
                html.H3("📋 Key Findings", style={"color": ACCENT, "margin": "0 0 16px 0"}),
                html.P(f"1. Strong negative correlation of {corr_hours:.2f} between load shedding hours and revenue — more load shedding consistently meant lower revenue.",
                       style={"color": TEXT, "marginBottom": "10px", "lineHeight": "1.6"}),
                html.P(f"2. Stage 6 load shedding reduced average daily revenue by {stage6_drop_pct:.1f}% compared to Stage 0 days.",
                       style={"color": TEXT, "marginBottom": "10px", "lineHeight": "1.6"}),
                html.P(f"3. Total estimated revenue lost over 3 months: R {total_loss:,.2f} across {ls_days} load shedding days.",
                       style={"color": TEXT, "marginBottom": "10px", "lineHeight": "1.6"}),
                html.P(f"4. Each hour of load shedding cost approximately R {ls_cost:,.0f} in lost revenue.",
                       style={"color": TEXT, "marginBottom": "10px", "lineHeight": "1.6"}),
                html.P("5. Recommendation: Investing in a generator or UPS system would likely pay for itself within 3–4 months based on estimated daily losses.",
                       style={"color": TEXT, "marginBottom": "0", "lineHeight": "1.6"}),
            ]),
        ])
    ])


def page_404():
    return html.Div([
        navbar("/"),
        html.Div(style={"textAlign": "center", "marginTop": "100px"}, children=[
            html.H1("404", style={"color": ACCENT, "fontSize": "72px", "margin": "0"}),
            html.P("Page not found.", style={"color": MUTED, "fontSize": "18px"}),
            html.A("← Home", href="/", style={**NAV_ACTIVE, "display": "inline-block", "marginTop": "16px"}),
        ])
    ])


# ── APP ───────────────────────────────────────────────────────────────────

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    style={"background": BG, "minHeight": "100vh",
           "fontFamily": "'Segoe UI', Arial, sans-serif"},
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
    ]
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/" or pathname is None:
        return page_home()
    elif pathname == "/dashboard":
        return page_dashboard()
    elif pathname == "/forecast":
        return page_forecast()
    else:
        return page_404()


if __name__ == "__main__":
    print("\n⚡ Load Shedding Dashboard starting...")
    print("   Open http://127.0.0.1:8050 in your browser")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=True)
