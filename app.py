"""
E-Commerce Sales Intelligence Platform
========================================
Production-grade Streamlit application: EDA dashboard, ML sales prediction
(Random Forest with confidence intervals + feature importance), data
explorer, and model performance monitoring.

Run:
    streamlit run app.py

Requires eCommercePK.csv in the same folder (or set DATA_PATH env var).
"""

from __future__ import annotations

import logging
import textwrap
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# LOGGING  (replaces silent `except: pass`)
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ecommerce_ai")


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="E-Commerce Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small helper so every HTML block is guaranteed flush-left.
# This is the fix for the "raw tags shown as text" bug: Streamlit's
# markdown parser treats 4+ space indented lines as a code block,
# so any HTML written with normal Python indentation renders as
# literal text instead of being parsed as HTML. dedent() strips
# that leading whitespace before it ever reaches st.markdown().


def html(block: str) -> None:
    st.markdown(textwrap.dedent(block).strip(), unsafe_allow_html=True)


# ============================================================
# THEME  (corporate navy / slate, high contrast, print-safe)
# ============================================================

html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0B1220; color: #E5E9F0; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }

    [data-testid="stSidebar"] { background: #0F172A; border-right: 1px solid #1E293B; }
    [data-testid="stSidebar"] * { color: #E2E8F0; }
    [data-testid="stSidebar"] .stCaption { color: #64748B !important; }

    h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; }
    p, span, label { color: #94A3B8; }

    .hero {
        padding: 28px 32px; border-radius: 18px;
        background: linear-gradient(135deg, #111827 0%, #0B1B33 100%);
        border: 1px solid #1E3A5F;
        box-shadow: 0 10px 40px rgba(0,0,0,0.35);
        margin-bottom: 22px;
    }
    .hero-kicker {
        color: #38BDF8; font-size: 0.75rem; font-weight: 800;
        letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 6px;
    }
    .hero-title { font-size: clamp(1.8rem, 3.2vw, 2.8rem); font-weight: 800; margin: 0; color: #F8FAFC; }
    .hero-subtitle { margin-top: 10px; color: #94A3B8; max-width: 820px; line-height: 1.6; font-size: 0.95rem; }

    .section-title {
        color: #F1F5F9; font-size: 1.05rem; font-weight: 800;
        margin: 22px 0 10px; text-transform: uppercase; letter-spacing: 0.05em;
    }

    .status-pill {
        display: inline-block; padding: 5px 10px; border-radius: 999px;
        background: rgba(16,185,129,0.12); border: 1px solid #10B981;
        color: #34D399; font-size: 0.72rem; font-weight: 700;
    }
    .status-pill.warn { background: rgba(245,158,11,0.12); border-color: #F59E0B; color: #FBBF24; }
    .status-pill.err  { background: rgba(239,68,68,0.12);  border-color: #EF4444; color: #F87171; }

    .info-box {
        border-left: 4px solid #38BDF8; background: rgba(56,189,248,0.06);
        padding: 12px 15px; border-radius: 8px; color: #CBD5E1; font-size: 0.85rem; line-height: 1.55;
    }

    .prediction-box {
        margin-top: 14px; padding: 28px; border-radius: 18px;
        background: linear-gradient(135deg, #0B1B33 0%, #0F2A2E 100%);
        border: 1px solid #1E3A5F; text-align: center;
        box-shadow: 0 8px 30px rgba(56,189,248,0.10);
    }
    .prediction-label { color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 700; }
    .prediction-value { color: #38BDF8; font-size: 2.6rem; font-weight: 800; margin: 4px 0; }
    .prediction-range { color: #64748B; font-size: 0.85rem; }

    [data-testid="stMetric"] {
        background: #111827; border: 1px solid #1E293B; padding: 16px; border-radius: 14px;
    }
    [data-testid="stMetricLabel"] { color: #94A3B8 !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #F1F5F9 !important; font-weight: 800; }
    [data-testid="stMetricDelta"] svg { display: inline; }

    div.stButton > button {
        width: 100%; border-radius: 10px; border: 1px solid #0EA5E9;
        background: linear-gradient(135deg, #0EA5E9, #0284C7);
        color: #F8FAFC; font-weight: 800; min-height: 46px;
        box-shadow: 0 8px 20px rgba(14,165,233,0.18); transition: all .15s ease;
    }
    div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 10px 25px rgba(14,165,233,0.28); }

    .stDownloadButton > button {
        border-radius: 10px; background: #111827; color: #38BDF8; border: 1px solid #0EA5E9; font-weight: 700;
    }

    [data-testid="stDataFrame"] { border: 1px solid #1E293B; border-radius: 10px; }
    hr { border-color: #1E293B !important; }

    .footer { text-align: center; color: #475569; font-size: 0.75rem; padding: 26px 0 4px; }
    </style>
    """
)


# ============================================================
# PATHS / SCHEMA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "eCommercePK.csv"
MODEL_PATH = BASE_DIR / "ecommerce_sales_model.joblib"

NUMERIC_FEATURES = ["quantity", "year", "month", "day", "day_of_week"]
CATEGORICAL_FEATURES = ["order_source", "category", "sku", "city"]
REQUIRED_COLUMNS = {"order_date", "sales", "quantity", *CATEGORICAL_FEATURES}


# ============================================================
# DATA LOADING  (validated, with real error surfacing)
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{path.name}'. Place eCommercePK.csv next to app.py, "
            "or update DATA_PATH."
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    object_cols = df.select_dtypes(include="object").columns.tolist()
    if object_cols:
        df[object_cols] = df[object_cols].apply(lambda col: col.str.strip())

    df["order_date"] = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
    n_bad_dates = df["order_date"].isna().sum()
    if n_bad_dates:
        logger.warning("%s rows had unparseable order_date and were dropped from date features.", n_bad_dates)

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["day"] = df["order_date"].dt.day
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["month_name"] = df["order_date"].dt.strftime("%b %Y")

    return df


# ============================================================
# MODEL TRAINING  (cached, with importances + residual std for CI)
# ============================================================

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    model_df = df.dropna(subset=["sales", "order_date"]).copy()
    if len(model_df) < 30:
        raise ValueError("Not enough clean rows to train a reliable model (need at least 30).")

    X = model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = model_df["sales"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    residuals = y_test.values - predictions

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": r2_score(y_test, predictions),
        "MAPE": mean_absolute_percentage_error(y_test.replace(0, np.nan).dropna(),
                                                 predictions[: len(y_test.replace(0, np.nan).dropna())])
        if (y_test != 0).any() else float("nan"),
        "residual_std": float(np.std(residuals)),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    # Feature importances mapped back to human-readable names
    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .assign(feature=lambda d: d["feature"].str.replace(r"^(numerical|categorical)__", "", regex=True))
            .groupby("feature", as_index=False)["importance"].sum()
            .sort_values("importance", ascending=False)
        )
    except Exception as exc:  # pragma: no cover - defensive, logged not swallowed
        logger.exception("Could not compute feature importances: %s", exc)
        importance_df = pd.DataFrame(columns=["feature", "importance"])

    try:
        joblib.dump(pipeline, MODEL_PATH)
    except OSError as exc:
        logger.warning("Could not persist model artifact to disk: %s", exc)

    return pipeline, metrics, importance_df


def predict_with_interval(pipeline: Pipeline, input_row: pd.DataFrame) -> tuple[float, float, float]:
    """Point estimate + 90% interval from the spread of individual tree predictions."""
    preprocessor = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(input_row)

    tree_preds = np.array([tree.predict(X_transformed)[0] for tree in rf.estimators_])
    point = float(np.mean(tree_preds))
    lower = float(np.percentile(tree_preds, 5))
    upper = float(np.percentile(tree_preds, 95))
    return max(0.0, point), max(0.0, lower), max(0.0, upper)


# ============================================================
# LOAD + TRAIN  (fail loudly and specifically, not with st.stop() only)
# ============================================================

try:
    df = load_data(DATA_PATH)
except (FileNotFoundError, ValueError) as exc:
    st.error("The application could not load its dataset.")
    st.code(str(exc))
    st.stop()

try:
    model, metrics, importance_df = train_model(df)
except ValueError as exc:
    st.error("The application could not train a model on this dataset.")
    st.code(str(exc))
    st.stop()


# ============================================================
# SIDEBAR — navigation + global filters
# ============================================================

with st.sidebar:
    html("<h2 style='margin-bottom:0;'>📊 ECOMMERCE AI</h2>")
    st.caption("Sales Intelligence Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Executive Dashboard", "Sales Prediction", "Data Explorer", "Model Performance"],
        index=0,
    )

    st.divider()
    st.markdown("**Global filters**")
    st.caption("Applied to Executive Dashboard & Data Explorer")

    min_date, max_date = df["order_date"].min(), df["order_date"].max()
    date_range = st.date_input(
        "Order date range",
        value=(min_date.date(), max_date.date()) if pd.notna(min_date) else (date.today(), date.today()),
        min_value=min_date.date() if pd.notna(min_date) else None,
        max_value=max_date.date() if pd.notna(max_date) else None,
    )

    filter_city = st.multiselect("City", sorted(df["city"].dropna().unique()), default=[])
    filter_category = st.multiselect("Category", sorted(df["category"].dropna().unique()), default=[])

    st.divider()
    st.markdown("**System status**")
    r2_ok = metrics["R2"] >= 0.5
    html(f"<span class='status-pill{'' if r2_ok else ' warn'}'>● MODEL ONLINE</span>")
    st.caption(f"Dataset: {len(df):,} records")
    st.caption(f"Model: Random Forest · R² {metrics['R2']:.3f}")

    st.divider()
    html(
        """
        <div class="info-box">
        <b>Prediction inputs</b><br><br>
        Order source, category, SKU, quantity, city and order date.
        <br><br>
        Order ID and order status are intentionally excluded — they leak
        target information and would not be known at prediction time.
        </div>
        """
    )


def apply_global_filters(source_df: pd.DataFrame) -> pd.DataFrame:
    out = source_df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        out = out[(out["order_date"] >= start) & (out["order_date"] <= end)]
    if filter_city:
        out = out[out["city"].isin(filter_city)]
    if filter_category:
        out = out[out["category"].isin(filter_category)]
    return out


# ============================================================
# HERO
# ============================================================

html(
    """
    <div class="hero">
        <div class="hero-kicker">Machine Learning · Production Interface</div>
        <h1 class="hero-title">E-Commerce Sales Intelligence</h1>
        <div class="hero-subtitle">
            Executive-grade forecasting platform: interactive analytics, an
            explainable Random Forest pricing model with confidence intervals,
            and full data governance controls.
        </div>
    </div>
    """
)


# ============================================================
# PAGE: EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":
    view = apply_global_filters(df)

    if view.empty:
        st.warning("No records match the current filters. Adjust the sidebar filters to see data.")
        st.stop()

    html('<div class="section-title">Executive Overview</div>')

    # --- KPIs with period-over-period delta ------------------------------
    view_sorted = view.sort_values("order_date")
    mid_point = view_sorted["order_date"].min() + (view_sorted["order_date"].max() - view_sorted["order_date"].min()) / 2
    first_half = view_sorted[view_sorted["order_date"] <= mid_point]
    second_half = view_sorted[view_sorted["order_date"] > mid_point]

    def pct_delta(new: float, old: float) -> float | None:
        if old in (0, None) or pd.isna(old) or old == 0:
            return None
        return (new - old) / old * 100

    total_sales = view["sales"].sum()
    delta_sales = pct_delta(second_half["sales"].sum(), first_half["sales"].sum())

    avg_order = view["sales"].mean()
    delta_avg = pct_delta(second_half["sales"].mean(), first_half["sales"].mean())

    total_orders = len(view)
    delta_orders = pct_delta(len(second_half), len(first_half))

    total_units = view["quantity"].sum()
    delta_units = pct_delta(second_half["quantity"].sum(), first_half["quantity"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales", f"PKR {total_sales:,.0f}", f"{delta_sales:+.1f}%" if delta_sales is not None else None)
    c2.metric("Average Order", f"PKR {avg_order:,.0f}", f"{delta_avg:+.1f}%" if delta_avg is not None else None)
    c3.metric("Median Order", f"PKR {view['sales'].median():,.0f}")
    c4.metric("Orders", f"{total_orders:,}", f"{delta_orders:+.1f}%" if delta_orders is not None else None)
    c5.metric("Units Sold", f"{total_units:,.0f}", f"{delta_units:+.1f}%" if delta_units is not None else None)

    st.caption("Deltas compare the second half of the selected date range against the first half.")

    # --- Trend --------------------------------------------------------------
    html('<div class="section-title">Sales Trend</div>')

    trend = (
        view.dropna(subset=["order_date"])
        .set_index("order_date")
        .resample("W")["sales"]
        .sum()
        .reset_index()
    )
    fig_trend = px.area(
        trend, x="order_date", y="sales",
        labels={"order_date": "Week", "sales": "Sales (PKR)"},
    )
    fig_trend.update_traces(line_color="#38BDF8", fillcolor="rgba(56,189,248,0.15)")
    fig_trend.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), height=320,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- Category / source breakdown ----------------------------------------
    html('<div class="section-title">Business Intelligence</div>')
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### Sales by Product Category")
        category_sales = view.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
        fig_cat = px.bar(category_sales, x="sales", y="category", orientation="h", color="sales",
                          color_continuous_scale="Blues", labels={"sales": "Sales (PKR)", "category": ""})
        fig_cat.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=10, b=10), height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    with right:
        st.markdown("#### Sales by Order Source")
        source_sales = view.groupby("order_source", as_index=False)["sales"].sum()
        fig_src = px.pie(source_sales, names="order_source", values="sales", hole=0.55,
                          color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_src.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=10, b=10), height=340,
                               legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_src, use_container_width=True)

    # --- Top products ---------------------------------------------------------
    html('<div class="section-title">Top Products</div>')
    top_products = (
        view.groupby(["sku", "category"], as_index=False)
        .agg(Orders=("sku", "size"), Units=("quantity", "sum"), Sales=("sales", "sum"), Avg_Order=("sales", "mean"))
        .sort_values("Sales", ascending=False)
        .head(10)
    )
    top_products["Sales"] = top_products["Sales"].map(lambda x: f"PKR {x:,.0f}")
    top_products["Avg_Order"] = top_products["Avg_Order"].map(lambda x: f"PKR {x:,.0f}")
    st.dataframe(top_products, use_container_width=True, hide_index=True)

    # --- Executive report download ---------------------------------------------
    report_lines = [
        "E-COMMERCE SALES INTELLIGENCE — EXECUTIVE SUMMARY",
        f"Period: {view['order_date'].min().date()} to {view['order_date'].max().date()}",
        "",
        f"Total Sales: PKR {total_sales:,.0f}",
        f"Average Order Value: PKR {avg_order:,.0f}",
        f"Total Orders: {total_orders:,}",
        f"Units Sold: {total_units:,.0f}",
        "",
        "Top 10 Products by Sales:",
        top_products.to_string(index=False),
    ]
    st.download_button(
        "📄 Download Executive Summary (.txt)",
        data="\n".join(report_lines),
        file_name="executive_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# PAGE: SALES PREDICTION
# ============================================================

elif page == "Sales Prediction":
    html('<div class="section-title">AI Sales Prediction</div>')
    html(
        """
        <div class="info-box">
        Enter an order configuration below. The model returns a point
        estimate plus a 90% confidence interval derived from the spread
        of predictions across all 400 trees in the forest — this tells
        you how confident the model actually is, not just a single number.
        </div>
        """
    )
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        order_source = st.selectbox("Order Source", sorted(df["order_source"].dropna().unique()))
        category = st.selectbox("Category", sorted(df["category"].dropna().unique()))
        sku = st.selectbox("SKU / Product", sorted(df["sku"].dropna().unique()))
        quantity = st.number_input("Quantity", min_value=1, max_value=100, value=1, step=1)

    with col2:
        city = st.selectbox("City", sorted(df["city"].dropna().unique()))
        order_date_input = st.date_input("Order Date", value=pd.Timestamp("2025-01-01").date())

        st.markdown("#### Derived Date Features")
        year, month, day = order_date_input.year, order_date_input.month, order_date_input.day
        day_of_week = order_date_input.weekday()
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Year", year)
        d2.metric("Month", month)
        d3.metric("Day", day)
        d4.metric("Weekday", day_of_week)

    st.write("")

    if "prediction_log" not in st.session_state:
        st.session_state.prediction_log = []

    if st.button("⚡ GENERATE SALES PREDICTION", use_container_width=True):
        input_data = pd.DataFrame({
            "quantity": [quantity], "year": [year], "month": [month],
            "day": [day], "day_of_week": [day_of_week],
            "order_source": [order_source], "category": [category],
            "sku": [sku], "city": [city],
        })

        point, lower, upper = predict_with_interval(model, input_data)

        html(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">Predicted Sales</div>
                <div class="prediction-value">PKR {point:,.0f}</div>
                <div class="prediction-range">90% confidence interval: PKR {lower:,.0f} – PKR {upper:,.0f}</div>
            </div>
            """
        )
        st.write("")

        st.markdown("#### Prediction Payload")
        st.dataframe(input_data, use_container_width=True, hide_index=True)

        low, mid, high = st.columns(3)
        low.metric("Quantity", quantity)
        mid.metric("Product", sku)
        high.metric("Location", city)

        st.session_state.prediction_log.insert(0, {
            "SKU": sku, "City": city, "Category": category, "Qty": quantity,
            "Date": str(order_date_input), "Predicted": round(point, 0),
            "Low (5%)": round(lower, 0), "High (95%)": round(upper, 0),
        })

    if st.session_state.prediction_log:
        st.markdown("#### Session Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.prediction_log), use_container_width=True, hide_index=True)


# ============================================================
# PAGE: DATA EXPLORER
# ============================================================

elif page == "Data Explorer":
    html('<div class="section-title">Dataset Explorer</div>')

    view = apply_global_filters(df)

    a, b, c, d = st.columns(4)
    a.metric("Rows (filtered)", f"{view.shape[0]:,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Missing Values", f"{int(view.isnull().sum().sum()):,}")
    d.metric("Duplicate Rows", f"{int(view.duplicated().sum()):,}")

    st.caption(f"Showing {len(view):,} of {len(df):,} records (sidebar filters applied)")

    display_df = view.copy()
    if "order_date" in display_df.columns:
        display_df["order_date"] = display_df["order_date"].dt.strftime("%Y-%m-%d")

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=520)

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Filtered Data (.csv)", data=csv,
        file_name="ecommerce_filtered_data.csv", mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":
    html('<div class="section-title">Model Performance</div>')
    html(
        """
        <div class="info-box">
        The deployed model is a <b>Random Forest Regressor</b> (400 trees)
        predicting <b>sales</b> from order, product and date features.
        Metrics below are computed on a held-out 20% test split the model
        never saw during training.
        </div>
        """
    )
    st.write("")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"PKR {metrics['MAE']:,.2f}")
    m2.metric("RMSE", f"PKR {metrics['RMSE']:,.2f}")
    m3.metric("R²", f"{metrics['R2']:.4f}")
    m4.metric("MAPE", f"{metrics['MAPE']*100:,.1f}%" if pd.notna(metrics["MAPE"]) else "n/a")

    st.markdown("#### Evaluation Split")
    e1, e2, e3 = st.columns(3)
    e1.metric("Training Rows", f"{metrics['train_rows']:,}")
    e2.metric("Test Rows", f"{metrics['test_rows']:,}")
    e3.metric("Test Size", "20%")

    st.markdown("#### Feature Importance")
    if not importance_df.empty:
        fig_imp = px.bar(
            importance_df.head(15).sort_values("importance"),
            x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Blues",
            labels={"importance": "Relative importance", "feature": ""},
        )
        fig_imp.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10), height=420, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance could not be computed for this model run.")

    st.markdown("#### Feature Architecture")
    feature_table = pd.DataFrame({
        "Feature Group": ["Numerical"] * len(NUMERIC_FEATURES) + ["Categorical"] * len(CATEGORICAL_FEATURES),
        "Feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "Processing": (
            ["Median imputation + StandardScaler"] * len(NUMERIC_FEATURES)
            + ["Most-frequent imputation + OneHotEncoder"] * len(CATEGORICAL_FEATURES)
        ),
    })
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    st.markdown("#### Deployment Artifact")
    if MODEL_PATH.exists():
        html(f"<span class='status-pill'>● Model artifact saved: {MODEL_PATH.name}</span>")
    else:
        html("<span class='status-pill warn'>● Model trained in memory (artifact not yet persisted)</span>")


# ============================================================
# FOOTER
# ============================================================

html(
    """
    <div class="footer">
        E-Commerce Sales Intelligence · Machine Learning Deployment Project<br>
        Built with Python, Scikit-learn, Plotly and Streamlit
    </div>
    """
)