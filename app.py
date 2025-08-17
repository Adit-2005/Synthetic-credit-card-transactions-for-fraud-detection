import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import uuid
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Check for XGBoost availability
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# Random seed
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# -----------------------------
# Synthetic Data Generation
# -----------------------------
DEFAULT_CONFIG = {
    "N_CUSTOMERS": 2000,
    "N_MERCHANTS": 800,
    "N_TXNS": 10000,
    "START_DATE": "2025-01-01",
    "DAYS": 45,
    "TARGET_FRAUD_RATE": 0.025,
}

def rand_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def sample_latlon(region):
    boxes = {
        "NA": (24.0, 49.0, -125.0, -66.0),
        "EU": (36.0, 60.0, -10.0, 30.0),
        "IN": (8.0, 28.0, 68.0, 88.0),
        "SEA": (-10.0, 10.0, 95.0, 125.0),
        "AU": (-43.0, -10.0, 113.0, 153.0),
    }
    lat_min, lat_max, lon_min, lon_max = boxes[region]
    return np.random.uniform(lat_min, lat_max), np.random.uniform(lon_min, lon_max)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def weighted_choice(choices, weights):
    cum = np.cumsum(weights)
    r = random.random() * cum[-1]
    return choices[np.searchsorted(cum, r)]


@st.cache_data
def generate_transactions(config=None, seed=SEED):
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        config = {**DEFAULT_CONFIG, **config}

    if isinstance(config["START_DATE"], str):
        config["START_DATE"] = pd.to_datetime(config["START_DATE"]) 

    np.random.seed(seed)
    random.seed(seed)

    regions = ["NA", "EU", "IN", "SEA", "AU"]
    region_weights = [0.28, 0.25, 0.24, 0.15, 0.08]
    mcc_codes = [
        "5411",
        "5812",
        "5942",
        "5732",
        "5999",
        "4111",
        "4812",
        "5967",
        "6011",
        "7995",
        "4814",
    ]
    currencies = ["USD", "EUR", "INR", "GBP", "AUD", "CAD"]

    # Customers
    customers = pd.DataFrame(
        {
            "customer_id": [rand_id("cust") for _ in range(config["N_CUSTOMERS"])],
            "home_region": [
                weighted_choice(regions, region_weights)
                for _ in range(config["N_CUSTOMERS"])
            ],
            "night_owl": np.random.rand(config["N_CUSTOMERS"]) < 0.25,
            "intl_traveler": np.random.rand(config["N_CUSTOMERS"]) < 0.10,
            "risk_score": np.random.beta(2, 8, size=config["N_CUSTOMERS"]),
        }
    )
    customers[["home_lat", "home_lon"]] = customers["home_region"].apply(
        lambda r: pd.Series(sample_latlon(r))
    )

    # Merchants
    merchants = pd.DataFrame(
        {
            "merchant_id": [rand_id("mrc") for _ in range(config["N_MERCHANTS"])],
            "mcc": np.random.choice(
                mcc_codes,
                size=config["N_MERCHANTS"],
                p=[0.15, 0.17, 0.06, 0.08, 0.08, 0.14, 0.07, 0.05, 0.08, 0.01, 0.11],
            ),
            "m_region": [
                weighted_choice(regions, region_weights)
                for _ in range(config["N_MERCHANTS"])
            ],
            "risk_level": np.random.beta(2.5, 7, size=config["N_MERCHANTS"]),
        }
    )
    merchants[["m_lat", "m_lon"]] = merchants["m_region"].apply(
        lambda r: pd.Series(sample_latlon(r))
    )
    merchants["is_online"] = merchants["mcc"].isin(["4814", "5967", "7995"]) | (
        np.random.rand(config["N_MERCHANTS"]) < 0.25
    )

    # Transactions
    txns = pd.DataFrame(
        {
            "transaction_id": [rand_id("txn") for _ in range(config["N_TXNS"])],
            "customer_id": np.random.choice(
                customers["customer_id"], size=config["N_TXNS"]
            ),
            "merchant_id": np.random.choice(
                merchants["merchant_id"], size=config["N_TXNS"]
            ),
            "timestamp": pd.to_datetime(config["START_DATE"]) 
            + pd.to_timedelta(
                np.random.randint(0, config["DAYS"] * 86400, size=config["N_TXNS"]),
                unit="s",
            ),
            "currency": np.random.choice(currencies, size=config["N_TXNS"]),
            "device_id": [rand_id("dev") for _ in range(config["N_TXNS"])],
            "channel": np.random.choice(
                ["POS", "ONLINE", "ATM"], size=config["N_TXNS"], p=[0.6, 0.3, 0.1]
            ),
            "pos_entry_mode": np.random.choice(
                ["CONTACTLESS", "CHIP", "MAGSTRIPE", "ECOM"],
                size=config["N_TXNS"],
                p=[0.4, 0.3, 0.2, 0.1],
            ),
            "ip_country": np.random.choice(
                ["US", "GB", "IN", "AU", "DE"], size=config["N_TXNS"]
            ),
            "billing_shipping_mismatch": np.random.choice(
                [0, 1], size=config["N_TXNS"], p=[0.9, 0.1]
            ),
            "declined_before_approved": np.random.choice(
                [0, 1], size=config["N_TXNS"], p=[0.95, 0.05]
            ),
        }
    )

    txns = txns.merge(customers, on="customer_id").merge(merchants, on="merchant_id")

    amount_ranges = {
        "5411": (15, 80),
        "5812": (8, 60),
        "5942": (10, 90),
        "5732": (40, 800),
        "5999": (5, 120),
        "4111": (2, 50),
        "4812": (10, 150),
        "5967": (15, 300),
        "6011": (20, 500),
        "7995": (10, 600),
        "4814": (5, 500),
    }
    txns["amount"] = txns["mcc"].map(amount_ranges).apply(
        lambda x: np.clip(
            np.random.gamma(2.0, (x[1] - x[0]) / 4.0) + x[0], x[0], x[1]
        ).round(2)
    )

    # Inject simple CNP clusters
    txns["is_fraud"] = 0
    n_cnp_clusters = max(1, int(config["TARGET_FRAUD_RATE"] * config["N_TXNS"] / 6))
    cnp_centers = txns.sample(n=n_cnp_clusters, random_state=seed).index
    for idx in cnp_centers:
        cust_id = txns.at[idx, "customer_id"]
        t0 = txns.at[idx, "timestamp"]
        mask = (
            (txns["customer_id"] == cust_id)
            & (txns["timestamp"] >= t0)
            & (txns["timestamp"] <= t0 + timedelta(hours=2))
        )
        affected = txns.loc[mask].head(6).index
        txns.loc[affected, "is_fraud"] = 1
        txns.loc[affected, "device_id"] = txns.loc[affected, "device_id"].apply(
            lambda _: rand_id("newdev")
        )
        txns.loc[affected, "channel"] = "ONLINE"
        txns.loc[affected, "pos_entry_mode"] = "ECOM"
        txns.loc[affected, "billing_shipping_mismatch"] = 1

    # Feature engineering
    txns["hour"] = txns["timestamp"].dt.hour
    txns["dow"] = txns["timestamp"].dt.dayofweek
    txns["km_home_to_merchant"] = txns.apply(
        lambda row: haversine(row["home_lat"], row["home_lon"], row["m_lat"], row["m_lon"]),
        axis=1,
    )
    txns["is_international"] = txns["home_region"] != txns["m_region"]

    def ip_match(row):
        if row["home_region"] == "NA":
            return row["ip_country"] in ["US", "CA"]
        if row["home_region"] == "EU":
            return row["ip_country"] in ["GB", "DE", "FR"]
        if row["home_region"] == "IN":
            return row["ip_country"] == "IN"
        if row["home_region"] == "AU":
            return row["ip_country"] in ["AU", "NZ"]
        if row["home_region"] == "SEA":
            return row["ip_country"] in ["SG", "MY"]
        return False

    txns["ip_matches_customer_region"] = txns.apply(ip_match, axis=1)

    # Sort & time deltas
    txns = txns.sort_values("timestamp")
    txns["minutes_since_prev"] = (
        txns.groupby("customer_id")["timestamp"].diff().dt.total_seconds() / 60
    )

    def calc_km_from_prev(group):
        if len(group) > 1:
            distances = []
            prev_lat, prev_lon = group.iloc[0]["home_lat"], group.iloc[0]["home_lon"]
            for i in range(1, len(group)):
                curr_lat, curr_lon = group.iloc[i]["m_lat"], group.iloc[i]["m_lon"]
                distances.append(haversine(prev_lat, prev_lon, curr_lat, curr_lon))
                prev_lat, prev_lon = curr_lat, curr_lon
            return [0] + distances
        return [0]

    txns["km_from_prev"] = (
        txns.groupby("customer_id").apply(calc_km_from_prev).explode().values
    )

    # Rolling counts
    txns["txns_last_10m"] = (
        txns.groupby("customer_id").rolling("10min", on="timestamp").count()[
            "transaction_id"
        ].values
    )
    txns["txns_last_1h"] = (
        txns.groupby("customer_id").rolling("1h", on="timestamp").count()[
            "transaction_id"
        ].values
    )
    txns["txns_last_24h"] = (
        txns.groupby("customer_id").rolling("24h", on="timestamp").count()[
            "transaction_id"
        ].values
    )
    txns["device_txns_last_1h"] = (
        txns.groupby("device_id").rolling("1h", on="timestamp").count()[
            "transaction_id"
        ].values
    )
    txns["merchant_txns_last_1h"] = (
        txns.groupby("merchant_id").rolling("1h", on="timestamp").count()[
            "transaction_id"
        ].values
    )

    # Rolling amount stats
    txns["amount_avg_1d"] = (
        txns.groupby("customer_id").rolling("1d", on="timestamp")["amount"].mean().values
    )
    txns["amount_std_1d"] = (
        txns.groupby("customer_id").rolling("1d", on="timestamp")["amount"].std().values
    )

    return {"txns": txns}


def plot_interactive_eda(txns: pd.DataFrame):
    figs = {}

    # KPI Pie
    fraud_counts = txns["is_fraud"].value_counts()
    figs["fraud_pie"] = px.pie(
        values=fraud_counts.values,
        names=["Legitimate", "Fraud"],
        title="Transaction Fraud Distribution",
        hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    # Violin amount by fraud
    figs["amount_violin"] = px.violin(
        txns,
        x="is_fraud",
        y="amount",
        color="is_fraud",
        box=True,
        points="all",
        labels={"is_fraud": "Fraud Status", "amount": "Amount"},
        title="Transaction Amount by Fraud Status",
    )

    # Hourly fraud rate
    hourly = txns.groupby(["hour", "is_fraud"]).size().unstack().fillna(0)
    hourly["fraud_rate"] = hourly.get(1, 0) / (hourly.sum(axis=1) + 1e-9)
    figs["hourly_trend"] = px.line(
        hourly.reset_index(),
        x="hour",
        y="fraud_rate",
        title="Fraud Rate by Hour of Day",
    )

    # Treemap by MCC
    mcc = txns.groupby("mcc")["is_fraud"].agg(["count", "mean"]).reset_index()
    mcc.columns = ["MCC", "Total", "Fraud Rate"]
    figs["mcc_tree"] = px.treemap(
        mcc, path=["MCC"], values="Total", color="Fraud Rate", title="MCC Breakdown"
    )

    # Geo (sampled)
    sample_txns = txns.sample(min(5000, len(txns)), random_state=SEED)
    figs["geo"] = px.scatter_geo(
        sample_txns,
        lat="home_lat",
        lon="home_lon",
        color="is_fraud",
        hover_data=["amount", "hour", "mcc"],
        title="Geographic Distribution of Transactions",
    )

    return figs


# -----------------------------
# Feature Prep & Modeling
# -----------------------------

def prepare_features(txns: pd.DataFrame):
    categorical_cols = [
        "currency",
        "channel",
        "pos_entry_mode",
        "ip_country",
        "home_region",
        "m_region",
        "mcc",
    ]
    numeric_cols = [
        "amount",
        "is_international",
        "billing_shipping_mismatch",
        "declined_before_approved",
        "km_home_to_merchant",
        "minutes_since_prev",
        "km_from_prev",
        "txns_last_10m",
        "txns_last_1h",
        "txns_last_24h",
        "device_txns_last_1h",
        "merchant_txns_last_1h",
        "amount_avg_1d",
        "amount_std_1d",
        "hour",
        "dow",
        "ip_matches_customer_region",
    ]

    X_cat = pd.get_dummies(txns[categorical_cols].astype(str), drop_first=True)
    X_num = txns[numeric_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_num), columns=X_num.columns, index=X_num.index
    )
    X = pd.concat([X_num_scaled, X_cat], axis=1)
    y = txns["is_fraud"].astype(int).values
    return X, y


@st.cache_data
def train_and_evaluate(X, y, models_to_run=None, seed=SEED):
    model_mapping = {
        "Logistic Regression": "LogReg",
        "Random Forest": "RF",
        "SVM": "SVM",
        "KNN": "KNN",
        "XGBoost": "XGB",
    }

    if models_to_run is None:
        models_to_run = ["LogReg", "RF", "SVM", "KNN"] + (["XGB"] if HAVE_XGB else [])
    else:
        models_to_run = [model_mapping[m] for m in models_to_run if m in model_mapping]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    models = {
        "LogReg": LogisticRegression(max_iter=200, class_weight="balanced", random_state=seed),
        "RF": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", random_state=seed, n_jobs=-1
        ),
        "SVM": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }

    if HAVE_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            scale_pos_weight=max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum())),
        )

    results = []
    roc_data = {}

    for name, model in models.items():
        if name not in models_to_run:
            continue
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
        }
        if y_proba is not None:
            metrics["ROC AUC"] = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[name] = {"fpr": fpr, "tpr": tpr}
        results.append(metrics)

    results_df = pd.DataFrame(results)
    return results_df, roc_data


# -----------------------------
# Streamlit App (Enhanced UI)
# -----------------------------

st.set_page_config(
    page_title="Fraud Detection Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS – modern, glassmorphism, hover states
st.markdown(
    """
    <style>
      :root { --card-bg: rgba(255,255,255,0.55); --glass: blur(10px); }
      .stApp { background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); }
      .glass {
        background: var(--card-bg);
        backdrop-filter: var(--glass);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 18px; padding: 18px; box-shadow: 0 10px 25px rgba(0,0,0,.15);
      }
      .metric-card { text-align:center; }
      .metric-card:hover { transform: translateY(-3px); transition: .2s; }
      .download-card button { width:100%; }
      .ticker { color:#fff; padding: 8px 12px; border-radius: 8px; background: rgba(231,76,60,.25);}
      .footer { color:#cfd8dc; font-size:12px; text-align:center; margin-top: 24px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar -----------------
with st.sidebar:
    st.title("🔍 Fraud Detection Pro")
    st.caption("Premium synthetic fraud analytics")

    st.markdown("---")
    # Navigation
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔍 Explorer", "🤖 Models", "📤 Export"],
        index=0,
    )

    with st.expander("⚙️ Configuration", expanded=True):
        n_txns = st.number_input(
            "Number of Transactions", min_value=1000, max_value=50000, value=10000, step=1000
        )
        target_fraud = st.slider("Target Fraud Rate (%)", 0.1, 10.0, 2.5, 0.1)
        seed = st.number_input("Random Seed", value=SEED, step=1)

    with st.expander("🤖 Model Selection (Optional)"):
        base_models = ["Logistic Regression", "Random Forest", "SVM", "KNN"]
        if HAVE_XGB:
            available_models = base_models + ["XGBoost"]
        else:
            available_models = base_models
            st.info("XGBoost not installed. Install with: pip install xgboost")
        models_selected = st.multiselect(
            "Models to Run (Leave empty to just generate data)", available_models, default=[]
        )

    run_btn = st.button("🚀 Generate & Analyze", type="primary", use_container_width=True)

# ------------- Run / Generate --------------
if run_btn:
    cfg = {
        "N_CUSTOMERS": 2000,
        "N_MERCHANTS": 800,
        "N_TXNS": int(n_txns),
        "TARGET_FRAUD_RATE": float(target_fraud) / 100.0,
        "START_DATE": "2025-01-01",
        "DAYS": 45,
    }
    with st.spinner("Generating synthetic transactions and features…"):
        out = generate_transactions(cfg, seed=int(seed))
        txns = out["txns"]
        st.session_state.generated_data = txns

    st.success("Dataset ready! Explore, model, or export from the sidebar.")
    st.balloons()

# ------------- Helper UI bits --------------
def fraud_ticker(txns: pd.DataFrame):
    if txns is None or len(txns) == 0:
        return
    rate = txns["is_fraud"].mean() * 100
    total = len(txns)
    flagged = int(txns["is_fraud"].sum())
    st.markdown(
        f"<div class='ticker'>🚨 Fraud Alert: <b>{rate:.2f}%</b> of <b>{total:,}</b> txns flagged (<b>{flagged:,}</b> cases)</div>",
        unsafe_allow_html=True,
    )

# ----------------- Pages -------------------

def dashboard_page():
    txns = st.session_state.get("generated_data")
    if txns is None:
        st.info("Configure parameters in the sidebar and click **Generate & Analyze** to begin.")
        return

    fraud_ticker(txns)

    # KPI cards
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container():
            st.markdown("<div class='glass metric-card'>", unsafe_allow_html=True)
            st.metric("Total Transactions", f"{len(txns):,}")
            st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        with st.container():
            st.markdown("<div class='glass metric-card'>", unsafe_allow_html=True)
            st.metric("Fraud Cases", f"{int(txns['is_fraud'].sum()):,}", f"{txns['is_fraud'].mean()*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        with st.container():
            st.markdown("<div class='glass metric-card'>", unsafe_allow_html=True)
            st.metric("Avg Amount", f"${txns['amount'].mean():.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # EDA charts
    figs = plot_interactive_eda(txns)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(figs["fraud_pie"], use_container_width=True)
    with r1c2:
        st.plotly_chart(figs["amount_violin"], use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(figs["hourly_trend"], use_container_width=True)
    with r2c2:
        st.plotly_chart(figs["mcc_tree"], use_container_width=True)

    st.plotly_chart(figs["geo"], use_container_width=True)

    with st.expander("📁 Preview Data (Top 150)", expanded=False):
        st.dataframe(txns.head(150), use_container_width=True)


def explorer_page():
    txns = st.session_state.get("generated_data")
    if txns is None:
        st.info("Generate data first from the sidebar.")
        return

    st.subheader("Transaction Explorer")
    c1, c2, c3 = st.columns(3)
    with c1:
        amt_min, amt_max = float(txns["amount"].min()), float(txns["amount"].max())
        amount_filter = st.slider("Filter by Amount", amt_min, amt_max, (amt_min, amt_max))
    with c2:
        fraud_filter = st.selectbox("Fraud Status", ["All", "Fraud Only", "Legitimate Only"]) 
    with c3:
        region = st.multiselect("Region", sorted(txns["m_region"].unique().tolist()))

    filtered = txns[(txns["amount"] >= amount_filter[0]) & (txns["amount"] <= amount_filter[1])]
    if fraud_filter == "Fraud Only":
        filtered = filtered[filtered["is_fraud"] == 1]
    elif fraud_filter == "Legitimate Only":
        filtered = filtered[filtered["is_fraud"] == 0]
    if region:
        filtered = filtered[filtered["m_region"].isin(region)]

    st.dataframe(filtered, use_container_width=True, height=420)

    st.plotly_chart(
        px.histogram(filtered, x="amount", color="is_fraud", title="Amount Distribution by Fraud Status"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.scatter(
            filtered.sample(min(len(filtered), 5000), random_state=SEED),
            x="hour",
            y="amount",
            color="is_fraud",
            hover_data=["mcc", "channel", "currency"],
            title="Transactions by Hour and Amount",
        ),
        use_container_width=True,
    )


def models_page():
    txns = st.session_state.get("generated_data")
    if txns is None:
        st.info("Generate data first from the sidebar.")
        return

    models_selected_local = st.session_state.get("models_selected_cache")

    # Let user (re)select models inline too
    base_models = ["Logistic Regression", "Random Forest", "SVM", "KNN"]
    available_models = base_models + (["XGBoost"] if HAVE_XGB else [])
    chosen = st.multiselect("Models to Run", available_models, default=models_selected_local or [])

    if st.button("▶️ Train / Evaluate", type="primary"):
        X, y = prepare_features(txns)
        with st.spinner("Training models & computing metrics…"):
            results_df, roc_data = train_and_evaluate(X, y, models_to_run=chosen, seed=SEED)
        st.session_state.model_results = results_df
        st.session_state.roc_data = roc_data
        st.success("Models evaluated!")
        st.balloons()

    results_df = st.session_state.get("model_results")
    roc_data = st.session_state.get("roc_data", {})

    if results_df is None or results_df.empty:
        st.info("Select models and click **Train / Evaluate**.")
        return

    # Summary table with styling
    st.markdown("### Model Performance Summary")
    display_df = results_df.copy()
    st.dataframe(
        display_df.style.highlight_max(axis=0).format({c: "{:.3f}" for c in display_df.columns if c != "Model"}),
        use_container_width=True,
    )

    # Gauges for each model
    st.markdown("### Metrics Gauges")
    gauge_cols = ["Precision", "Recall", "F1"]
    for _, row in results_df.iterrows():
        st.markdown(f"#### {row['Model']}")
        g1, g2, g3 = st.columns(3)
        for col, val, container in zip(gauge_cols, [row[c] for c in gauge_cols], [g1, g2, g3]):
            with container:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=float(val) if pd.notnull(val) else 0.0,
                        title={"text": col},
                        gauge={"axis": {"range": [0, 1]}}
                    )
                )
                fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

    # ROC Overlay
    if roc_data:
        st.markdown("### ROC Curves (Overlay)")
        fig = go.Figure()
        for name, curves in roc_data.items():
            fig.add_trace(go.Scatter(x=curves["fpr"], y=curves["tpr"], mode="lines", name=name))
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash"))
        )
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)


def export_page():
    txns = st.session_state.get("generated_data")
    if txns is None:
        st.info("Generate data first from the sidebar.")
        return

    st.subheader("Download the generated dataset")

    # CSV
    csv_bytes = txns.to_csv(index=False).encode("utf-8")

    # JSON
    json_str = txns.to_json(indent=2)

    # Try Excel (optional)
    excel_bytes = None
    try:
        import io
        with pd.ExcelWriter(io.BytesIO(), engine="xlsxwriter") as writer:
            txns.to_excel(writer, index=False, sheet_name="data")
            writer.book.close()
            excel_bytes = writer._writer.fp.getvalue()
    except Exception:
        st.warning("Excel export unavailable (xlsxwriter not installed). Use CSV/JSON.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "💾 Download CSV",
            data=csv_bytes,
            file_name=f"fraud_data_{len(txns)}_txns.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        if excel_bytes is not None:
            st.download_button(
                "📗 Download Excel",
                data=excel_bytes,
                file_name=f"fraud_data_{len(txns)}_txns.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.info("Install `xlsxwriter` to enable Excel downloads.")
    with c3:
        st.download_button(
            "🧾 Download JSON",
            data=json_str,
            file_name=f"fraud_data_{len(txns)}_txns.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    st.write("Preview (first 100 rows):")
    st.dataframe(txns.head(100), use_container_width=True, height=400)


# ---------------- Render chosen page ----------------
if page == "📊 Dashboard":
    dashboard_page()
elif page == "🔍 Explorer":
    explorer_page()
elif page == "🤖 Models":
    models_page()
elif page == "📤 Export":
    export_page()

st.markdown("---")
st.caption("© 2025 Fraud Detection Pro | v3.0 | Enhanced UI & Export")
