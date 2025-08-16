import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
import uuid
import random
import math
import io
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
except ImportError:
    HAVE_XGB = False

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Configuration
DEFAULT_CONFIG = {
    "N_CUSTOMERS": 2000,
    "N_MERCHANTS": 800,
    "N_TXNS": 10000,
    "START_DATE": "2025-01-01",
    "DAYS": 45,
    "TARGET_FRAUD_RATE": 0.025
}

# Helper Functions
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
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
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
    mcc_codes = ["5411", "5812", "5942", "5732", "5999", "4111", "4812", "5967", "6011", "7995", "4814"]
    currencies = ["USD", "EUR", "INR", "GBP", "AUD", "CAD"]

    # Generate customers
    customers = pd.DataFrame({
        "customer_id": [rand_id("cust") for _ in range(config["N_CUSTOMERS"])],
        "home_region": [weighted_choice(regions, region_weights) for _ in range(config["N_CUSTOMERS"])],
        "night_owl": np.random.rand(config["N_CUSTOMERS"]) < 0.25,
        "intl_traveler": np.random.rand(config["N_CUSTOMERS"]) < 0.10,
        "risk_score": np.random.beta(2, 8, size=config["N_CUSTOMERS"])
    })
    customers[["home_lat", "home_lon"]] = customers["home_region"].apply(
        lambda r: pd.Series(sample_latlon(r))
    )

    # Generate merchants
    merchants = pd.DataFrame({
        "merchant_id": [rand_id("mrc") for _ in range(config["N_MERCHANTS"])],
        "mcc": np.random.choice(mcc_codes, size=config["N_MERCHANTS"], p=[0.15, 0.17, 0.06, 0.08, 0.08, 0.14, 0.07, 0.05, 0.08, 0.01, 0.11]),
        "m_region": [weighted_choice(regions, region_weights) for _ in range(config["N_MERCHANTS"])],
        "risk_level": np.random.beta(2.5, 7, size=config["N_MERCHANTS"])
    })
    merchants[["m_lat", "m_lon"]] = merchants["m_region"].apply(
        lambda r: pd.Series(sample_latlon(r))
    )
    merchants["is_online"] = merchants["mcc"].isin(["4814", "5967", "7995"]) | (np.random.rand(config["N_MERCHANTS"]) < 0.25)

    # Generate transactions
    txns = pd.DataFrame({
        "transaction_id": [rand_id("txn") for _ in range(config["N_TXNS"])],
        "customer_id": np.random.choice(customers["customer_id"], size=config["N_TXNS"]),
        "merchant_id": np.random.choice(merchants["merchant_id"], size=config["N_TXNS"]),
        "timestamp": pd.to_datetime(config["START_DATE"]) + pd.to_timedelta(
            np.random.randint(0, config["DAYS"] * 86400, size=config["N_TXNS"]), unit='s'),
        "currency": np.random.choice(currencies, size=config["N_TXNS"]),
        "device_id": [rand_id("dev") for _ in range(config["N_TXNS"])],
        "channel": np.random.choice(["POS", "ONLINE", "ATM"], size=config["N_TXNS"], p=[0.6, 0.3, 0.1]),
        "pos_entry_mode": np.random.choice(["CONTACTLESS", "CHIP", "MAGSTRIPE", "ECOM"], 
                                         size=config["N_TXNS"], p=[0.4, 0.3, 0.2, 0.1]),
        "ip_country": np.random.choice(["US", "GB", "IN", "AU", "DE"], size=config["N_TXNS"]),
        "billing_shipping_mismatch": np.random.choice([0, 1], size=config["N_TXNS"], p=[0.9, 0.1]),
        "declined_before_approved": np.random.choice([0, 1], size=config["N_TXNS"], p=[0.95, 0.05])
    })

    # Merge with customer and merchant data
    txns = txns.merge(customers, on="customer_id")
    txns = txns.merge(merchants, on="merchant_id")

    # Generate transaction amounts
    amount_ranges = {
        "5411": (15, 80), "5812": (8, 60), "5942": (10, 90), "5732": (40, 800),
        "5999": (5, 120), "4111": (2, 50), "4812": (10, 150), "5967": (15, 300),
        "6011": (20, 500), "7995": (10, 600), "4814": (5, 500)
    }
    txns["amount"] = txns["mcc"].map(amount_ranges).apply(
        lambda x: np.clip(np.random.gamma(2.0, (x[1]-x[0])/4.0) + x[0], x[0], x[1]).round(2)
    )

    # Add fraud patterns
    txns["is_fraud"] = 0
    n_cnp_clusters = max(1, int(config["TARGET_FRAUD_RATE"] * config["N_TXNS"] / 6))
    cnp_centers = txns.sample(n=n_cnp_clusters, random_state=seed).index
    
    for idx in cnp_centers:
        cust_id = txns.at[idx, "customer_id"]
        t0 = txns.at[idx, "timestamp"]
        mask = (txns["customer_id"] == cust_id) & (txns["timestamp"] >= t0) & (txns["timestamp"] <= t0 + timedelta(hours=2))
        affected = txns.loc[mask].head(6).index
        txns.loc[affected, "is_fraud"] = 1
        txns.loc[affected, "device_id"] = txns.loc[affected, "device_id"].apply(lambda _: rand_id("newdev"))
        txns.loc[affected, "channel"] = "ONLINE"
        txns.loc[affected, "pos_entry_mode"] = "ECOM"
        txns.loc[affected, "billing_shipping_mismatch"] = 1

    # Feature engineering
    txns["hour"] = txns["timestamp"].dt.hour
    txns["dow"] = txns["timestamp"].dt.dayofweek
    txns["km_home_to_merchant"] = txns.apply(
        lambda row: haversine(row["home_lat"], row["home_lon"], row["m_lat"], row["m_lon"]), axis=1
    )
    txns["is_international"] = txns["home_region"] != txns["m_region"]
    txns["ip_matches_customer_region"] = txns.apply(
        lambda row: row["ip_country"] in ["US", "CA"] if row["home_region"] == "NA" else
                    row["ip_country"] in ["GB", "DE", "FR"] if row["home_region"] == "EU" else
                    row["ip_country"] == "IN" if row["home_region"] == "IN" else
                    row["ip_country"] in ["AU", "NZ"] if row["home_region"] == "AU" else
                    row["ip_country"] in ["SG", "MY"] if row["home_region"] == "SEA" else False,
        axis=1
    )

    # Time-based features
    txns = txns.sort_values("timestamp")
    txns["minutes_since_prev"] = txns.groupby("customer_id")["timestamp"].diff().dt.total_seconds() / 60
    
    # Simplified km_from_prev calculation
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
    
    txns["km_from_prev"] = txns.groupby("customer_id").apply(calc_km_from_prev).explode().values

    # Transaction count features
    txns["txns_last_10m"] = txns.groupby("customer_id").rolling("10min", on="timestamp").count()["transaction_id"].values
    txns["txns_last_1h"] = txns.groupby("customer_id").rolling("1h", on="timestamp").count()["transaction_id"].values
    txns["txns_last_24h"] = txns.groupby("customer_id").rolling("24h", on="timestamp").count()["transaction_id"].values
    txns["device_txns_last_1h"] = txns.groupby("device_id").rolling("1h", on="timestamp").count()["transaction_id"].values
    txns["merchant_txns_last_1h"] = txns.groupby("merchant_id").rolling("1h", on="timestamp").count()["transaction_id"].values

    # Amount statistics
    txns["amount_avg_1d"] = txns.groupby("customer_id").rolling("1d", on="timestamp")["amount"].mean().values
    txns["amount_std_1d"] = txns.groupby("customer_id").rolling("1d", on="timestamp")["amount"].std().values

    return {
        "txns": txns,
        "customers": customers,
        "merchants": merchants
    }

def plot_interactive_eda(txns):
    """Enhanced EDA with Plotly visualizations"""
    figs = {}
    
    # 1. Fraud Distribution (Pie Chart)
    fraud_counts = txns['is_fraud'].value_counts()
    figs['fraud_pie'] = px.pie(
        values=fraud_counts.values,
        names=['Legitimate', 'Fraud'],
        title='Transaction Fraud Distribution',
        color=['Legitimate', 'Fraud'],
        color_discrete_map={'Legitimate':'#2ecc71', 'Fraud':'#e74c3c'},
        hole=0.4
    )
    
    # 2. Amount Distribution (Violin Plot)
    figs['amount_dist'] = px.violin(
        txns, 
        x='is_fraud', 
        y='amount', 
        color='is_fraud',
        box=True,
        points="all",
        title='Transaction Amount by Fraud Status',
        labels={'is_fraud': 'Fraud Status', 'amount': 'Amount'},
        color_discrete_map={0:'#2ecc71', 1:'#e74c3c'}
    )
    
    # 3. Hourly Fraud Rate (Line Chart)
    hourly_data = txns.groupby(['hour', 'is_fraud']).size().unstack().fillna(0)
    hourly_data['fraud_rate'] = hourly_data[1] / (hourly_data[0] + hourly_data[1])
    figs['hourly_trend'] = px.line(
        hourly_data.reset_index(),
        x='hour',
        y='fraud_rate',
        title='Fraud Rate by Hour of Day',
        labels={'hour': 'Hour of Day', 'fraud_rate': 'Fraud Rate'}
    ).update_traces(line=dict(color='#e74c3c', width=3))
    
    # 4. Geographic Distribution (Scatter Map)
    sample_txns = txns.sample(min(5000, len(txns)))
    figs['geo_dist'] = px.scatter_geo(
        sample_txns,
        lat='home_lat',
        lon='home_lon',
        color='is_fraud',
        hover_data=['amount', 'hour'],
        title='Geographic Distribution of Transactions',
        color_discrete_map={0:'#2ecc71', 1:'#e74c3c'}
    )
    
    # 5. MCC Fraud Heatmap
    mcc_data = txns.groupby('mcc')['is_fraud'].agg(['count', 'mean']).reset_index()
    mcc_data.columns = ['MCC', 'Total Transactions', 'Fraud Rate']
    figs['mcc_heatmap'] = px.treemap(
        mcc_data,
        path=['MCC'],
        values='Total Transactions',
        color='Fraud Rate',
        title='Fraud Rate by Merchant Category Code',
        color_continuous_scale='RdYlGn_r'
    )
    
    return figs

def prepare_features(txns):
    categorical_cols = ["currency", "channel", "pos_entry_mode", "ip_country", 
                       "home_region", "m_region", "mcc"]
    numeric_cols = ["amount", "is_international", "billing_shipping_mismatch", 
                   "declined_before_approved", "km_home_to_merchant", 
                   "minutes_since_prev", "km_from_prev", "txns_last_10m", 
                   "txns_last_1h", "txns_last_24h", "device_txns_last_1h", 
                   "merchant_txns_last_1h", "amount_avg_1d", "amount_std_1d", 
                   "hour", "dow", "ip_matches_customer_region"]

    X_cat = pd.get_dummies(txns[categorical_cols].astype(str), drop_first=True)
    X_num = txns[numeric_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), 
                               columns=X_num.columns, 
                               index=X_num.index)
    X = pd.concat([X_num_scaled, X_cat], axis=1)
    y = txns['is_fraud'].astype(int).values
    return X, y

@st.cache_data
def train_and_evaluate(X, y, models_to_run=None, seed=SEED):
    model_mapping = {
        'Logistic Regression': 'LogReg',
        'Random Forest': 'RF',
        'SVM': 'SVM',
        'KNN': 'KNN',
        'XGBoost': 'XGB'
    }
    
    if models_to_run is None:
        models_to_run = ['LogReg', 'RF', 'SVM', 'KNN'] + (['XGB'] if HAVE_XGB else [])
    else:
        models_to_run = [model_mapping[m] for m in models_to_run if m in model_mapping]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      test_size=0.2, 
                                                      random_state=seed, 
                                                      stratify=y)
    
    models = {
        'LogReg': LogisticRegression(max_iter=200, 
                                   class_weight='balanced', 
                                   random_state=seed),
        'RF': RandomForestClassifier(n_estimators=200, 
                                   class_weight='balanced_subsample', 
                                   random_state=seed, 
                                   n_jobs=-1),
        'SVM': SVC(kernel='linear', 
                  probability=True, 
                  class_weight='balanced', 
                  random_state=seed),
        'KNN': KNeighborsClassifier(n_neighbors=15)
    }
    
    if HAVE_XGB:
        models['XGB'] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            scale_pos_weight=max(1.0, (y_train==0).sum() / max(1, (y_train==1).sum()))
        )
    
    results = []
    figs = {}
    roc_data = {}
    
    for name, model in models.items():
        if name not in models_to_run:
            continue
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            metrics['ROC AUC'] = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[name] = {'fpr': fpr, 'tpr': tpr}
        
        results.append(metrics)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax1, cmap='Blues')
        ax1.set_title(f'Confusion Matrix - {name}')
        
        if y_proba is not None:
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax2)
            ax2.set_title(f'ROC Curve - {name}')
            
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            ax3.plot(rec, prec)
            ax3.set_title(f'Precision-Recall Curve - {name}')
            ax3.set_xlabel('Recall')
            ax3.set_ylabel('Precision')
        
        figs[name] = fig
        plt.close(fig)

    results_df = pd.DataFrame(results)
    return results_df, figs, roc_data, models

def enhanced_model_comparison(results_df):
    """Interactive model comparison with table and parallel coordinates"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Model Performance Summary")
        display_df = results_df.copy()
        display_df['Best'] = display_df.idxmax(axis=0)
        st.dataframe(
            display_df.style
            .highlight_max(axis=0, color='#2ecc71')
            .format("{:.3f}"),
            use_container_width=True
        )
    
    with col2:
        fig = px.parallel_coordinates(
            results_df,
            color='ROC AUC',
            dimensions=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'],
            color_continuous_scale=px.colors.diverging.Tealrose,
            title='Model Performance Comparison'
        )
        st.plotly_chart(fig, use_container_width=True)

def detailed_feature_analysis(model, X, y, model_name):
    """Comprehensive feature analysis with multiple tabs"""
    tabs = st.tabs(["📊 Importance", "📈 Partial Dependence", "🔍 Interactions"])
    
    with tabs[0]:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(20)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'{model_name} - Top 20 Features'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance not available for this model type")
    
    with tabs[1]:
        st.info("Partial dependence plots would show how each feature affects predictions")
        # Implementation would use sklearn.inspection.partial_dependence
    
    with tabs[2]:
        st.info("Feature interaction analysis would show combined effects")
        # Implementation would use SHAP or similar

# Streamlit UI
st.set_page_config(
    page_title="Fraud Detection Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stApp > div, .stButton > button, .stMetric, .stDataFrame, .stPlotlyChart {
        animation: fadeIn 0.6s ease-out;
    }
    .stMetric:hover, .stButton>button:hover {
        transform: translateY(-2px);
        transition: all 0.2s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2c3e50, #1a1a2e);
        color: white;
        padding: 2rem 1.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .dataframe tbody tr:hover {
        background-color: #f8f9fa !important;
        transform: scale(1.01);
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🔍 Fraud Detection Pro")
    st.markdown("---")
    
    with st.expander("⚙️ Configuration", expanded=True):
        n_txns = st.number_input(
            "Number of Transactions",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000
        )
        target_fraud = st.slider(
            "Target Fraud Rate (%)",
            0.1, 10.0, 2.5, 0.1
        )
        seed = st.number_input(
            "Random Seed",
            value=SEED,
            step=1
        )
    
    with st.expander("🤖 Model Selection"):
        base_models = ['Logistic Regression', 'Random Forest', 'SVM', 'KNN']
        
        if HAVE_XGB:
            available_models = base_models + ['XGBoost']
            default_models = ['Logistic Regression', 'Random Forest', 'XGBoost']
        else:
            available_models = base_models
            default_models = ['Logistic Regression', 'Random Forest']
            st.markdown(
                '<div style="background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;">'
                '⚠️ XGBoost not installed. <code>pip install xgboost</code>'
                '</div>',
                unsafe_allow_html=True
            )
        
        models_selected = st.multiselect(
            "Models to Run",
            available_models,
            default=default_models
        )
    
    run_btn = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True
    )

st.title("💳 Fraud Detection Dashboard")
st.caption("Advanced synthetic transaction analysis system")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔍 Exploration", "🤖 Models", "📤 Export", "⚙️ Settings"])

if run_btn:
    cfg = {
        "N_CUSTOMERS": 2000,
        "N_MERCHANTS": 800,
        "N_TXNS": int(n_txns),
        "TARGET_FRAUD_RATE": float(target_fraud)/100,
        "START_DATE": "2025-01-01",
        "DAYS": 45
    }
    out = generate_transactions(config=cfg, seed=int(seed))
    txns = out['txns']
    
    X, y = prepare_features(txns)
    
    results_df, figs, roc_data, models = train_and_evaluate(X, y, models_to_run=models_selected, seed=int(seed))
    
    # Dashboard Tab
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", f"{len(txns):,}")
        with col2:
            st.metric("Fraud Cases", 
                     f"{txns['is_fraud'].sum():,}",
                     f"{txns['is_fraud'].mean()*100:.2f}%")
        with col3:
            st.metric("Avg Amount", f"${txns['amount'].mean():.2f}")
        
        st.divider()
        
        # Data Preview and Download
        with st.expander("📁 Transaction Data Preview & Download", expanded=True):
            st.dataframe(txns.head(100), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                export_format = st.selectbox(
                    "Export Format",
                    ["CSV", "Excel", "JSON"],
                    key="main_export"
                )
            with col2:
                st.download_button(
                    "💾 Download Full Dataset",
                    data=txns.to_csv(index=False) if export_format == "CSV" else 
                         txns.to_excel(index=False) if export_format == "Excel" else 
                         txns.to_json(indent=2),
                    file_name=f"fraud_data_{datetime.now().strftime('%Y%m%d')}.{export_format.lower()}",
                    mime="text/csv" if export_format == "CSV" else 
                         "application/vnd.ms-excel" if export_format == "Excel" else 
                         "application/json"
                )
        
        st.divider()
        
        # Interactive EDA
        st.subheader("Interactive Data Exploration")
        eda_figs = plot_interactive_eda(txns)
        for fig in eda_figs.values():
            st.plotly_chart(fig, use_container_width=True)

    # Exploration Tab
    with tab2:
        st.subheader("Transaction Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            amount_filter = st.slider(
                "Filter by Amount",
                float(txns['amount'].min()),
                float(txns['amount'].max()),
                (float(txns['amount'].min()), float(txns['amount'].max()))
            )
        with col2:
            fraud_filter = st.selectbox(
                "Filter by Fraud Status",
                ["All", "Fraud Only", "Legitimate Only"]
            )
        
        filtered = txns[
            (txns['amount'] >= amount_filter[0]) & 
            (txns['amount'] <= amount_filter[1])
        ]
        if fraud_filter == "Fraud Only":
            filtered = filtered[filtered['is_fraud'] == 1]
        elif fraud_filter == "Legitimate Only":
            filtered = filtered[filtered['is_fraud'] == 0]
        
        st.dataframe(filtered, use_container_width=True)
        
        st.plotly_chart(px.histogram(filtered, x='amount', color='is_fraud',
                       title='Amount Distribution by Fraud Status'), use_container_width=True)
        
        st.plotly_chart(px.scatter(filtered, x='hour', y='amount', color='is_fraud',
                       title='Transactions by Hour and Amount'), use_container_width=True)
    
    # Models Tab
    with tab3:
        st.subheader("Enhanced Model Evaluation")
        enhanced_model_comparison(results_df)
        
        for model_name in results_df['Model']:
            with st.expander(f"🔍 Detailed Analysis: {model_name}", expanded=False):
                detailed_feature_analysis(models[model_name], X, y, model_name)
                
                if model_name in figs:
                    st.pyplot(figs[model_name])
    
    # Export Tab
    with tab4:
        st.subheader("Data Export")
        
        export_format = st.radio(
            "Export Format",
            ["CSV", "Excel", "JSON"],
            horizontal=True
        )
        
        if st.button("💾 Export Transaction Data"):
            buffer = io.BytesIO()
            if export_format == "CSV":
                txns.to_csv(buffer, index=False)
                st.download_button(
                    label="Download CSV",
                    data=buffer,
                    file_name="fraud_transactions.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                with pd.ExcelWriter(buffer) as writer:
                    txns.to_excel(writer, index=False)
                st.download_button(
                    label="Download Excel",
                    data=buffer,
                    file_name="fraud_transactions.xlsx",
                    mime="application/vnd.ms-excel"
                )
            elif export_format == "JSON":
                txns.to_json(buffer, orient="records")
                st.download_button(
                    label="Download JSON",
                    data=buffer,
                    file_name="fraud_transactions.json",
                    mime="application/json"
                )
    
    # Settings Tab
    with tab5:
        st.subheader("Advanced Settings")
        st.write("Configuration options coming soon...")

else:
    # Welcome message (Dashboard tab)
    with tab1:
        st.info("Configure parameters in the sidebar and click 'Run Analysis' to begin.")
        
        with st.expander("📌 Quick Start Guide", expanded=True):
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px;">
                <h3 style="margin-top:0;">Getting Started</h3>
                <ol>
                    <li>Set transaction volume and fraud rate</li>
                    <li>Select machine learning models</li>
                    <li>Click "Run Analysis" to generate results</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.info("Run analysis to explore transaction data")
    
    with tab3:
        st.info("Run analysis to view model performance")
    
    with tab4:
        st.info("Run analysis to enable data export")
    
    with tab5:
        st.info("Run analysis to access advanced settings")

st.markdown("---")
st.caption("© 2025 Fraud Detection Pro | v2.2 | Enhanced Analytics")
