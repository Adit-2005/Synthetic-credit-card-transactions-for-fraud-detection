import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import uuid
import random
import math
from io import BytesIO
import time
import os
import gc
import signal
from contextlib import contextmanager
import psutil
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

# Configure pandas to prevent future warnings
pd.set_option('future.no_silent_downcasting', True)

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

# Cloud-optimized configuration
DEFAULT_CONFIG = {
    "N_CUSTOMERS": 500 if os.environ.get('IS_STREAMLIT_CLOUD') else 2000,
    "N_MERCHANTS": 200 if os.environ.get('IS_STREAMLIT_CLOUD') else 800,
    "N_TXNS": 2000 if os.environ.get('IS_STREAMLIT_CLOUD') else 10000,
    "START_DATE": "2025-01-01",
    "DAYS": 45,
    "TARGET_FRAUD_RATE": 0.025
}

# Initialize session state
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'initialized': False,
        'data': None,
        'models': None,
        'last_activity': time.time(),
        'cleanup_counter': 0
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

@st.cache_data(show_spinner=False)
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
    
    # Safe distance calculation
    def calc_km_from_prev(group, max_iterations=1000):
        distances = [0]
        if len(group) > 1:
            distances = []
            prev_lat, prev_lon = group.iloc[0]["home_lat"], group.iloc[0]["home_lon"]
            for i in range(1, min(len(group), max_iterations)):
                curr_lat, curr_lon = group.iloc[i]["m_lat"], group.iloc[i]["m_lon"]
                distances.append(haversine(prev_lat, prev_lon, curr_lat, curr_lon))
                prev_lat, prev_lon = curr_lat, curr_lon
            distances = [0] + distances
        return distances
    
    txns["km_from_prev"] = txns.groupby("customer_id").apply(lambda g: calc_km_from_prev(g)).explode().values

    # Transaction count features
    txns["txns_last_10m"] = txns.groupby("customer_id").rolling("10min", on="timestamp").count()["transaction_id"].values
    txns["txns_last_1h"] = txns.groupby("customer_id").rolling("1h", on="timestamp").count()["transaction_id"].values
    txns["txns_last_24h"] = txns.groupby("customer_id").rolling("24h", on="timestamp").count()["transaction_id"].values

    # Clean up memory
    gc.collect()
    
    return {
        "txns": txns,
        "customers": customers,
        "merchants": merchants
    }

# Timeout handler for model training
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

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
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    models = {
        'LogReg': LogisticRegression(max_iter=200, 
                                   class_weight='balanced', 
                                   random_state=seed),
        'RF': RandomForestClassifier(n_estimators=100,
                                   class_weight='balanced_subsample', 
                                   random_state=seed),
        'SVM': SVC(kernel='linear', 
                  probability=True, 
                  class_weight='balanced', 
                  random_state=seed),
        'KNN': KNeighborsClassifier(n_neighbors=15)
    }
    
    if HAVE_XGB and 'XGB' in models_to_run:
        if psutil.virtual_memory().percent < 80:  # Only use XGB if enough memory
            models['XGB'] = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                tree_method='hist',
                random_state=seed,
                scale_pos_weight=max(1.0, (y_train==0).sum() / max(1, (y_train==1).sum())
            )
        else:
            st.warning("Skipping XGBoost due to high memory usage")
            models_to_run.remove('XGB')
    
    results = []
    figs = {}
    roc_data = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        if name not in models_to_run:
            continue
            
        status_text.text(f"Training {name}... ({i+1}/{len(models_to_run)})")
        progress_bar.progress((i+1)/len(models_to_run))
        
        try:
            with time_limit(120):  # 2 minutes per model
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
                
        except TimeoutException:
            st.error(f"{name} training timed out - skipping")
            continue
        except Exception as e:
            st.error(f"Error training {name}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    gc.collect()
    
    results_df = pd.DataFrame(results)
    return results_df, figs, roc_data, models

# Streamlit UI setup
def main():
    st.set_page_config(
        page_title="Fraud Detection Pro",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .stApp > div, .stButton > button, .stMetric, .stDataFrame, .stPlotlyChart {
            animation: fadeIn 0.6s ease-out;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #2c3e50, #1a1a2e);
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Connection keep-alive
    if time.time() - st.session_state.app_state['last_activity'] > 30:
        st.session_state.app_state['last_activity'] = time.time()
        st.rerun()
    
    with st.sidebar:
        st.title("🔍 Fraud Detection Pro")
        st.markdown("---")
        
        with st.expander("⚙️ Configuration", expanded=True):
            n_txns = st.number_input(
                "Number of Transactions",
                min_value=500,
                max_value=50000,
                value=2000 if os.environ.get('IS_STREAMLIT_CLOUD') else 10000,
                step=500
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
            if os.environ.get('IS_STREAMLIT_CLOUD'):
                st.warning("Cloud mode: XGBoost may be disabled if memory is low")
            
            base_models = ['Logistic Regression', 'Random Forest', 'SVM', 'KNN']
            available_models = base_models + (['XGBoost'] if HAVE_XGB else [])
            
            models_selected = st.multiselect(
                "Models to Run",
                available_models,
                default=['Logistic Regression', 'Random Forest']
            )
        
        run_btn = st.button(
            "🚀 Generate & Analyze",
            type="primary",
            use_container_width=True
        )
    
    # Main app tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Exploration", "🤖 Models", "📤 Export"])
    
    if run_btn:
        try:
            cfg = {
                "N_CUSTOMERS": 500 if os.environ.get('IS_STREAMLIT_CLOUD') else 2000,
                "N_MERCHANTS": 200 if os.environ.get('IS_STREAMLIT_CLOUD') else 800,
                "N_TXNS": int(n_txns),
                "TARGET_FRAUD_RATE": float(target_fraud)/100,
                "START_DATE": "2025-01-01",
                "DAYS": 45
            }
            
            with st.spinner("Generating synthetic transactions..."):
                out = generate_transactions(config=cfg, seed=int(seed))
                st.session_state.app_state['data'] = out['txns']
                st.session_state.app_state['initialized'] = True
            
            if models_selected:
                with st.spinner("Preparing features..."):
                    X, y = prepare_features(st.session_state.app_state['data'])
                
                with st.spinner("Training models..."):
                    results_df, figs, roc_data, models = train_and_evaluate(
                        X, y, models_to_run=models_selected, seed=int(seed))
                    st.session_state.app_state['models'] = {
                        'results': results_df,
                        'figures': figs,
                        'roc_data': roc_data,
                        'models': models
                    }
            
            st.success("Analysis complete!")
            st.session_state.app_state['last_activity'] = time.time()
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.stop()
    
    # Tab content rendering
    if st.session_state.app_state['initialized']:
        txns = st.session_state.app_state['data']
        
        with tab1:
            # Dashboard content
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", f"{len(txns):,}")
            with col2:
                st.metric("Fraud Cases", f"{txns['is_fraud'].sum():,}")
            with col3:
                st.metric("Avg Amount", f"${txns['amount'].mean():.2f}")
            
            st.dataframe(txns.head(100))
            
            # Export options
            export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
            if export_format == "Excel":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    txns.to_excel(writer, index=False)
                st.download_button(
                    "Download Data",
                    data=buffer.getvalue(),
                    file_name="fraud_data.xlsx",
                    mime="application/vnd.ms-excel"
                )
            elif export_format == "CSV":
                st.download_button(
                    "Download Data",
                    data=txns.to_csv(index=False),
                    file_name="fraud_data.csv",
                    mime="text/csv"
                )
            else:
                st.download_button(
                    "Download Data",
                    data=txns.to_json(indent=2),
                    file_name="fraud_data.json",
                    mime="application/json"
                )
        
        # Other tabs would follow similar patterns...
        
    else:
        with tab1:
            st.info("Configure parameters and click 'Generate & Analyze' to begin")
    
    # Periodic cleanup
    if st.session_state.app_state['cleanup_counter'] % 10 == 0:
        gc.collect()
    st.session_state.app_state['cleanup_counter'] += 1

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.stop()
```
