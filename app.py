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

# ---------------------
# Import pipeline functions
# ---------------------
from fraud_pipeline import generate_transactions, prepare_features, train_and_evaluate

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(
    page_title="Fraud Detection Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Credit Card Fraud Detection")
st.caption("Advanced synthetic transaction analysis system")

with st.sidebar:
    st.title("Fraud Detection Dashboard")
    st.markdown("### Configuration")
    
    n_txns = st.number_input(
        "Number of Transactions",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        help="Lower values will process faster"
    )
    target_fraud = st.slider(
        "Target Fraud Rate (%)",
        0.1, 10.0, 2.5, 0.1,
        help="Percentage of fraudulent transactions to generate"
    )
    seed = st.number_input(
        "Random Seed",
        value=SEED,
        step=1,
        help="For reproducible results"
    )
    
    with st.expander("Model Selection"):
        base_models = ['Logistic Regression', 'Random Forest', 'SVM', 'KNN']
        
        if HAVE_XGB:
            available_models = base_models + ['XGBoost']
            default_models = ['Logistic Regression', 'Random Forest', 'XGBoost']
        else:
            available_models = base_models
            default_models = ['Logistic Regression', 'Random Forest']
            st.markdown(
                '<div class="warning-box">XGBoost not installed - some models unavailable<br>'
                'Install with: <code>pip install xgboost</code></div>',
                unsafe_allow_html=True
            )
        
        models_selected = st.multiselect(
            "Select Models to Run",
            available_models,
            default=default_models,
            help="Select which models to train and evaluate"
        )
    
    run_btn = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True
    )

# ---------------------
# Main App Logic
# ---------------------
if run_btn:
    with st.status("🔍 Processing...", expanded=True) as status:
        st.write("📊 Generating synthetic transactions...")
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
        
        st.write("⚙️ Engineering features...")
        X, y = prepare_features(txns)
        
        st.write("🤖 Training models...")
        results_df, figs, roc_data = train_and_evaluate(X, y, models_to_run=models_selected, seed=int(seed))
        
        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
    
    # Add new Data tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 Exploration", "🤖 Models", "📥 Export", "📄 Data"])
    
    # ----- Overview -----
    with tab1:
        col1, col2, col3 = st.columns(3)
        fraud_rate = txns['is_fraud'].mean() * 100
        col1.metric("Total Transactions", f"{len(txns):,}")
        col2.metric("Fraud Cases", f"{txns['is_fraud'].sum():,}", f"{fraud_rate:.2f}%")
        col3.metric("Average Amount", f"${txns['amount'].mean():.2f}")
        
        st.subheader("Fraud by Merchant Category")
        mcc_fraud = txns.groupby('mcc')['is_fraud'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(
            mcc_fraud.reset_index(),
            x='mcc',
            y='is_fraud',
            color='is_fraud',
            color_continuous_scale='Reds',
            labels={'mcc': 'MCC Code', 'is_fraud': 'Fraud Rate'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ----- Exploration -----
    with tab2:
        st.subheader("Transaction Amount Distribution")
        fig = px.box(
            txns,
            x='is_fraud',
            y='amount',
            color='is_fraud',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            log_y=True,
            labels={'is_fraud': 'Fraud Status', 'amount': 'Amount (log scale)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ----- Models -----
    with tab3:
        st.subheader("Model Comparison")
        st.dataframe(results_df, use_container_width=True)
    
    # ----- Export -----
    with tab4:
        st.subheader("Export Results")
        with st.expander("📋 Transaction Data Sample"):
            st.dataframe(txns.head(1000), use_container_width=True)
        
        csv = txns.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "fraud_transactions.csv", "text/csv")
    
    # ----- Data Tab (Enhancement) -----
    with tab5:
        st.subheader("Generated Synthetic Dataset")
        st.write(f"Showing {len(txns):,} transactions")

        # --- Filters ---
        col1, col2, col3 = st.columns(3)

        fraud_filter = col1.selectbox("Fraud Status", ["All", "Fraud Only", "Legit Only"])
        region_filter = col2.multiselect(
            "Customer Region",
            txns["home_region"].unique().tolist(),
            default=txns["home_region"].unique().tolist()
        )
        mcc_filter = col3.multiselect(
            "Merchant Category (MCC)",
            txns["mcc"].unique().tolist(),
            default=txns["mcc"].unique().tolist()
        )

        # --- Search ---
        search_term = st.text_input(
            "🔎 Search by Transaction ID, Customer ID, or Merchant ID",
            value="",
            placeholder="Enter ID..."
        )

        # --- Apply filters ---
        filtered_txns = txns.copy()
        if fraud_filter == "Fraud Only":
            filtered_txns = filtered_txns[filtered_txns["is_fraud"] == 1]
        elif fraud_filter == "Legit Only":
            filtered_txns = filtered_txns[filtered_txns["is_fraud"] == 0]

        filtered_txns = filtered_txns[filtered_txns["home_region"].isin(region_filter)]
        filtered_txns = filtered_txns[filtered_txns["mcc"].isin(mcc_filter)]

        if search_term.strip():
            filtered_txns = filtered_txns[
                filtered_txns["transaction_id"].str.contains(search_term, case=False, na=False) |
                filtered_txns["customer_id"].str.contains(search_term, case=False, na=False) |
                filtered_txns["merchant_id"].str.contains(search_term, case=False, na=False)
            ]

        # --- Show Data ---
        if len(filtered_txns) > 10000:
            st.warning(f"Large dataset detected! Showing first 10,000 of {len(filtered_txns):,} rows.")
            st.dataframe(filtered_txns.head(10000), use_container_width=True)
        else:
            st.dataframe(filtered_txns, use_container_width=True)

        # --- Download filtered dataset ---
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=filtered_txns.to_csv(index=False).encode("utf-8"),
            file_name="filtered_transactions.csv",
            mime="text/csv"
        )

else:
    st.info("Configure parameters in the sidebar and click 'Run Analysis' to begin.")
