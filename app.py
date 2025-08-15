import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fraud_pipeline import (
    generate_transactions,
    plot_eda,
    prepare_features,
    train_and_evaluate,
    HAVE_XGB,
    SEED
)

# Page Configuration
st.set_page_config(
    page_title="Fraud Detection Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 15px;
    }
    .st-eb {
        background-color: #f0f2f6 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50;
        color: white;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
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
    
    # Model selection
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

# Main Dashboard
st.title("Credit Card Fraud Detection")
st.caption("Advanced synthetic transaction analysis system")

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
    
    # Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Exploration", "🤖 Models", "📥 Export"])
    
    with tab1:
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        fraud_rate = txns['is_fraud'].mean() * 100
        col1.metric("Total Transactions", f"{len(txns):,}")
        col2.metric("Fraud Cases", f"{txns['is_fraud'].sum():,}", f"{fraud_rate:.2f}%")
        col3.metric("Average Amount", f"${txns['amount'].mean():.2f}")
        
        # Fraud Over Time
        st.subheader("Fraud Timeline")
        hourly_fraud = txns.groupby(txns['timestamp'].dt.hour)['is_fraud'].mean().reset_index()
        fig = px.area(
            hourly_fraud,
            x='timestamp',
            y='is_fraud',
            labels={'timestamp': 'Hour of Day', 'is_fraud': 'Fraud Rate'},
            color_discrete_sequence=['#e74c3c']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_tickformat=".1%",
            hovermode="x"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # MCC Analysis
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
    
    with tab2:
        # Amount Distribution
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
        
        # Geographic Heatmap
        st.subheader("Fraud Geographic Distribution")
        sample_txns = txns.sample(min(5000, len(txns)))
        fig = px.density_mapbox(
            sample_txns,
            lat='home_lat',
            lon='home_lon',
            z='is_fraud',
            radius=10,
            center=dict(lat=30, lon=0),
            zoom=1,
            mapbox_style="carto-positron",
            color_continuous_scale='Reds',
            title='Fraud Density Map'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Model Performance
        st.subheader("Model Comparison")
        
        # Metrics Table
        st.markdown("##### Evaluation Metrics")
        metrics_df = results_df.copy()
        metrics_df['ROC AUC'] = metrics_df['ROC AUC'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
        for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
        
        st.dataframe(
            metrics_df,
            column_config={
                "Model": st.column_config.TextColumn("Model"),
                "Accuracy": st.column_config.TextColumn("Accuracy"),
                "Precision": st.column_config.TextColumn("Precision"),
                "Recall": st.column_config.TextColumn("Recall"),
                "F1": st.column_config.TextColumn("F1 Score"),
                "ROC AUC": st.column_config.TextColumn("ROC AUC")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # ROC Curves
        if len(roc_data) > 0:
            st.markdown("##### ROC Curves")
            roc_fig = go.Figure()
            roc_fig.add_shape(type='line', line=dict(dash='dash'),
                            x0=0, x1=1, y0=0, y1=1)
            
            for model_name in roc_data:
                roc_fig.add_trace(go.Scatter(
                    x=roc_data[model_name]['fpr'],
                    y=roc_data[model_name]['tpr'],
                    name=f'{model_name} (AUC = {results_df[results_df["Model"]==model_name]["ROC AUC"].values[0]:.3f})',
                    mode='lines'
                ))
            
            roc_fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(roc_fig, use_container_width=True)
        
        # Model Evaluation Plots
        model_mapping = {
            'Logistic Regression': 'LogReg',
            'Random Forest': 'RF',
            'SVM': 'SVM',
            'KNN': 'KNN',
            'XGBoost': 'XGB'
        }
        
        for model_name in models_selected:
            internal_name = model_mapping.get(model_name)
            if internal_name in figs:
                st.markdown(f"### {model_name} Evaluation")
                st.pyplot(figs[internal_name])
    
    with tab4:
        # Data Export
        st.subheader("Export Results")
        
        with st.expander("📋 Transaction Data Sample"):
            st.dataframe(txns.head(1000), use_container_width=True)
        
        col1, col2 = st.columns(2)
        csv = txns.to_csv(index=False).encode('utf-8')
        col1.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name='fraud_transactions.csv',
            mime='text/csv',
            use_container_width=True
        )
        
        json = txns.to_json(orient='records')
        col2.download_button(
            label="📥 Download JSON",
            data=json,
            file_name='fraud_transactions.json',
            mime='application/json',
            use_container_width=True
        )

else:
    st.info("Configure parameters in the sidebar and click 'Run Analysis' to begin.")
    
    with st.expander("📌 Quick Start Guide", expanded=True):
        st.markdown("""
        **1. Data Configuration**
        - Start with 5K-10K transactions for quick testing
        - Default 2.5% fraud rate matches industry averages
        
        **2. Model Selection**
        - Logistic Regression: Fast baseline
        - Random Forest: Best balance of speed/accuracy
        - XGBoost: Highest accuracy (requires installation)
        
        **3. Interpretation**
        - Check ROC curves for model discrimination
        - Review feature importance for insights
        - Export data for further analysis
        """)

# Footer
st.markdown("---")
st.caption("© 2023 Fraud Detection Pro | v2.0")
