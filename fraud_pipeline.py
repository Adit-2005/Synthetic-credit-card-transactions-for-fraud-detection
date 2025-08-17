import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import random
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
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

DEFAULT_CONFIG = {
    "N_CUSTOMERS": 2000,
    "N_MERCHANTS": 800,
    "N_TXNS": 10000,
    "START_DATE": "2025-01-01",
    "DAYS": 45,
    "TARGET_FRAUD_RATE": 0.025
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
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def weighted_choice(choices, weights):
    cum = np.cumsum(weights)
    r = random.random() * cum[-1]
    return choices[np.searchsorted(cum, r)]

def generate_transactions(config=None, seed=SEED):
    # Merge with default config
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        config = {**DEFAULT_CONFIG, **config}
    
    # Ensure START_DATE is datetime
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

def plot_eda(txns):
    figs = {}
    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # Plot 1: Fraud Distribution
    ax1 = plt.subplot(gs[0, 0])
    fraud_counts = txns['is_fraud'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    ax1.pie(fraud_counts, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%', 
            colors=colors, startangle=90, wedgeprops=dict(width=0.4))
    ax1.set_title('Transaction Fraud Distribution', fontsize=14, pad=20)
    
    # Plot 2: Amount Distribution
    ax2 = plt.subplot(gs[0, 1])
    for label, grp in txns.groupby('is_fraud'):
        sns.kdeplot(np.log1p(grp['amount']), ax=ax2, 
                   label='Fraud' if label else 'Legitimate', 
                   linewidth=2)
    ax2.set_xlabel('Log(1+Amount)')
    ax2.set_ylabel('Density')
    ax2.set_title('Transaction Amount Distribution by Fraud Status')
    ax2.legend()
    
    # Plot 3: Hourly Fraud Rate
    ax3 = plt.subplot(gs[1, 0])
    hourly_fraud = txns.groupby('hour')['is_fraud'].mean()
    ax3.plot(hourly_fraud.index, hourly_fraud.values, 
            marker='o', color='#3498db', linewidth=2)
    ax3.fill_between(hourly_fraud.index, hourly_fraud.values, 
                    color='#3498db', alpha=0.2)
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Fraud Rate')
    ax3.set_title('Fraud Rate by Hour of Day')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: MCC Fraud Rate
    ax4 = plt.subplot(gs[1, 1])
    mcc_fraud = txns.groupby('mcc')['is_fraud'].mean().sort_values(ascending=False)
    mcc_fraud.plot(kind='bar', color='#9b59b6', ax=ax4)
    ax4.set_xlabel('Merchant Category Code')
    ax4.set_ylabel('Fraud Rate')
    ax4.set_title('Fraud Rate by Merchant Category')
    ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Geographic Distribution
    ax5 = plt.subplot(gs[2, :])
    fraud_locations = txns[txns['is_fraud'] == 1]
    legit_locations = txns[txns['is_fraud'] == 0].sample(n=min(10000, len(txns[txns['is_fraud'] == 0])))
    ax5.scatter(legit_locations['home_lon'], legit_locations['home_lat'], 
               color='#2ecc71', alpha=0.3, label='Legitimate', s=10)
    ax5.scatter(fraud_locations['home_lon'], fraud_locations['home_lat'], 
               color='#e74c3c', alpha=0.7, label='Fraud', s=20)
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')
    ax5.set_title('Geographic Distribution of Fraudulent Transactions')
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    figs['overview'] = fig
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
            'ROC AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
        results.append(metrics)

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[name] = {'fpr': fpr, 'tpr': tpr}
        
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
    return results_df, figs, roc_data
