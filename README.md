# 🔍 Fraud Detection Pro

Fraud Detection Pro is a **Streamlit-powered app** that generates synthetic financial transaction data, performs exploratory fraud analytics, trains ML models, and allows exporting datasets — all with a **modern, professional UI**.

## ✨ Features

- **💻 Modern UI/UX**
  - Sidebar navigation (Dashboard • Explorer • Models • Export)
  - Glassmorphism metric cards with animations
  - Fraud ticker banner and stylish visuals

- **📊 Dashboard**
  - KPIs (Total Transactions, Fraud Cases, Avg Amount)
  - Fraud distribution pie & violin plots
  - Hourly fraud trend, MCC treemap, Geo scatter
  - Expandable raw data preview

- **🔍 Explorer**
  - Interactive filters (Amount, Fraud Status, Region)
  - Data table with live filters
  - Histograms & scatter plots for deep dive

- **🤖 Models**
  - Train multiple ML models: Logistic Regression, Random Forest, SVM, KNN, (XGBoost optional)
  - Metrics table (Accuracy, Precision, Recall, F1, ROC AUC)
  - Animated gauge charts for Precision/Recall/F1
  - Overlay ROC curves for comparison

- **📤 Export**
  - Download dataset in **CSV, JSON, or Excel**
  - Preview top 100 rows inline

---

## 🚀 Quick Start

### 1️⃣ Clone and Install
```bash
git clone https://github.com/your-repo/fraud-detection-pro.git
cd fraud-detection-pro
pip install -r requirements.txt
```

### 2️⃣ (Optional) Install Excel export support
```bash
pip install xlsxwriter
```

### 3️⃣ Run the App
```bash
streamlit run app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).

---

## ⚙️ Configuration
- **Transactions:** 1,000 – 50,000
- **Fraud Rate:** 0.1% – 10%
- **Models:** Select models to train, or leave empty to just generate/export data.

---

## 📂 Project Structure
```
fraud-detection-pro/
│── app.py              # Main Streamlit app (Enhanced UI)
│── fraud_pipeline.py   # Data generation & feature pipeline
│── requirements.txt    # Python dependencies
│── README.md           # Project overview (this file)
│── devcontainer.json   # Dev container config
```

---

## 📸 Screenshots

### Dashboard
![Dashboard Screenshot](assets/dashboard.png)

### Explorer
![Explorer Screenshot](assets/explorer.png)

### Models
![Models Screenshot](assets/models.png)

### Export
![Export Screenshot](assets/export.png)

(*Screenshots are placeholders — capture them after running locally*)

---

## 🌐 Deployment

You can deploy Fraud Detection Pro easily:

### ▶️ Streamlit Cloud
1. Push your repo to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect your GitHub repo and deploy.
4. Done! Your app will be available at `https://your-app.streamlit.app`.

### 🐳 Docker
```bash
docker build -t fraud-detection-pro .
docker run -p 8501:8501 fraud-detection-pro
```
Access at [http://localhost:8501](http://localhost:8501).

### 🤗 Hugging Face Spaces
1. Create a new Space (type: **Streamlit**).
2. Upload your project files (or link GitHub).
3. The app will auto-build and run in the browser.

---

## 🛠️ Tech Stack
- **Frontend/UI:** Streamlit + Plotly
- **ML Models:** scikit-learn, (XGBoost optional)
- **Data:** Synthetic generator with realistic fraud injection
- **Export:** CSV, JSON, Excel

---

## 📜 License
MIT License © 2025 Fraud Detection Pro
