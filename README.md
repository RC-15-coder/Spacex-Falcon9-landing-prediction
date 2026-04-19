# 🚀 Falcon 9 Mission Intelligence

[![Live Website](https://img.shields.io/badge/Live%20Website-Launch-8b5cf6?style=for-the-badge&logo=streamlit&logoColor=white)](https://falcon9-mission-intelligence.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

An end-to-end machine learning pipeline that predicts SpaceX Falcon 9 first-stage booster landings before liftoff — achieving 94.4% test accuracy with LightGBM and full SHAP explainability.

This project tackles a real-world challenge faced by SpaceX: Can we predict whether a Falcon 9 first-stage booster will land successfully, before launch, using historical mission data?

Leveraging data from the SpaceX API and public sources, I built an end-to-end data science pipeline—from automated data collection and rigorous cleaning, to in-depth exploratory analysis, interactive mapping, dashboard creation, and machine learning modeling.


🌐 **[Try the Live Website →](https://falcon9-mission-intelligence.streamlit.app)**

---

## 💰 The $103M Question

SpaceX launches Falcon 9 at ~$62M per flight while competitors charge $165M+. That entire **$103M cost advantage** comes from one thing — recovering and reusing the first-stage booster. This pipeline predicts landing success *before* launch, enabling launch providers to price insurance risk accurately, set competitive bids, and optimize mission planning. Every percentage point of prediction accuracy translates directly to financial impact.

---

## 🎯 Project Highlights

- **94.4% test accuracy** with LightGBM — a 27-point improvement over the naive baseline
- **6 classifiers compared** — Logistic Regression, SVM, Decision Tree, KNN, LightGBM, XGBoost, all tuned with 10-fold GridSearchCV
- **83 engineered features** built from 90 real SpaceX missions via one-hot encoding of orbit, launch site, landing pad, and booster serial
- **SHAP explainability** — global feature importance, beeswarm plots, and per-prediction waterfall charts showing exactly why each prediction was made
- **Interactive Streamlit dashboard** with live prediction engine, Folium launch site map, and skeleton loading UI
- **Production inference pipeline** — joblib-cached models load in under 1 second on subsequent runs

---

## 🛠 Tech Stack

**ML & Data:** Python · Pandas · NumPy · Scikit-learn · LightGBM · XGBoost · SHAP  
**Visualization:** Matplotlib · Seaborn · Folium · Streamlit  
**Data Collection:** SpaceX REST API · BeautifulSoup · SQLite  
**Deployment:** Streamlit Community Cloud · GitHub · joblib  

---

## 📂 Repository Structure

```text
Spacex-Falcon9-landing-prediction/
├── app.py                          # Streamlit dashboard (5 pages, production-ready)
├── requirements.txt                # Pinned dependencies for reproducibility
├── data/
│   ├── dataset_part_2.csv          # Cleaned mission metadata
│   └── dataset_part_3.csv          # Feature matrix (83 one-hot encoded columns)
├── notebooks/
│   ├── 01_data_collection.ipynb    # SpaceX REST API ingestion
│   ├── 02_webscraping.ipynb        # BeautifulSoup Wikipedia scraping
│   ├── 03_data_wrangling.ipynb     # Null handling, merging, target creation
│   ├── 04_eda_sql.ipynb            # SQLite-based exploratory analysis
│   ├── 05_eda_viz.ipynb            # Matplotlib / Seaborn EDA
│   ├── 06_launch_site_maps.ipynb   # Folium geospatial analysis
│   └── 07_ml_modeling.ipynb        # 6-model comparison + SHAP
└── reports/
    ├── model_comparison.png
    ├── shap_beeswarm.png
    └── shap_feature_importance.png
```

---

## 🔑 Key Findings

| Finding | Insight |
| --- | --- |
| **Best model** | LightGBM — 94.4% test accuracy, 27 points above naive baseline |
| **Top features** | Orbit type, payload mass, booster flight count dominate predictions |
| **Best launch site** | KSC LC-39A leads in recovery rate (near 100% on recent missions) |
| **Payload threshold** | Above ~6,000 kg, outcomes become fuel-margin constrained |
| **Orbit impact** | LEO/ISS/SSO → high recovery (>75%); GTO → ~50% (fuel-limited) |

---

## 📊 Dashboard Features

The live website delivers five interactive pages:

1. **Overview** — executive summary with business framing and key metrics
2. **Data Explorer** — interactive Folium map, payload/orbit distribution analysis
3. **Model Performance** — ranked leaderboard, confusion matrices, precision/recall breakdown
4. **SHAP Explainability** — global feature importance and beeswarm plots
5. **Landing Predictor** — configure a live mission and see the probability + SHAP waterfall for your specific input

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/RC-15-coder/Spacex-Falcon9-landing-prediction.git
cd Spacex-Falcon9-landing-prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`. First launch trains and caches the 6 ML models (~45 seconds); subsequent launches are instant.

