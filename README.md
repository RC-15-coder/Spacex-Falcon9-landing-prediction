# SpaceX Falcon 9 First Stage Landing Prediction

This project predicts the landing outcome (success/failure) of SpaceX Falcon 9 first-stage boosters. It covers real-world data collection, cleaning, EDA, interactive mapping, dashboarding, and ML classification models.

## Repository Contents

- **jupyter-labs-spacex-data-collection-api.ipynb**  
  *Collects SpaceX launch data via API, merging with Wikipedia data for comprehensive features.*

- **jupyter-labs-webscraping.ipynb**  
  *Scrapes web data for SpaceX payloads and booster recovery details.*

- **labs-jupyter-spacex-Data wrangling.ipynb**  
  *Cleans, merges, and prepares the data for analysis.*

- **jupyter-labs-eda-sql-coursera_sqlite (2).ipynb**  
  *Performs exploratory data analysis (EDA) and SQL-based analysis to uncover trends and correlations.*

- **edadataviz.ipynb**  
  *Creates visualizations to understand launch outcomes by site, orbit, and payload mass.*

- **lab_jupyter_launch_site_location (1).ipynb**  
  *Maps launch sites using Folium to visualize site locations and nearby infrastructure.*

- **Lab Dashboard Application with Plotly Dash.pdf**  
  *Interactive dashboard for exploring launch outcomes and success ratios by site and payload.*

- **SpaceX_Machine Learning Prediction_Part_5.ipynb**  
  *Builds and evaluates machine learning models (Logistic Regression, SVM, Decision Tree, KNN) to predict landing success. Includes model comparison, accuracy, and confusion matrix.*

  ## Project Highlights

- **End‑to‑end pipeline:** Data collection (SpaceX API + Wikipedia scraping) → data wrangling → EDA (plots & SQL) → interactive Folium maps → Plotly Dash dashboard → ML model training & evaluation.
- Created a unified, clean dataset of Falcon 9 launches, outcomes, and site/orbit/payload details.
- Built dashboards and maps to enable rapid exploration and visualization of launch trends.
- Trained and compared multiple classification models (Logistic Regression, SVM, Decision Tree, KNN) to predict first‑stage landing outcomes.

## Key Findings

- **Top‑performing launch sites:** In this dataset, **CCAFS SLC‑40** and **KSC LC‑39A** consistently achieved the highest first‑stage landing success rates.
- **Payload mass effect:** Most landing successes cluster below **~6,000 kg**; heavier payloads show more variable outcomes.
- **Orbit impact:** **SSO (polar), GEO, and HEO** orbits show near‑perfect recovery rates in this sample, while **GTO** lags at ~50%.
- **Best model:** The **Decision Tree** classifier achieved the highest **cross‑validated** accuracy (≈ **0.89**). On the **hold‑out test** set, all models achieved **~0.83** accuracy.

## 📄 Full Project Report

A comprehensive slide deck walks through every step of this project, from data collection and wrangling all the way through EDA, interactive maps, dashboards, and machine-learning results.  

👉 [SpaceX Falcon 9 Landing Prediction – Full Report](SpaceX%20Falcon%209%20Landing%20Prediction%20%E2%80%93%20Full%20Report.pdf)

The report includes:
- Data collection & wrangling flow
- Visual EDA (plots + SQL results)
- Interactive analytics (Folium + Plotly Dash)
- Predictive modeling (LogReg, SVM, Decision Tree, KNN), hyperparameter tuning, accuracy, and confusion matrix
- Conclusions and next steps

