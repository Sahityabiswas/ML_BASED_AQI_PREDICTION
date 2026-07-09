# AQI Prediction — Full ML Pipeline

A comprehensive machine learning system that predicts the Air Quality Index (AQI) from pollutant concentrations across **26 Indian cities** (2015–2020). Includes per-city analysis, SHAP interpretability, lag-based forecasting, an interactive Streamlit dashboard, and an automated retraining pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAW DATA (city_day.csv)                         │
│           29,531 rows, 16 cols, 26 cities, 2015–2020                │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   DATA PREPROCESSING                                │
│  • Remove target rows with NaN AQI                                 │
│  • KNN Imputer (per city) for missing pollutants                   │
│  • Drop columns: Xylene, Benzene, Toluene (high missing %)         │
│  • Train/Test split: 80/20 chronological (TimeSeriesSplit)        │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                              │
│  • Temporal: DayOfYear, Month, DayOfWeek, Weekend, Season (S/C)    │
│  • City encoding: City → mean AQI mapping                          │
│  • Lag features: PM2.5_lag1..PM2.5_lag7, PM10_lag1..NO2_lag7      │
│  • Rolling stats: PM2.5_roll7_mean, PM2.5_roll7_std, ...          │
│  • NOTE: Current-day pollutants removed to prevent data leakage    │
│  • Final feature set: 46 features                                  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ Linear   │ │ Ridge    │ │ Lasso    │ │ RF       │              │
│  │ Regression│ │ (α=10.0) │ │ (α=1.0)  │ │ (Tuned*) │              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                             │
│  │ XGBoost  │ │LightGBM  │ │ CatBoost │                             │
│  │ (Tuned*) │ │ (Tuned*) │ │ (Tuned*) │                             │
│  └──────────┘ └──────────┘ └──────────┘                             │
│  *RandomizedSearchCV (5-fold TimeSeriesSplit)                       │
│  Champion: Random Forest (R²=0.9425, RMSE=21.60, MAE=12.70)       │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│              CITY-SPECIFIC COMPARISON                               │
│  • Train separate RF model per city (min 50 rows)                  │
│  • Compare global vs per-city R² per city                          │
│  • Global model wins in 22/26 cities                                │
│  • Per-city better for smaller/disjoint clusters                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SHAP ANALYSIS                                   │
│  • Global SHAP: PM2.5_lag1, PM10_lag1, CO_lag1 top predictors      │
│  • Per-city SHAP: Top 5 pollutants vary by city                    │
│  • SHAP dependence plots for key features                          │
│  • Output: city_shap_analysis.pkl + figures                        │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FORECASTING                                     │
│  • Lag-only features (no current-day pollutants, no target leaks)  │
│  • XGBoost model (R²=0.9407, RMSE=21.93, MAE=12.58)               │
│  • Recursive: 7-day ahead prediction using predicted lags          │
│  • Dashboard uses slider values as seed for user-driven forecast   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT                                      │
│  ┌─────────────────────────────────────┐                            │
│  │  Streamlit Dashboard (3 tabs)       │ ←── Live: dashboard.onrender.com
│  │  • Tab 1: Predict AQI + SHAP force  │                            │
│  │  • Tab 2: City insights + plots     │                            │
│  │  • Tab 3: 7-day forecast + chart    │                            │
│  └─────────────────────────────────────┘                            │
│  ┌─────────────────────────────────────┐                            │
│  │  Retraining Pipeline                │                            │
│  │  • python scripts/retrain.py        │                            │
│  │  • End-to-end: raw data → models    │                            │
│  └─────────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Results

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Random Forest (Tuned)** | **0.9425** | **21.60** | **12.70** |
| LightGBM (Tuned) | 0.9389 | 22.26 | 13.09 |
| CatBoost (Tuned) | 0.9372 | 22.57 | 13.20 |
| XGBoost (Tuned) | 0.9367 | 22.66 | 13.69 |
| Lasso (α=1.0) | 0.8788 | 31.36 | 19.50 |
| Ridge (α=10.0) | 0.8514 | 34.72 | 18.44 |
| Linear Regression | 0.8451 | 35.44 | 18.49 |
| **Forecast Model (XGBoost)** | **0.9407** | **21.93** | **12.58** |

**City-specific vs Global:** Global model better for **22/26** cities.

## Quick Start

```bash
pip install -r requirements.txt

# Option 1: Run the full pipeline
jupyter notebook notebooks/AQI_full_pipeline.ipynb

# Option 2: Launch dashboard
streamlit run dashboard/app.py

# Option 3: Retrain from scratch
python scripts/retrain.py
```

## Repository Structure

```
AQI_ML_PROJECT/
├── data/
│   └── raw/
│       └── city_day.csv           # Kaggle dataset (29,531 rows, 26 cities)
├── notebooks/
│   └── AQI_full_pipeline.ipynb    # Master notebook — 7 phases
├── dashboard/
│   └── app.py                     # Streamlit dashboard (3 tabs)
├── scripts/
│   └── retrain.py                 # End-to-end retraining pipeline
├── models/                        # Trained artifacts
├── docs/
│   ├── index.html                 # GitHub Pages → redirects to dashboard
│   └── figures/                   # EDA + SHAP visualizations
└── requirements.txt
```

## Saved Artifacts

| File | Description |
|---|---|
| `models/best_model.pkl` | Champion: Random Forest (16 MB, compressed) |
| `models/forecast_model.pkl` | Forecast: XGBoost (lag-only) |
| `models/scaler.pkl` | StandardScaler for features |
| `models/city_encoder.pkl` | City → mean AQI map |
| `models/feature_columns.pkl` | 46 feature column names |
| `models/city_shap_analysis.pkl` | Per-city top pollutants |
| `data/processed/train.csv` | Training set (chronological 80%) |
| `data/processed/test.csv` | Test set (chronological 20%) |

## Live Deployment

| Service | URL |
|---|---|
| **Dashboard** | [https://aqi-prediction.onrender.com](https://aqi-prediction.onrender.com) |
| **GitHub Repo** | [https://github.com/Sahityabiswas/ML_BASED_AQI_PREDICTION](https://github.com/Sahityabiswas/ML_BASED_AQI_PREDICTION) |

## Dataset

Kaggle: [Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
