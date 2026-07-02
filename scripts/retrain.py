"""
retrain.py — End-to-end pipeline: load raw data, preprocess, train models, save artifacts.
Usage: python scripts/retrain.py
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import os
import sys
from datetime import datetime

from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'city_day.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'docs', 'figures')

for d in [PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

print('=' * 60)
print('AQI PIPELINE: RETRAINING')
print('=' * 60)

df = pd.read_csv(RAW_PATH)
print(f'Raw data: {df.shape}')

poll_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
available_poll = [c for c in poll_cols if c in df.columns]

df_work = df.copy()
df_knn = df_work.copy()
city_groups = df_knn.groupby('City')
for city, group in city_groups:
    if len(group) > 1:
        idx = group.index
        subset = group[available_poll].copy()
        # Drop columns that are entirely NaN (KNNImputer can't handle them)
        valid_cols = [c for c in subset.columns if subset[c].notna().any()]
        if subset.isnull().any().any() and len(valid_cols) > 0:
            imputer = KNNImputer(n_neighbors=min(5, len(subset) - 1))
            imputed = pd.DataFrame(
                imputer.fit_transform(subset[valid_cols]),
                columns=valid_cols,
                index=idx
            )
            df_knn.loc[idx, valid_cols] = imputed
for col in available_poll:
    df_knn[col] = df_knn[col].fillna(df_knn[col].mean())

df_feat = df_knn.copy()

def get_aqi_bucket(val):
    if pd.isna(val):
        return None
    if val <= 50:
        return 'Good'
    elif val <= 100:
        return 'Satisfactory'
    elif val <= 200:
        return 'Moderate'
    elif val <= 300:
        return 'Poor'
    elif val <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

df_feat['AQI_Bucket'] = df_feat['AQI'].apply(get_aqi_bucket)
df_feat['Date'] = pd.to_datetime(df_feat['Date'], errors='coerce')
df_feat['Year'] = df_feat['Date'].dt.year
df_feat['Month'] = df_feat['Date'].dt.month
df_feat['Day'] = df_feat['Date'].dt.day
df_feat['DayOfWeek'] = df_feat['Date'].dt.dayofweek
df_feat['IsWeekend'] = df_feat['DayOfWeek'].isin([5, 6]).astype(int)
df_feat['Season'] = df_feat['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
}).astype('category').cat.codes
df_feat['Season'] = df_feat['Season'].astype('category').cat.codes

df_feat['PM25_x_PM10'] = df_feat['PM2.5'] * df_feat['PM10']
df_feat['CO_x_NO2'] = df_feat['CO'] * df_feat['NO2']
df_feat['SO2_x_NO2'] = df_feat['SO2'] * df_feat['NO2']
df_feat['PM25_x_CO'] = df_feat['PM2.5'] * df_feat['CO']
df_feat['O3_x_NO2'] = df_feat['O3'] * df_feat['NO2']
df_feat['PM25_div_PM10'] = df_feat['PM2.5'] / (df_feat['PM10'] + 1e-6)
df_feat['NO2_div_NOx'] = df_feat['NO2'] / (df_feat['NOx'] + 1e-6)
df_feat['CO_div_SO2'] = df_feat['CO'] / (df_feat['SO2'] + 1e-6)

df_feat = df_feat.sort_values(['City', 'Date']).reset_index(drop=True)
df_feat = df_feat.dropna(subset=['AQI'])

city_mean_aqi = df_feat.groupby('City')['AQI'].mean()
df_feat['City_TargetEncoded'] = df_feat['City'].map(city_mean_aqi)
city_freq = df_feat['City'].value_counts()
df_feat['City_Frequency'] = df_feat['City'].map(city_freq)
joblib.dump({'city_mean_aqi': city_mean_aqi, 'city_freq': city_freq},
            os.path.join(MODELS_DIR, 'city_encoder.pkl'))

df_feat = df_feat.sort_values(['City', 'Date'])
lag_cols = ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
for col in lag_cols:
    for lag in [1, 2, 3, 7]:
        df_feat[f'{col}_lag{lag}'] = df_feat.groupby('City')[col].shift(lag)
for col in ['PM2.5', 'PM10', 'CO', 'AQI']:
    df_feat[f'{col}_roll3'] = df_feat.groupby('City')[col].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean())
    df_feat[f'{col}_roll7'] = df_feat.groupby('City')[col].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean())
df_feat = df_feat.dropna()

raw_poll_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
target_lag_cols = [c for c in df_feat.columns if c.startswith('AQI_')]
exclude_cols = ['City', 'Date', 'AQI_Bucket', 'AQI'] + raw_poll_cols + target_lag_cols
feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
X = df_feat[feature_cols].copy()
y = df_feat['AQI'].copy()

df_feat_sorted = df_feat.sort_values('Date')
split_idx = int(len(df_feat_sorted) * 0.8)
train_idx = df_feat_sorted.index[:split_idx]
test_idx = df_feat_sorted.index[split_idx:]
X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_columns.pkl'))
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

def evaluate(name, y_true, y_pred):
    return {'Model': name, 'R²': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100}

results = []
for name, model in [
    ('Linear Regression', LinearRegression()),
    ('Ridge', Ridge(alpha=1.0, random_state=42)),
    ('Lasso', Lasso(alpha=0.1, random_state=42)),
]:
    model.fit(X_train_scaled, y_train)
    res = evaluate(name, y_test, model.predict(X_test_scaled))
    results.append(res)
    print(f'{name:25s} | R²={res["R²"]:.4f}')

rf = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                         {'n_estimators': [100, 200], 'max_depth': [10, 15, None],
                          'min_samples_split': [2, 5]},
                         n_iter=10, cv=3, scoring='r2', random_state=42, verbose=0)
rf.fit(X_train_scaled, y_train)
res_rf = evaluate('Random Forest', y_test, rf.predict(X_test_scaled))
results.append(res_rf)
print(f'Random Forest (Tuned)     | R²={res_rf["R²"]:.4f}')

xgb = RandomizedSearchCV(XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
                          {'n_estimators': [100, 200], 'max_depth': [5, 7],
                           'learning_rate': [0.05, 0.1]},
                          n_iter=6, cv=3, scoring='r2', random_state=42, verbose=0)
xgb.fit(X_train_scaled, y_train)
res_xgb = evaluate('XGBoost', y_test, xgb.predict(X_test_scaled))
results.append(res_xgb)
print(f'XGBoost (Tuned)          | R²={res_xgb["R²"]:.4f}')

lgb = RandomizedSearchCV(LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                          {'n_estimators': [100, 200], 'max_depth': [5, 10],
                           'learning_rate': [0.05, 0.1]},
                          n_iter=6, cv=3, scoring='r2', random_state=42, verbose=0)
lgb.fit(X_train_scaled, y_train)
res_lgb = evaluate('LightGBM', y_test, lgb.predict(X_test_scaled))
results.append(res_lgb)
print(f'LightGBM (Tuned)         | R²={res_lgb["R²"]:.4f}')

cb = RandomizedSearchCV(CatBoostRegressor(random_state=42, verbose=0),
                         {'iterations': [100, 200], 'depth': [6, 8],
                          'learning_rate': [0.05, 0.1]},
                         n_iter=6, cv=3, scoring='r2', random_state=42, verbose=0)
cb.fit(X_train_scaled, y_train)
res_cb = evaluate('CatBoost', y_test, cb.predict(X_test_scaled))
results.append(res_cb)
print(f'CatBoost (Tuned)         | R²={res_cb["R²"]:.4f}')

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
print(f'\nChampion: {results_df.iloc[0]["Model"]} (R²={results_df.iloc[0]["R²"]:.4f})')

best_name = results_df.iloc[0]['Model']
if 'XGBoost' in best_name:
    champion = xgb.best_estimator_
elif 'Random Forest' in best_name:
    champion = rf.best_estimator_
elif 'LightGBM' in best_name:
    champion = lgb.best_estimator_
elif 'CatBoost' in best_name:
    champion = cb.best_estimator_
else:
    champion = LinearRegression()
joblib.dump(champion, os.path.join(MODELS_DIR, 'best_model.pkl'))
print(f'Model saved: best_model.pkl')

forecast_features = [c for c in feature_cols if 'lag' in c or 'roll' in c or c in ['Month', 'Season', 'IsWeekend']]
current_poll = [c for c in available_poll if c in df_feat.columns]
X_forecast = pd.concat([df_feat[forecast_features], df_feat[current_poll]], axis=1)
y_forecast = df_feat['AQI']
fc_split = int(len(df_feat_sorted) * 0.8)
fc_train_idx = df_feat_sorted.index[:fc_split]
fc_test_idx = df_feat_sorted.index[fc_split:]
Xfc_train, Xfc_test = X_forecast.loc[fc_train_idx], X_forecast.loc[fc_test_idx]
yfc_train, yfc_test = y_forecast.loc[fc_train_idx], y_forecast.loc[fc_test_idx]
scaler_fc = StandardScaler()
Xfc_train_scaled = scaler_fc.fit_transform(Xfc_train)
Xfc_test_scaled = scaler_fc.transform(Xfc_test)
fc_model = XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
fc_model.fit(Xfc_train_scaled, yfc_train)
fc_r2 = r2_score(yfc_test, fc_model.predict(Xfc_test_scaled))
joblib.dump(fc_model, os.path.join(MODELS_DIR, 'forecast_model.pkl'))
joblib.dump(scaler_fc, os.path.join(MODELS_DIR, 'scaler_forecast.pkl'))
joblib.dump(list(Xfc_train.columns), os.path.join(MODELS_DIR, 'forecast_features.pkl'))
print(f'Forecast model saved (R²={fc_r2:.4f})')

df_train = df_feat.loc[train_idx]
df_test = df_feat.loc[test_idx]
df_train.to_csv(os.path.join(PROCESSED_DIR, 'train.csv'), index=False)
df_test.to_csv(os.path.join(PROCESSED_DIR, 'test.csv'), index=False)
print(f'Processed data saved: {len(df_train)} train / {len(df_test)} test rows')

print('\n' + '=' * 60)
print('RETRAINING COMPLETE')
print('=' * 60)
