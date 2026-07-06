import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(parent, 'models')

st.set_page_config(page_title='AQI Prediction Dashboard', layout='wide')
st.title('Air Quality Index (AQI) Prediction System')

@st.cache_resource
def load_artifacts():
    base = MODELS_DIR
    try:
        model = joblib.load(os.path.join(base, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(base, 'scaler.pkl'))
        features = joblib.load(os.path.join(base, 'feature_columns.pkl'))
        city_enc = joblib.load(os.path.join(base, 'city_encoder.pkl'))
        return model, scaler, features, city_enc
    except FileNotFoundError:
        return None, None, None, None

@st.cache_resource
def load_forecast_artifacts():
    base = MODELS_DIR
    try:
        fc_model = joblib.load(os.path.join(base, 'forecast_model.pkl'))
        fc_scaler = joblib.load(os.path.join(base, 'scaler_forecast.pkl'))
        fc_features = joblib.load(os.path.join(base, 'forecast_features.pkl'))
        return fc_model, fc_scaler, fc_features
    except FileNotFoundError:
        return None, None, None

@st.cache_resource
def load_city_shap():
    try:
        return joblib.load(os.path.join(MODELS_DIR, 'city_shap_analysis.pkl'))
    except FileNotFoundError:
        return None

model, scaler, features, city_enc = load_artifacts()
fc_model, fc_scaler, fc_features = load_forecast_artifacts()
city_shap_df = load_city_shap()

model_loaded = model is not None

AQI_BUCKETS = [
    ('Good', 0, 50),
    ('Satisfactory', 51, 100),
    ('Moderate', 101, 200),
    ('Poor', 201, 300),
    ('Very Poor', 301, 400),
    ('Severe', 401, 999)
]

def get_bucket(val):
    for name, lo, hi in AQI_BUCKETS:
        if lo <= val <= hi:
            return name
    return 'Severe'

p1, p2, p3 = st.tabs(['AQI Prediction + SHAP', 'City Insights', '7-Day Forecast'])

with p1:
    st.header('Predict AQI from Pollutant Levels')
    if not model_loaded:
        st.warning('Model not found. Run the notebook (AQI_full_pipeline.ipynb) first to train models.')
    else:
        col1, col2, col3 = st.columns(3)
        poll_inputs = {
            'PM2.5': col1.number_input('PM2.5 (ug/m³)', 0.0, 1000.0, 80.0, 1.0),
            'PM10': col2.number_input('PM10 (ug/m³)', 0.0, 1000.0, 150.0, 1.0),
            'NO': col3.number_input('NO (ppb)', 0.0, 500.0, 30.0, 1.0),
            'NO2': col1.number_input('NO2 (ppb)', 0.0, 500.0, 50.0, 1.0),
            'NOx': col2.number_input('NOx (ppb)', 0.0, 500.0, 80.0, 1.0),
            'NH3': col3.number_input('NH3 (ppm)', 0.0, 100.0, 10.0, 0.1),
            'CO': col1.number_input('CO (mg/m³)', 0.0, 50.0, 2.0, 0.1),
            'SO2': col2.number_input('SO2 (ppb)', 0.0, 200.0, 15.0, 1.0),
            'O3': col3.number_input('O3 (ppb)', 0.0, 500.0, 60.0, 1.0),
            'Benzene': col1.number_input('Benzene (ug/m³)', 0.0, 100.0, 5.0, 0.1),
            'Toluene': col2.number_input('Toluene (ug/m³)', 0.0, 200.0, 15.0, 0.1),
            'Xylene': col3.number_input('Xylene (ug/m³)', 0.0, 200.0, 10.0, 0.1),
        }
        city = col1.selectbox('City', list(city_enc['city_mean_aqi'].index) if city_enc else ['Delhi'])

        if st.button('Predict AQI', type='primary'):
            city_mean = city_enc['city_mean_aqi'].get(city, 200)
            city_freq = city_enc['city_freq'].get(city, 1000)

            feat_dict = {
                'Year': 2020, 'Month': 1, 'Day': 15, 'DayOfWeek': 3, 'IsWeekend': 0,
                'Season': 1,
                'PM25_x_PM10': poll_inputs['PM2.5'] * poll_inputs['PM10'],
                'CO_x_NO2': poll_inputs['CO'] * poll_inputs['NO2'],
                'SO2_x_NO2': poll_inputs['SO2'] * poll_inputs['NO2'],
                'PM25_x_CO': poll_inputs['PM2.5'] * poll_inputs['CO'],
                'O3_x_NO2': poll_inputs['O3'] * poll_inputs['NO2'],
                'PM25_div_PM10': poll_inputs['PM2.5'] / (poll_inputs['PM10'] + 1e-6),
                'NO2_div_NOx': poll_inputs['NO2'] / (poll_inputs['NOx'] + 1e-6),
                'CO_div_SO2': poll_inputs['CO'] / (poll_inputs['SO2'] + 1e-6),
                'City_TargetEncoded': city_mean,
                'City_Frequency': city_freq,
            }
            for lag in [1, 2, 3, 7]:
                for col in ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']:
                    val = poll_inputs.get(col, 0) if col in poll_inputs else 0
                    feat_dict[f'{col}_lag{lag}'] = val
            for col in ['PM2.5', 'PM10', 'CO', 'AQI']:
                val = poll_inputs.get(col, 0) if col in poll_inputs else 0
                feat_dict[f'{col}_roll3'] = val
                feat_dict[f'{col}_roll7'] = val

            input_df = pd.DataFrame([feat_dict])
            for c in features:
                if c not in input_df.columns:
                    input_df[c] = 0
            input_df = input_df[features]
            input_scaled = scaler.transform(input_df)

            pred = model.predict(input_scaled)[0]
            bucket = get_bucket(pred)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric('Predicted AQI', f'{pred:.1f}')
            col_b.metric('AQI Category', bucket)
            color_map = {'Good': 'green', 'Satisfactory': 'lime', 'Moderate': 'orange',
                         'Poor': 'red', 'Very Poor': 'purple', 'Severe': 'maroon'}
            col_c.markdown(f"<div style='background:{color_map[bucket]};padding:10px;border-radius:5px;text-align:center;color:white;font-weight:bold'> {bucket} </div>", unsafe_allow_html=True)

            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(input_scaled)
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(values=shap_vals[0], base_values=explainer.expected_value,
                                     data=input_scaled[0], feature_names=features),
                    max_display=15, show=False
                )
                st.pyplot(fig)
                plt.close()
            except Exception:
                st.info('SHAP plot not available for this model type.')

with p2:
    st.header('City-Level Insights')
    if city_shap_df is not None and not city_shap_df.empty:
        st.subheader('Top Pollutants by City (SHAP Analysis)')
        st.dataframe(city_shap_df, use_container_width=True)
    else:
        st.warning('City SHAP analysis not found. Run the master notebook first.')

    st.subheader('Feature Reference')
    st.markdown("""
    | Feature | Description |
    |---|---|
    | PM2.5, PM10 | Particulate matter (fine & coarse) |
    | NO, NO2, NOx | Nitrogen oxides |
    | NH3 | Ammonia |
    | CO | Carbon monoxide |
    | SO2 | Sulfur dioxide |
    | O3 | Ozone |
    | Benzene, Toluene, Xylene | Volatile organic compounds |
    | Lag features | Previous day(s) values (1, 2, 3, 7) |
    | Rolling means | 3-day & 7-day moving averages |
    """)

with p3:
    st.header('7-Day AQI Forecast')
    if not model_loaded or fc_model is None:
        st.warning('Forecast model not found. Run the master notebook first.')
    else:
        fc_city = st.selectbox('Select City for Forecast', list(city_enc['city_mean_aqi'].index) if city_enc else ['Delhi'], key='fc_city')

        st.info('Generates 7-day recursive forecast using lag + temporal features.')
        if st.button('Generate 7-Day Forecast', type='primary'):
            dummy_date = pd.Timestamp.today()
            city_mean = city_enc['city_mean_aqi'].get(fc_city, 200)
            city_freq = city_enc['city_freq'].get(fc_city, 1000)
            fc_rows = []
            seed_poll = {p: v for p, v in poll_inputs.items()}
            seed_poll['AQI'] = 150
            season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}
            for day_offset in range(7):
                row = {
                    'Month': dummy_date.month,
                    'Season': season_map.get(dummy_date.month, 0),
                    'IsWeekend': 1 if dummy_date.dayofweek >= 5 else 0,
                    'City_TargetEncoded': city_mean,
                    'City_Frequency': city_freq,
                }
                for col in ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']:
                    for lag in [1, 2, 3, 7]:
                        if len(fc_rows) >= lag:
                            val = fc_rows[-lag].get(col, seed_poll.get(col, 150))
                        else:
                            val = seed_poll.get(col, 150)
                        row[f'{col}_lag{lag}'] = val
                for col in ['PM2.5', 'PM10', 'CO']:
                    vals = [r.get(col, seed_poll.get(col, 150)) for r in fc_rows[-3:]] if fc_rows else [seed_poll.get(col, 150)]
                    row[f'{col}_roll3'] = np.mean(vals)
                    vals7 = [r.get(col, seed_poll.get(col, 150)) for r in fc_rows[-7:]] if fc_rows else [seed_poll.get(col, 150)]
                    row[f'{col}_roll7'] = np.mean(vals7)
                fc_rows.append(row)
                dummy_date += pd.Timedelta(days=1)

            fc_df = pd.DataFrame(fc_rows)
            for c in fc_features:
                if c not in fc_df.columns:
                    fc_df[c] = 0
            fc_df = fc_df[fc_features]
            fc_scaled = fc_scaler.transform(fc_df)
            forecast_values = fc_model.predict(fc_scaled)

            # Update rows with predicted AQI for recursive consistency
            for i, val in enumerate(forecast_values):
                fc_rows[i]['AQI'] = val

            forecast_dates = pd.date_range(pd.Timestamp.today(), periods=7)
            res_df = pd.DataFrame({'Date': forecast_dates.strftime('%Y-%m-%d'), 'Predicted AQI': forecast_values.round(1),
                                    'Category': [get_bucket(v) for v in forecast_values]})
            st.dataframe(res_df, use_container_width=True)

            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['green' if v <= 50 else 'lime' if v <= 100 else 'orange' if v <= 200 else 'red' if v <= 300 else 'purple' if v <= 400 else 'maroon' for v in forecast_values]
                bars = ax.bar(forecast_dates, forecast_values, color=colors, width=0.6)
                ax.axhline(y=50, color='green', linestyle='--', alpha=0.5)
                ax.axhline(y=100, color='orange', linestyle='--', alpha=0.5)
                ax.axhline(y=200, color='red', linestyle='--', alpha=0.5)
                ax.set_ylabel('AQI')
                ax.set_title('7-Day AQI Forecast')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception:
                pass
