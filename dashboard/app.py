import numpy as np
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(parent, 'models')

app = Flask(__name__)

def load_artifacts():
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        features = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
        city_enc = joblib.load(os.path.join(MODELS_DIR, 'city_encoder.pkl'))
        return model, scaler, features, city_enc
    except FileNotFoundError:
        return None, None, None, None

def load_forecast_artifacts():
    try:
        fc_model = joblib.load(os.path.join(MODELS_DIR, 'forecast_model.pkl'))
        fc_scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler_forecast.pkl'))
        fc_features = joblib.load(os.path.join(MODELS_DIR, 'forecast_features.pkl'))
        return fc_model, fc_scaler, fc_features
    except FileNotFoundError:
        return None, None, None

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
    ('Good', 0, 50), ('Satisfactory', 51, 100), ('Moderate', 101, 200),
    ('Poor', 201, 300), ('Very Poor', 301, 400), ('Severe', 401, 999)
]

def get_bucket(val):
    for name, lo, hi in AQI_BUCKETS:
        if lo <= val <= hi:
            return name
    return 'Severe'

COLOR_MAP = {'Good': '#28a745', 'Satisfactory': '#82c91e', 'Moderate': '#fd7e14',
             'Poor': '#dc3545', 'Very Poor': '#6f42c1', 'Severe': '#800000'}

POLLUTANT_META = {
    'PM2.5': {'display': 'PM2.5 (µg/m³)', 'default': 80.0, 'min': 0, 'max': 1000, 'step': 1},
    'PM10':  {'display': 'PM10 (µg/m³)',  'default': 150.0, 'min': 0, 'max': 1000, 'step': 1},
    'NO':    {'display': 'NO (ppb)',       'default': 30.0,  'min': 0, 'max': 500,  'step': 1},
    'NO2':   {'display': 'NO2 (ppb)',      'default': 50.0,  'min': 0, 'max': 500,  'step': 1},
    'NOx':   {'display': 'NOx (ppb)',      'default': 80.0,  'min': 0, 'max': 500,  'step': 1},
    'NH3':   {'display': 'NH3 (ppm)',      'default': 10.0,  'min': 0, 'max': 100,  'step': 0.1},
    'CO':    {'display': 'CO (mg/m³)',     'default': 2.0,   'min': 0, 'max': 50,   'step': 0.1},
    'SO2':   {'display': 'SO2 (ppb)',      'default': 15.0,  'min': 0, 'max': 200,  'step': 1},
    'O3':    {'display': 'O3 (ppb)',       'default': 60.0,  'min': 0, 'max': 500,  'step': 1},
    'Benzene': {'display': 'Benzene (µg/m³)', 'default': 5.0, 'min': 0, 'max': 100,  'step': 0.1},
    'Toluene': {'display': 'Toluene (µg/m³)', 'default': 15.0, 'min': 0, 'max': 200, 'step': 0.1},
    'Xylene':  {'display': 'Xylene (µg/m³)',  'default': 10.0, 'min': 0, 'max': 200, 'step': 0.1},
}

COL_LAYOUT = [
    ['PM2.5', 'NO2', 'CO', 'Benzene'],
    ['PM10', 'NOx', 'SO2', 'Toluene'],
    ['NO', 'NH3', 'O3', 'Xylene'],
]

season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}

@app.route('/')
def index():
    cities = list(city_enc['city_mean_aqi'].index) if city_enc is not None else ['Delhi']
    shap_html = city_shap_df.to_html(classes='table table-striped table-hover', index=False) if city_shap_df is not None and not city_shap_df.empty else None
    poll_meta_list = [(k, v) for k, v in POLLUTANT_META.items()]
    return render_template('index.html',
        cities=cities, poll_meta=POLLUTANT_META, col_layout=COL_LAYOUT,
        model_loaded=model_loaded, shap_html=shap_html)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not found. Train models first.'}), 400
    data = request.get_json()
    p = data.get('pollutants', {})
    city = data.get('city', 'Delhi')
    city_mean = city_enc['city_mean_aqi'].get(city, 200)
    city_freq = city_enc['city_freq'].get(city, 1000)

    def f(k):
        return float(p.get(k, POLLUTANT_META[k]['default']))

    feat_dict = {
        'Year': 2020, 'Month': 1, 'Day': 15, 'DayOfWeek': 3, 'IsWeekend': 0, 'Season': 1,
        'PM25_x_PM10': f('PM2.5') * f('PM10'),
        'CO_x_NO2': f('CO') * f('NO2'),
        'SO2_x_NO2': f('SO2') * f('NO2'),
        'PM25_x_CO': f('PM2.5') * f('CO'),
        'O3_x_NO2': f('O3') * f('NO2'),
        'PM25_div_PM10': f('PM2.5') / (f('PM10') + 1e-6),
        'NO2_div_NOx': f('NO2') / (f('NOx') + 1e-6),
        'CO_div_SO2': f('CO') / (f('SO2') + 1e-6),
        'City_TargetEncoded': city_mean,
        'City_Frequency': city_freq,
    }
    for lag in [1, 2, 3, 7]:
        for col in ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']:
            feat_dict[f'{col}_lag{lag}'] = f(col) if col in p else 0
    for col in ['PM2.5', 'PM10', 'CO', 'AQI']:
        val = f(col) if col in p else 0
        feat_dict[f'{col}_roll3'] = val
        feat_dict[f'{col}_roll7'] = val

    input_df = pd.DataFrame([feat_dict])
    for c in features:
        if c not in input_df.columns:
            input_df[c] = 0
    input_df = input_df[features]
    input_scaled = scaler.transform(input_df)
    pred = float(model.predict(input_scaled)[0])
    bucket = get_bucket(pred)
    result = {'aqi': round(pred, 1), 'category': bucket, 'color': COLOR_MAP[bucket]}

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_scaled)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(values=shap_vals[0], base_values=explainer.expected_value,
                             data=input_scaled[0], feature_names=features),
            max_display=15, show=False)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result['shap_plot'] = base64.b64encode(buf.read()).decode()
        plt.close()
    except Exception:
        result['shap_plot'] = None

    return jsonify(result)

@app.route('/forecast', methods=['POST'])
def forecast():
    if not model_loaded or fc_model is None:
        return jsonify({'error': 'Forecast model not found. Train models first.'}), 400
    data = request.get_json()
    fc_city = data.get('city', 'Delhi')
    p = data.get('pollutants', {})

    def f(k):
        return float(p.get(k, POLLUTANT_META[k]['default']))

    seed_poll = {k: f(k) for k in POLLUTANT_META}
    seed_poll['AQI'] = 150
    dummy_date = pd.Timestamp.today()
    city_mean = city_enc['city_mean_aqi'].get(fc_city, 200)
    city_freq = city_enc['city_freq'].get(fc_city, 1000)
    fc_rows = []

    for _ in range(7):
        row = {
            'Month': dummy_date.month,
            'Season': season_map.get(dummy_date.month, 0),
            'IsWeekend': 1 if dummy_date.dayofweek >= 5 else 0,
            'City_TargetEncoded': city_mean,
            'City_Frequency': city_freq,
        }
        for col in ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']:
            for lag in [1, 2, 3, 7]:
                val = fc_rows[-lag].get(col, seed_poll.get(col, 150)) if len(fc_rows) >= lag else seed_poll.get(col, 150)
                row[f'{col}_lag{lag}'] = val
        for col in ['PM2.5', 'PM10', 'CO']:
            vals = [r.get(col, seed_poll.get(col, 150)) for r in fc_rows[-3:]] if fc_rows else [seed_poll.get(col, 150)]
            row[f'{col}_roll3'] = float(np.mean(vals))
            vals7 = [r.get(col, seed_poll.get(col, 150)) for r in fc_rows[-7:]] if fc_rows else [seed_poll.get(col, 150)]
            row[f'{col}_roll7'] = float(np.mean(vals7))
        fc_rows.append(row)
        dummy_date += pd.Timedelta(days=1)

    fc_df = pd.DataFrame(fc_rows)
    for c in fc_features:
        if c not in fc_df.columns:
            fc_df[c] = 0
    fc_df = fc_df[fc_features]
    fc_scaled = fc_scaler.transform(fc_df)
    forecast_values = fc_model.predict(fc_scaled)
    for i, val in enumerate(forecast_values):
        fc_rows[i]['AQI'] = float(val)

    forecast_dates = pd.date_range(pd.Timestamp.today(), periods=7)
    results = []
    for i in range(7):
        bucket = get_bucket(float(forecast_values[i]))
        results.append({'date': forecast_dates[i].strftime('%Y-%m-%d'),
                        'aqi': round(float(forecast_values[i]), 1),
                        'category': bucket, 'color': COLOR_MAP[bucket]})

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['green' if v <= 50 else 'lime' if v <= 100 else 'orange' if v <= 200 else 'red' if v <= 300 else 'purple' if v <= 400 else 'maroon' for v in forecast_values]
    ax.bar(forecast_dates, forecast_values, color=colors, width=0.6)
    ax.axhline(y=50, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=100, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=200, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('AQI')
    ax.set_title('7-Day AQI Forecast')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return jsonify({'results': results, 'chart': chart_b64})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
