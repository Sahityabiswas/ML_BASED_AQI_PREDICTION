import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
import json
import urllib.request
from flask import Flask, render_template, request, jsonify

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(parent, 'models')
app = Flask(__name__)

ARTIFACTS = [
    ('model', 'best_model.pkl'), ('scaler', 'scaler.pkl'),
    ('features', 'feature_columns.pkl'), ('city_enc', 'city_encoder.pkl'),
    ('fc_model', 'forecast_model.pkl'), ('fc_scaler', 'scaler_forecast.pkl'),
    ('fc_features', 'forecast_features.pkl'), ('city_shap_df', 'city_shap_analysis.pkl'),
]
loaded = {}
for name, fname in ARTIFACTS:
    try:
        loaded[name] = joblib.load(os.path.join(MODELS_DIR, fname))
    except FileNotFoundError:
        loaded[name] = None

model = loaded['model']; scaler = loaded['scaler']
features = loaded['features']; city_enc = loaded['city_enc']
fc_model = loaded['fc_model']; fc_scaler = loaded['fc_scaler']
fc_features = loaded['fc_features']; city_shap_df = loaded['city_shap_df']
model_loaded = model is not None

AQI_BUCKETS = [
    ('Good', 0, 50), ('Satisfactory', 51, 100), ('Moderate', 101, 200),
    ('Poor', 201, 300), ('Very Poor', 301, 400), ('Severe', 401, 999)
]
COLOR_MAP = {b[0]: c for b, c in zip(AQI_BUCKETS, ['#28a745','#82c91e','#fd7e14','#dc3545','#6f42c1','#800000'])}
GLOW_MAP = {k: v.replace(')', ',0.3)').replace('#28a745','rgba(40,167,69').replace('#82c91e','rgba(130,201,30').replace('#fd7e14','rgba(253,126,20').replace('#dc3545','rgba(220,53,69').replace('#6f42c1','rgba(111,66,193').replace('#800000','rgba(128,0,0') for k, v in COLOR_MAP.items()}

def get_bucket(val):
    for name, lo, hi in AQI_BUCKETS:
        if lo <= val <= hi:
            return name
    return 'Severe'

POLLUTANT_META = {
    'PM2.5': {'display': 'PM2.5 (\u00b5g/m\u00b3)', 'default': 80.0, 'min': 0, 'max': 1000, 'step': 1, 'icon': 'fa-wind'},
    'PM10':  {'display': 'PM10 (\u00b5g/m\u00b3)',  'default': 150.0, 'min': 0, 'max': 1000, 'step': 1, 'icon': 'fa-smog'},
    'NO':    {'display': 'NO (ppb)',       'default': 30.0,  'min': 0, 'max': 500,  'step': 1, 'icon': 'fa-flask'},
    'NO2':   {'display': 'NO2 (ppb)',      'default': 50.0,  'min': 0, 'max': 500,  'step': 1, 'icon': 'fa-flask'},
    'NOx':   {'display': 'NOx (ppb)',      'default': 80.0,  'min': 0, 'max': 500,  'step': 1, 'icon': 'fa-flask'},
    'NH3':   {'display': 'NH3 (ppm)',      'default': 10.0,  'min': 0, 'max': 100,  'step': 0.1, 'icon': 'fa-vial'},
    'CO':    {'display': 'CO (mg/m\u00b3)',     'default': 2.0,   'min': 0, 'max': 50,   'step': 0.1, 'icon': 'fa-industry'},
    'SO2':   {'display': 'SO2 (ppb)',      'default': 15.0,  'min': 0, 'max': 200,  'step': 1, 'icon': 'fa-smoke'},
    'O3':    {'display': 'O3 (ppb)',       'default': 60.0,  'min': 0, 'max': 500,  'step': 1, 'icon': 'fa-sun'},
    'Benzene': {'display': 'Benzene (\u00b5g/m\u00b3)', 'default': 5.0, 'min': 0, 'max': 100,  'step': 0.1, 'icon': 'fa-skull'},
    'Toluene': {'display': 'Toluene (\u00b5g/m\u00b3)', 'default': 15.0, 'min': 0, 'max': 200, 'step': 0.1, 'icon': 'fa-skull'},
    'Xylene':  {'display': 'Xylene (\u00b5g/m\u00b3)',  'default': 10.0, 'min': 0, 'max': 200, 'step': 0.1, 'icon': 'fa-skull'},
}
COL_LAYOUT = [['PM2.5', 'NO2', 'CO', 'Benzene'], ['PM10', 'NOx', 'SO2', 'Toluene'], ['NO', 'NH3', 'O3', 'Xylene']]
season_map = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}

LAG_COLS = ['AQI', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
ROLL_COLS = ['PM2.5', 'PM10', 'CO', 'AQI']
CITY_COORDS = {
    'Ahmedabad':(23.0225,72.5714),'Aizawl':(23.7271,92.7176),'Amaravati':(16.5417,80.5150),
    'Amritsar':(31.6340,74.8723),'Bengaluru':(12.9716,77.5946),'Bhopal':(23.2599,77.4126),
    'Brajrajnagar':(21.8333,83.9167),'Chandigarh':(30.7333,76.7794),'Chennai':(13.0827,80.2707),
    'Coimbatore':(11.0168,76.9558),'Delhi':(28.7041,77.1025),'Ernakulam':(9.9816,76.2995),
    'Gurugram':(28.4595,77.0266),'Guwahati':(26.1445,91.7362),'Hyderabad':(17.3850,78.4867),
    'Jaipur':(26.9124,75.7873),'Jorapokhar':(23.6833,86.4333),'Kochi':(9.9312,76.2673),
    'Kolkata':(22.5726,88.3639),'Lucknow':(26.8467,80.9462),'Mumbai':(19.0760,72.8777),
    'Patna':(25.5941,85.1376),'Shillong':(25.5788,91.8933),'Talcher':(20.9500,85.2333),
    'Thiruvananthapuram':(8.5241,76.9366),'Visakhapatnam':(17.6868,83.2185),
}
_UM3_TO_PPB = {'nitrogen_dioxide':24.465/46.0055,'sulphur_dioxide':24.465/64.066,'ozone':24.465/48.0,'nitrogen_monoxide':24.465/30.006}
_OM_VARS = 'pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,nitrogen_monoxide,sulphur_dioxide,ozone,ammonia'

def fetch_openmeteo(city):
    coord = CITY_COORDS.get(city)
    if not coord: return None
    try:
        with urllib.request.urlopen(
            f'https://air-quality-api.open-meteo.com/v1/air-quality?latitude={coord[0]}&longitude={coord[1]}&current={_OM_VARS}',
            timeout=10) as r:
            cur = json.loads(r.read().decode()).get('current', {})
            if not cur: return None
            v = lambda k: cur.get(k)
            rv = {}
            if v('pm2_5') is not None: rv['PM2.5'] = round(float(v('pm2_5')), 1)
            if v('pm10') is not None: rv['PM10'] = round(float(v('pm10')), 1)
            if v('carbon_monoxide') is not None: rv['CO'] = round(float(v('carbon_monoxide'))/1000, 2)
            if v('nitrogen_dioxide') is not None: rv['NO2'] = round(float(v('nitrogen_dioxide'))*_UM3_TO_PPB['nitrogen_dioxide'], 2)
            if v('nitrogen_monoxide') is not None: rv['NO'] = round(float(v('nitrogen_monoxide'))*_UM3_TO_PPB['nitrogen_monoxide'], 2)
            if v('sulphur_dioxide') is not None: rv['SO2'] = round(float(v('sulphur_dioxide'))*_UM3_TO_PPB['sulphur_dioxide'], 2)
            if v('ozone') is not None: rv['O3'] = round(float(v('ozone'))*_UM3_TO_PPB['ozone'], 2)
            if v('ammonia') is not None: rv['NH3'] = round(float(v('ammonia')), 1)
            return rv if rv else None
    except Exception:
        return None

def fill_missing(df, template_cols):
    for c in template_cols:
        if c not in df.columns:
            df[c] = 0
    return df[template_cols]

def f_val(p, k):
    return float(p.get(k, POLLUTANT_META[k]['default']))

@app.route('/')
def index():
    cities = list(city_enc['city_mean_aqi'].index) if city_enc else ['Delhi']
    shap_html = city_shap_df.to_html(classes='table table-striped table-hover', index=False) if city_shap_df is not None and not city_shap_df.empty else None
    return render_template('index.html', cities=cities, poll_meta=POLLUTANT_META,
        col_layout=COL_LAYOUT, model_loaded=model_loaded, shap_html=shap_html)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not found. Train models first.'}), 400
    data = request.get_json(); p = data.get('pollutants', {})
    city = data.get('city', 'Delhi'); cm = city_enc['city_mean_aqi'].get(city, 200); cf = city_enc['city_freq'].get(city, 1000)
    f = lambda k: f_val(p, k)
    fd = {
        'Year':2020,'Month':1,'Day':15,'DayOfWeek':3,'IsWeekend':0,'Season':1,
        'PM25_x_PM10': f('PM2.5')*f('PM10'), 'CO_x_NO2': f('CO')*f('NO2'),
        'SO2_x_NO2': f('SO2')*f('NO2'), 'PM25_x_CO': f('PM2.5')*f('CO'),
        'O3_x_NO2': f('O3')*f('NO2'), 'PM25_div_PM10': f('PM2.5')/(f('PM10')+1e-6),
        'NO2_div_NOx': f('NO2')/(f('NOx')+1e-6), 'CO_div_SO2': f('CO')/(f('SO2')+1e-6),
        'City_TargetEncoded': cm, 'City_Frequency': cf,
    }
    for lag in [1,2,3,7]:
        for col in LAG_COLS:
            fd[f'{col}_lag{lag}'] = f(col) if col in p else 0
    for col in ROLL_COLS:
        v = f(col) if col in p else 0; fd[f'{col}_roll3'] = v; fd[f'{col}_roll7'] = v
    inp = fill_missing(pd.DataFrame([fd]), features)
    pred = float(model.predict(scaler.transform(inp))[0])
    bucket = get_bucket(pred)
    result = {'aqi': round(pred,1), 'category': bucket, 'color': COLOR_MAP[bucket], 'glow': GLOW_MAP[bucket]}
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(scaler.transform(inp))[0]
        combined = sorted(zip(features, sv), key=lambda x: abs(x[1]), reverse=True)[:15]
        result['shap_values'] = [{'feature': cf[0], 'value': round(float(cf[1]),4)} for cf in combined]
    except Exception:
        result['shap_values'] = None
    return jsonify(result)

@app.route('/api/openmeteo')
def api_openmeteo():
    city = request.args.get('city', 'Delhi')
    vals = fetch_openmeteo(city)
    return jsonify({'values': vals} if vals else {'values': None})

@app.route('/forecast', methods=['POST'])
def forecast():
    if not model_loaded or fc_model is None:
        return jsonify({'error': 'Forecast model not found. Train models first.'}), 400
    data = request.get_json(); fc_city = data.get('city', 'Delhi')
    p = data.get('pollutants', {}); f = lambda k: f_val(p, k)
    seed = {k: f(k) for k in POLLUTANT_META}; seed['AQI'] = 150
    d = pd.Timestamp.today(); cm = city_enc['city_mean_aqi'].get(fc_city, 200); cf = city_enc['city_freq'].get(fc_city, 1000)
    rows = []
    for _ in range(7):
        row = {'Month': d.month, 'Season': season_map.get(d.month,0), 'IsWeekend': 1 if d.dayofweek>=5 else 0,
               'City_TargetEncoded': cm, 'City_Frequency': cf}
        for col in LAG_COLS:
            for lag in [1,2,3,7]:
                row[f'{col}_lag{lag}'] = rows[-lag].get(col, seed.get(col,150)) if len(rows)>=lag else seed.get(col,150)
        for col in ['PM2.5','PM10','CO']:
            v = [r.get(col,seed.get(col,150)) for r in rows[-3:]] if rows else [seed.get(col,150)]
            row[f'{col}_roll3'] = float(np.mean(v))
            v7 = [r.get(col,seed.get(col,150)) for r in rows[-7:]] if rows else [seed.get(col,150)]
            row[f'{col}_roll7'] = float(np.mean(v7))
        rows.append(row); d += pd.Timedelta(days=1)
    fc_df = fill_missing(pd.DataFrame(rows), fc_features)
    vals = fc_model.predict(fc_scaler.transform(fc_df))
    for i, v in enumerate(vals): rows[i]['AQI'] = float(v)
    dates = pd.date_range(pd.Timestamp.today(), periods=7)
    results = []
    for i in range(7):
        b = get_bucket(float(vals[i]))
        results.append({'date': dates[i].strftime('%Y-%m-%d'), 'aqi': round(float(vals[i]),1),
                        'category': b, 'color': COLOR_MAP[b], 'glow': GLOW_MAP[b]})
    return jsonify({'results': results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
