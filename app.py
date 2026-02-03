import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import logging
import re
import json
from datetime import date
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. SETUP & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. DEFINE CLASSES (Must match Notebook Exactly) ---
class CarAgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['RegistrationYear', 'RegistrationMonth'])
        X['RegistrationYear'] = pd.to_numeric(X['RegistrationYear'], errors='coerce')
        X['RegistrationMonth'] = pd.to_numeric(X['RegistrationMonth'], errors='coerce')
        X['RegistrationMonth'] = X['RegistrationMonth'].apply(lambda x: 6 if (x < 1 or x > 12) else x)
        X['RegistrationDate'] = pd.to_datetime(
            X['RegistrationYear'].astype(int).astype(str) + '-' + 
            X['RegistrationMonth'].astype(int).astype(str), format='%Y-%m', errors='coerce')
        X['CarAge'] = ((pd.to_datetime('today') - X['RegistrationDate']).dt.days / 365.25).round(2)
        X['CarAge'] = X['CarAge'].fillna(10)
        return X[['CarAge']]

class WeightedEnsemble(BaseEstimator):
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
    def fit(self, X, y):
        for model in self.models: model.fit(X, y)
        return self
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)
        return predictions

# --- 3. LOAD MODEL & DATA ---
try:
    pipeline = joblib.load("joblib/CarPricePipeline.joblib")
    df_clean = pd.read_csv("data/CleanedCarPrices.csv")
    
    unique_brands = sorted(df_clean['Brand'].unique())
    unique_types = sorted(df_clean['VehicleType'].unique())
    unique_fuels = sorted(df_clean['FuelType'].unique())
    
    logger.info("✅ Pipeline and Data loaded successfully.")
except Exception as e:
    logger.error(f"❌ Critical Load Error: {e}")
    pipeline = None
    df_clean = pd.DataFrame()
    unique_brands, unique_types, unique_fuels = [], [], []

def get_exchange_rate():
    try:
        r = requests.get('https://api.exchangerate-api.com/v4/latest/EUR', timeout=2)
        return r.json()['rates']['INR']
    except:
        return 102.88

# --- 4. ROBUST WEB SCRAPER ---
def scrape_listing(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None, "Connection blocked"

        soup = BeautifulSoup(response.content, 'html.parser')
        data = {}
        msg_extra = []
        
        # Helper: Regex for Year (1990-2029)
        year_pattern = r'\b(199\d|20[0-2]\d)\b'
        
        # --- A. JSON-LD (Structured Data) ---
        scripts = soup.find_all('script', type='application/ld+json')
        for script in scripts:
            try:
                js_data = json.loads(script.string)
                if isinstance(js_data, list): js_data = js_data[0]
                
                # Year
                if 'productionDate' in js_data:
                    val = str(js_data['productionDate'])
                    y_match = re.search(year_pattern, val)
                    if y_match: data['year'] = int(y_match.group(1))
                
                # Power
                if 'vehicleEngine' in js_data and isinstance(js_data['vehicleEngine'], dict):
                     pwr_match = re.search(r'(\d+)', str(js_data['vehicleEngine'].get('enginePower', '')))
                     if pwr_match: data['hp'] = int(pwr_match.group(1))

                # Distance (Mileage)
                if 'mileageFromOdometer' in js_data:
                    val = str(js_data['mileageFromOdometer'])
                    data['km'] = int(re.sub(r'[^\d]', '', val))
                    msg_extra.append("JSON")
            except: continue

        # --- B. Metadata ---
        title = soup.find("meta", property="og:title")
        desc = soup.find("meta", property="og:description")
        meta_content = ((title['content'] if title else "") + " " + (desc['content'] if desc else "")).lower()
        
        if 'year' not in data:
            y_match = re.search(year_pattern, meta_content)
            if y_match: data['year'] = int(y_match.group(1))

        # --- C. Visible Text (Scanning the page) ---
        page_text = soup.get_text(separator=' ', strip=True).lower()[:15000]
        
        # 1. Distance / Odometer Scraping (Smart Match)
        if 'km' not in data:
            # Case 1: "Mileage: 50,000" (eBay Style - usually Miles)
            mile_explicit = re.search(r'mileage[:\s]+([\d,]+)', page_text)
            
            # Case 2: "50,000 kms" (Indian Style - CarDekho/OLX)
            # We look for digits followed specifically by 'km' or 'kms'
            km_explicit = re.search(r'(\d{2,}[\d,]*)\s*(kms?|kilometers?)', page_text)
            
            if mile_explicit:
                # If explicit "Mileage" label found, assume Miles (US/UK site)
                val = int(mile_explicit.group(1).replace(',', ''))
                data['km'] = int(val * 1.60934) # Convert to KM
                msg_extra.append("Miles->KM")
            elif km_explicit:
                # If "kms" found, take it directly
                data['km'] = int(km_explicit.group(1).replace(',', ''))
                msg_extra.append("KMs Found")

        # 2. Year & Month
        if 'year' not in data:
            y_match = re.search(year_pattern, page_text)
            if y_match: data['year'] = int(y_match.group(1))
            
        if 'year' in data:
            # Look for month pattern like 05/2015 or May 2015
            year_val = data['year']
            month_match = re.search(r'(\d{1,2})[/-]' + str(year_val), page_text)
            if month_match:
                m = int(month_match.group(1))
                if 1 <= m <= 12: data['month'] = m
        
        # 3. Power
        if 'hp' not in data:
            hp_match = re.search(r'(\d{2,4})\s*(hp|ps|cv|bhp)', page_text)
            if hp_match: data['hp'] = int(hp_match.group(1))

        if not data:
            return None, "No Data"
            
        return data, f"✅ Scraped! {', '.join(msg_extra)}"

    except:
        return None, "Error"

# --- 5. LAYOUT ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Car Price Predictor"

app.layout = dbc.Container([
    dbc.Row(dbc.Col([
        html.H1("Car Price Predictor", className="display-4 text-center mt-4 fw-bold text-primary"),
        html.P("Intelligent Vehicle Valuation", className="text-center text-muted mb-5")
    ])),

    dbc.Row([
        # LEFT: Inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Vehicle Details", className="mb-0")),
                dbc.CardBody([
                    html.Label("Auto-Fill from URL"),
                    dbc.InputGroup([
                        dbc.Input(id='url-input', placeholder="Paste listing URL..."),
                        dbc.Button("Scrape", id='btn-scrape', color="dark")
                    ], className="mb-2"),
                    html.Div(id='scrape-status', className="small text-danger fw-bold mb-3"),
                    html.Hr(),
                    
                    dbc.Row([
                        dbc.Col([html.Label("Brand"), dcc.Dropdown(id='brand', options=unique_brands)], width=6),
                        dbc.Col([html.Label("Model"), dcc.Dropdown(id='model')], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([html.Label("Year"), dbc.Input(id='year', type='number')], width=6),
                        dbc.Col([html.Label("Month"), dbc.Input(id='month', type='number')], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([html.Label("Power (HP)"), dbc.Input(id='power', type='number')], width=6),
                        # Renamed label to avoid confusion
                        dbc.Col([html.Label("Kilometers Driven"), dbc.Input(id='km', type='number')], width=6),
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([html.Label("Fuel"), dcc.Dropdown(id='fuel', options=unique_fuels)], width=4),
                        dbc.Col([html.Label("Type"), dcc.Dropdown(id='type', options=unique_types)], width=4),
                        dbc.Col([html.Label("Gearbox"), dcc.Dropdown(id='gear', options=['manual', 'auto'])], width=4),
                    ], className="mb-4"),
                    
                    dbc.Button("PREDICT VALUE", id='btn-predict', color="primary", size="lg", className="w-100 shadow-sm")
                ])
            ], className="shadow-sm border-0")
        ], lg=5, md=12),

        # RIGHT: Output
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("ESTIMATED MARKET VALUE", className="text-center text-muted small"),
                    html.Div(id='price-eur', className="text-center display-4 text-success fw-bold"),
                    html.Div(id='price-inr', className="text-center h4 text-secondary mb-4"),
                    dcc.Graph(id='gauge-chart', style={'height': '220px'}),
                    html.Hr(),
                    html.H6("DEPRECIATION FORECAST", className="text-center mt-3 small text-muted"),
                    dcc.Graph(id='depreciation-chart', style={'height': '250px'})
                ])
            ], className="h-100 shadow border-0")
        ], lg=7, md=12)
    ]),
    dcc.Store(id='exchange-rate', data=get_exchange_rate())
], fluid=True, className="bg-light min-vh-100 pb-5")

# --- 6. CALLBACKS ---

@app.callback(Output('model', 'options'), Input('brand', 'value'))
def update_models(brand):
    if not brand: return []
    models = sorted(df_clean[df_clean['Brand'] == brand]['Model'].unique())
    return [{'label': m.title(), 'value': m} for m in models]

@app.callback(
    [Output('price-eur', 'children'), Output('price-inr', 'children'),
     Output('gauge-chart', 'figure'), Output('depreciation-chart', 'figure'),
     Output('scrape-status', 'children'), 
     # Inputs
     Output('brand', 'value'), Output('model', 'value'),
     Output('year', 'value'), Output('month', 'value'),
     Output('power', 'value'), Output('km', 'value'),
     Output('fuel', 'value'), Output('type', 'value'), Output('gear', 'value')],
    [Input('btn-predict', 'n_clicks'), Input('btn-scrape', 'n_clicks')],
    [State('url-input', 'value'), State('exchange-rate', 'data'),
     State('brand', 'value'), State('model', 'value'),
     State('year', 'value'), State('month', 'value'),
     State('power', 'value'), State('km', 'value'),
     State('fuel', 'value'), State('type', 'value'), State('gear', 'value')]
)
def main_logic(n_pred, n_scrape, url, rate, brand, model, year, month, hp, km, fuel, v_type, gear):
    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Zero State
    zero_fig = go.Figure(go.Indicator(mode="gauge+number", value=0))
    zero_fig.update_layout(margin=dict(t=30, b=10, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
    empty_plot = go.Figure()
    empty_plot.update_layout(paper_bgcolor='rgba(0,0,0,0)', xaxis_title="Year", yaxis_title="Value")
    
    # 1. SCRAPE
    if trigger == 'btn-scrape':
        if not url: return "0", "0", zero_fig, empty_plot, "Enter URL", None, None, None, None, None, None, None, None, None
        
        data, msg = scrape_listing(url)
        if not data:
            # Clear all if invalid
            return "0", "0", zero_fig, empty_plot, "Invalid Link (or URL)", None, None, None, None, None, None, None, None, None
            
        # Update found fields
        return "0", "0", zero_fig, empty_plot, msg, brand, model, data.get('year', year), data.get('month', month), data.get('hp', hp), data.get('km', km), fuel, v_type, gear

    # 2. PREDICT
    if trigger == 'btn-predict':
        try:
            input_df = pd.DataFrame({
                'RegistrationYear': [year], 'RegistrationMonth': [month if month else 6],
                'Power': [hp], 'Kilometer': [km], 'Brand': [brand],
                'Model': [model if model else 'unknown'], 'FuelType': [fuel],
                'VehicleType': [v_type], 'Gearbox': [gear], 'Repaired': ['no']
            })
            pred_eur = pipeline.predict(input_df)[0]
            pred_inr = pred_eur * rate
            
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=pred_eur, gauge={'axis': {'range': [pred_eur*0.6, pred_eur*1.4]}, 'bar': {'color': "#2c3e50"}}))
            gauge.update_layout(margin=dict(t=30, b=10, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
            
            years, values = [], []
            temp_df = input_df.copy()
            for i in range(6):
                years.append(date.today().year + i)
                temp_df['RegistrationYear'] = int(year) - i 
                temp_df['Kilometer'] = int(km) + (15000 * i)
                try: val = pipeline.predict(temp_df)[0]
                except: val = 0
                values.append(max(val, 0))
            
            dep_chart = go.Figure(go.Scatter(x=years, y=values, fill='tozeroy', line=dict(color='#3498db')))
            dep_chart.update_layout(margin=dict(t=10, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
            
            return f"€ {pred_eur:,.0f}", f"₹ {pred_inr:,.0f}", gauge, dep_chart, "", brand, model, year, month, hp, km, fuel, v_type, gear
        except Exception:
             return "0", "0", zero_fig, empty_plot, "Fill Inputs", brand, model, year, month, hp, km, fuel, v_type, gear

    return "0", "0", zero_fig, empty_plot, "", None, None, None, None, None, None, None, None, None

if __name__ == '__main__':
    app.run(debug=True)