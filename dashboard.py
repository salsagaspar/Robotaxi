import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

# ==========================================
# 1. LOAD DATA & MODELS
# ==========================================
print("Loading data and models for dashboard...")
rides = pd.read_csv('robotaxi_rides.csv')
users = pd.read_csv('robotaxi_users.csv')

# EDA Data Prep
rides['user_rating'] = rides['user_rating'].fillna(-1)
df = pd.merge(rides, users, on='user_id', how='left')
df['total_revenue'] = df['fare_amount_usd'] + df['tip_amount_usd']
df['surge_bucket'] = pd.cut(df['surge_multiplier'], bins=[0.9, 1.1, 1.5, 2.0, 5.0], labels=['1.0x (Normal)', '1.2x - 1.5x', '1.6x - 2.0x', '> 2.0x'])

plotly_template = "plotly_dark"
color_palette = px.colors.qualitative.Pastel

# Generate static charts for Tab 1
status_counts = df['ride_status'].value_counts(normalize=True).reset_index()
status_counts.columns = ['Ride Status', 'Percentage']
fig_status = px.pie(status_counts, values='Percentage', names='Ride Status', hole=0.4, title='Distribusi Status Perjalanan', color_discrete_sequence=color_palette)
fig_status.update_layout(template=plotly_template, title_x=0.5)

wait_time = df.groupby('ride_status')['wait_time_minutes'].mean().reset_index()
fig_wait = px.bar(wait_time, x='ride_status', y='wait_time_minutes', color='ride_status', title='Rata-rata Waktu Tunggu (Menit) per Status', color_discrete_sequence=color_palette)
fig_wait.update_layout(template=plotly_template, title_x=0.5, showlegend=False)

completed = df[df['ride_status'] == 'Completed']
revenue_tier = completed.groupby('loyalty_tier')['total_revenue'].mean().reset_index()
tier_order = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
fig_revenue = px.bar(revenue_tier, x='loyalty_tier', y='total_revenue', category_orders={'loyalty_tier': tier_order}, title='Rata-rata Pendapatan per Tier Loyalitas ($)', color='loyalty_tier', color_discrete_sequence=px.colors.sequential.matter)
fig_revenue.update_layout(template=plotly_template, title_x=0.5, showlegend=False)

surge_status = df.groupby('surge_bucket', observed=False)['ride_status'].value_counts(normalize=True).unstack() * 100
cancel_rates = surge_status['Cancelled'].reset_index()
fig_surge = px.line(cancel_rates, x='surge_bucket', y='Cancelled', markers=True, title='Tingkat Pembatalan vs Lonjakan Harga', color_discrete_sequence=['#ff4b4b'])
fig_surge.update_traces(line=dict(width=3), marker=dict(size=10))
fig_surge.update_layout(template=plotly_template, title_x=0.5, yaxis_title="Pembatalan (%)", xaxis_title="Surge Multiplier")

# Load ML Models
try:
    model_demand = joblib.load('models/demand_model.pkl')
    le_city = joblib.load('models/le_city.pkl')
    model_price = joblib.load('models/pricing_model.pkl')
    model_maint = joblib.load('models/maintenance_model.pkl')
    model_churn = joblib.load('models/churn_model.pkl')
    model_eta = joblib.load('models/eta_model.pkl')
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False

cities = rides['pickup_city'].unique() if 'pickup_city' in rides.columns else ['Denver', 'Miami', 'Chicago', 'New York']

# ==========================================
# 2. LAYOUT DASHBOARD
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

tab1_content = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_status))], className="mb-4 mt-4", style={"backgroundColor": "#111111"}), md=6),
        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_wait))], className="mb-4 mt-4", style={"backgroundColor": "#111111"}), md=6)
    ]),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_revenue))], className="mb-4", style={"backgroundColor": "#111111"}), md=6),
        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_surge))], className="mb-4", style={"backgroundColor": "#111111"}), md=6)
    ])
])

tab2_content = dbc.Container([
    dbc.Row([
        # Model 1: Demand Forecasting
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H4("📈 Demand Forecasting (24h)", className="text-info")),
            dbc.CardBody([
                html.Label("Pilih Kota:"),
                dcc.Dropdown(id='demand-city-dropdown', options=[{'label': c, 'value': c} for c in cities], value=cities[0], className="text-dark mb-3"),
                dcc.Graph(id='demand-graph')
            ])
        ], className="mb-4 mt-4", style={"backgroundColor": "#111111"}), md=12),
    ]),
    
    dbc.Row([
        # Model 2: Dynamic Pricing
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("💰 Dynamic Pricing Predictor", className="text-warning")),
            dbc.CardBody([
                html.Label("Jarak (Miles):"), dcc.Input(id='price-dist', type='number', value=5, className="form-control mb-2"),
                html.Label("Durasi (Menit):"), dcc.Input(id='price-dur', type='number', value=15, className="form-control mb-2"),
                html.Label("Jam Pickup:"), dcc.Slider(0, 23, 1, value=12, id='price-hour', marks={i: str(i) for i in range(0,24,4)}, className="mb-4"),
                dbc.Button("Hitung Harga", id="btn-price", color="warning", className="mt-2 w-100"),
                html.H3(id='out-price', className="text-center mt-3 text-success")
            ])
        ], className="mb-4", style={"backgroundColor": "#111111"}), md=4),
        
        # Model 5: ETA Predictor
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("⏱️ ETA Predictor", className="text-success")),
            dbc.CardBody([
                html.Label("Jarak (Miles):"), dcc.Input(id='eta-dist', type='number', value=5, className="form-control mb-2"),
                html.Label("Jam Pickup:"), dcc.Slider(0, 23, 1, value=12, id='eta-hour', marks={i: str(i) for i in range(0,24,4)}, className="mb-4"),
                dbc.Button("Prediksi ETA", id="btn-eta", color="success", className="mt-2 w-100"),
                html.H3(id='out-eta', className="text-center mt-3 text-info")
            ])
        ], className="mb-4", style={"backgroundColor": "#111111"}), md=4),

        # Model 3: Predictive Maintenance
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("🛠️ Predictive Maintenance", className="text-danger")),
            dbc.CardBody([
                html.Label("Battery Level (%):"), dcc.Input(id='maint-batt', type='number', value=80, className="form-control mb-2"),
                html.Label("Mileage since last service:"), dcc.Input(id='maint-mil', type='number', value=2000, className="form-control mb-2"),
                html.Label("Sensor Warnings:"), dcc.Input(id='maint-warn', type='number', value=0, className="form-control mb-2"),
                html.Label("Tire Pressure (PSI):"), dcc.Input(id='maint-tire', type='number', value=32, className="form-control mb-2"),
                dbc.Button("Cek Status", id="btn-maint", color="danger", className="mt-2 w-100"),
                html.H4(id='out-maint', className="text-center mt-3")
            ])
        ], className="mb-4", style={"backgroundColor": "#111111"}), md=4),
    ]),
    
    dbc.Row([
        # Model 4: Churn Prediction
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("📉 Customer Churn Risk", className="text-primary")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([html.Label("Umur:"), dcc.Input(id='churn-age', type='number', value=25, className="form-control mb-2")]),
                    dbc.Col([html.Label("Total Rides:"), dcc.Input(id='churn-rides', type='number', value=10, className="form-control mb-2")]),
                    dbc.Col([html.Label("Promo Balance:"), dcc.Input(id='churn-promo', type='number', value=5.0, className="form-control mb-2")])
                ]),
                dbc.Button("Prediksi Churn", id="btn-churn", color="primary", className="mt-2 w-100"),
                html.H3(id='out-churn', className="text-center mt-3")
            ])
        ], className="mb-4", style={"backgroundColor": "#111111"}), md=12)
    ])
])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.H1("ROBOTAXI AI & ANALYTICS", className="text-center mt-4 mb-1", style={"fontWeight": "bold", "color": "#00d2ff"}),
            html.P("Executive Dashboard & Machine Learning Control Center", className="text-center mb-4 text-muted")
        ]), width=12)
    ]),
    
    dbc.Tabs([
        dbc.Tab(tab1_content, label="📊 Executive Insights", tab_style={"backgroundColor": "#222"}, active_tab_style={"backgroundColor": "#000", "color": "#00d2ff"}),
        dbc.Tab(tab2_content, label="🤖 AI Models (Predictive)", tab_style={"backgroundColor": "#222"}, active_tab_style={"backgroundColor": "#000", "color": "#ff4b4b"}),
    ])
], fluid=True, style={"backgroundColor": "#000000", "minHeight": "100vh"})


# ==========================================
# 3. CALLBACKS UNTUK AI MODELS
# ==========================================
if models_loaded:
    @app.callback(
        Output('demand-graph', 'figure'),
        Input('demand-city-dropdown', 'value')
    )
    def update_demand(city):
        city_enc = le_city.transform([city])[0]
        # Simulate next 24 hours starting from hour 0, day 0
        hours = list(range(24))
        days = [0] * 24 # Monday
        X = pd.DataFrame({'city_encoded': [city_enc]*24, 'pickup_hour': hours, 'pickup_dayofweek': days})
        preds = model_demand.predict(X)
        
        fig = px.line(x=hours, y=preds, markers=True, title=f"Prediksi Permintaan Taksi dalam 24 Jam ke Depan di {city}",
                      labels={'x': 'Jam (0-23)', 'y': 'Jumlah Permintaan / Rides'}, color_discrete_sequence=['#00d2ff'])
        fig.update_layout(template=plotly_template)
        return fig

    @app.callback(
        Output('out-price', 'children'),
        Input('btn-price', 'n_clicks'),
        State('price-dist', 'value'), State('price-dur', 'value'), State('price-hour', 'value')
    )
    def predict_price(n, dist, dur, hour):
        if n is None: return ""
        # Default city 0
        X = pd.DataFrame({'distance_miles': [dist], 'duration_minutes': [dur], 'pickup_hour': [hour], 'city_encoded': [0]})
        pred = model_price.predict(X)[0]
        return f"${pred:.2f}"

    @app.callback(
        Output('out-eta', 'children'),
        Input('btn-eta', 'n_clicks'),
        State('eta-dist', 'value'), State('eta-hour', 'value')
    )
    def predict_eta(n, dist, hour):
        if n is None: return ""
        X = pd.DataFrame({'distance_miles': [dist], 'pickup_hour': [hour], 'city_encoded': [0]})
        pred = model_eta.predict(X)[0]
        return f"{int(pred)} Menit"

    @app.callback(
        Output('out-maint', 'children'),
        Output('out-maint', 'className'),
        Input('btn-maint', 'n_clicks'),
        State('maint-batt', 'value'), State('maint-mil', 'value'), State('maint-warn', 'value'), State('maint-tire', 'value')
    )
    def predict_maint(n, batt, mil, warn, tire):
        if n is None: return "", "text-center mt-3"
        X = pd.DataFrame({'battery_level': [batt], 'mileage_since_last_service': [mil], 'sensor_warning_count': [warn], 'tire_pressure_psi': [tire]})
        pred = model_maint.predict(X)[0]
        if pred == 1:
            return "⚠️ Butuh Perawatan", "text-center mt-3 text-danger font-weight-bold"
        return "✅ Kondisi Aman", "text-center mt-3 text-success"

    @app.callback(
        Output('out-churn', 'children'),
        Output('out-churn', 'className'),
        Input('btn-churn', 'n_clicks'),
        State('churn-age', 'value'), State('churn-rides', 'value'), State('churn-promo', 'value')
    )
    def predict_churn(n, age, rides, promo):
        if n is None: return "", "text-center mt-3"
        X = pd.DataFrame({'age': [age], 'total_rides_taken': [rides], 'promo_credits_balance': [promo]})
        pred = model_churn.predict(X)[0]
        if pred == 1:
            return "🔴 Berisiko Churn (Meninggalkan Layanan)", "text-center mt-3 text-danger"
        return "🟢 Pelanggan Setia", "text-center mt-3 text-success"

if __name__ == '__main__':
    print("Membuka Dashboard Robotaxi... Silakan klik link http://127.0.0.1:8050 di terminal Anda.")
    app.run(debug=True, port=8050)
