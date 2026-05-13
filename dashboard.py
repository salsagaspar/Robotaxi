import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np

# ==========================================
# 1. DATA PREPARATION (Sesuai EDA kita)
# ==========================================
print("Loading data for dashboard...")
rides = pd.read_csv('robotaxi_rides.csv')
users = pd.read_csv('robotaxi_users.csv')

# Cleaning & Merging
rides['user_rating'] = rides['user_rating'].fillna(-1)
df = pd.merge(rides, users, on='user_id', how='left')

# Feature Engineering
df['total_revenue'] = df['fare_amount_usd'] + df['tip_amount_usd']
df['tip_percentage'] = np.where(df['fare_amount_usd'] > 0, (df['tip_amount_usd'] / df['fare_amount_usd']) * 100, 0)
df['surge_bucket'] = pd.cut(df['surge_multiplier'], bins=[0.9, 1.1, 1.5, 2.0, 5.0], labels=['1.0x (Normal)', '1.2x - 1.5x', '1.6x - 2.0x', '> 2.0x'])

# ==========================================
# 2. MENGHASILKAN GRAFIK (PLOTLY EXPRESS)
# ==========================================
# TEMA GELAP UNTUK GRAFIK
plotly_template = "plotly_dark"
color_palette = px.colors.qualitative.Pastel

# GRAFIK 1: Ride Status Distribution (Donut Chart)
status_counts = df['ride_status'].value_counts(normalize=True).reset_index()
status_counts.columns = ['Ride Status', 'Percentage']
fig_status = px.pie(status_counts, values='Percentage', names='Ride Status', hole=0.4, 
                    title='Distribusi Status Perjalanan', color_discrete_sequence=color_palette)
fig_status.update_layout(template=plotly_template, title_x=0.5)

# GRAFIK 2: Wait Time by Status (Bar Chart)
wait_time = df.groupby('ride_status')['wait_time_minutes'].mean().reset_index()
fig_wait = px.bar(wait_time, x='ride_status', y='wait_time_minutes', color='ride_status', 
                  title='Rata-rata Waktu Tunggu (Menit) per Status', color_discrete_sequence=color_palette)
fig_wait.update_layout(template=plotly_template, title_x=0.5, showlegend=False)

# GRAFIK 3: Revenue by Loyalty Tier (Bar Chart)
completed = df[df['ride_status'] == 'Completed']
revenue_tier = completed.groupby('loyalty_tier')['total_revenue'].mean().reset_index()
tier_order = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
fig_revenue = px.bar(revenue_tier, x='loyalty_tier', y='total_revenue', 
                     category_orders={'loyalty_tier': tier_order}, 
                     title='Rata-rata Pendapatan per Tier Loyalitas ($)', 
                     color='loyalty_tier', color_discrete_sequence=px.colors.sequential.matter)
fig_revenue.update_layout(template=plotly_template, title_x=0.5, showlegend=False)

# GRAFIK 4: Cancellation Rate vs Surge (Line Chart)
surge_status = df.groupby('surge_bucket', observed=False)['ride_status'].value_counts(normalize=True).unstack() * 100
cancel_rates = surge_status['Cancelled'].reset_index()
fig_surge = px.line(cancel_rates, x='surge_bucket', y='Cancelled', markers=True, 
                    title='Tingkat Pembatalan vs Lonjakan Harga', 
                    color_discrete_sequence=['#ff4b4b'])
fig_surge.update_traces(line=dict(width=3), marker=dict(size=10))
fig_surge.update_layout(template=plotly_template, title_x=0.5, yaxis_title="Pembatalan (%)", xaxis_title="Surge Multiplier")


# ==========================================
# 3. MEMBANGUN LAYOUT DASHBOARD (DASH)
# ==========================================
# Menggunakan tema CYBORG (Dark Theme Premium) dari Dash Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    # HEADER
    dbc.Row([
        dbc.Col(html.Div([
            html.H1("ROBOTAXI ANALYTICS", className="text-center mt-4 mb-1", style={"fontWeight": "bold", "color": "#00d2ff"}),
            html.P("Executive Business Dashboard | Operational & Revenue Insights", className="text-center mb-4 text-muted")
        ]), width=12)
    ]),
    
    # ROW 1: STATUS & WAKTU TUNGGU
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_status))], className="mb-4", style={"backgroundColor": "#111111"}), md=6),
        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_wait))], className="mb-4", style={"backgroundColor": "#111111"}), md=6)
    ]),
    
    # ROW 2: PENDAPATAN & PRICING
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_revenue))], className="mb-4", style={"backgroundColor": "#111111"}), md=6),
        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_surge))], className="mb-4", style={"backgroundColor": "#111111"}), md=6)
    ])
], fluid=True, style={"backgroundColor": "#000000", "minHeight": "100vh"})

# ==========================================
# 4. RUN SERVER
# ==========================================
if __name__ == '__main__':
    print("Membuka Dashboard Robotaxi... Silakan klik link http://127.0.0.1:8050 di terminal Anda.")
    app.run(debug=True, port=8050)
