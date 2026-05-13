import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
rides_df = pd.read_csv('robotaxi_rides.csv')
users_df = pd.read_csv('robotaxi_users.csv')

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# ---------------------------------------------------------
# 1. Demand Forecasting Model
# ---------------------------------------------------------
print("Training Demand Forecasting Model...")
rides_df['pickup_datetime'] = pd.to_datetime(rides_df['pickup_datetime'])
rides_df['pickup_hour'] = rides_df['pickup_datetime'].dt.hour
rides_df['pickup_dayofweek'] = rides_df['pickup_datetime'].dt.dayofweek
rides_df['pickup_date'] = rides_df['pickup_datetime'].dt.date

# Aggregate rides by date, hour, and city
demand_df = rides_df.groupby(['pickup_city', 'pickup_date', 'pickup_hour', 'pickup_dayofweek']).size().reset_index(name='demand')

# Encode city
le_city = LabelEncoder()
demand_df['city_encoded'] = le_city.fit_transform(demand_df['pickup_city'])

X_demand = demand_df[['city_encoded', 'pickup_hour', 'pickup_dayofweek']]
y_demand = demand_df['demand']

model_demand = RandomForestRegressor(n_estimators=50, random_state=42)
model_demand.fit(X_demand, y_demand)
joblib.dump(model_demand, 'models/demand_model.pkl')
joblib.dump(le_city, 'models/le_city.pkl')


# ---------------------------------------------------------
# 2. Dynamic / Surge Pricing Model
# ---------------------------------------------------------
print("Training Dynamic Pricing Model...")
# Features: distance, duration, hour, city
pricing_df = rides_df.dropna(subset=['distance_miles', 'duration_minutes', 'fare_amount_usd']).copy()
pricing_df['city_encoded'] = le_city.transform(pricing_df['pickup_city'])

X_price = pricing_df[['distance_miles', 'duration_minutes', 'pickup_hour', 'city_encoded']]
y_price = pricing_df['fare_amount_usd']

model_price = RandomForestRegressor(n_estimators=50, random_state=42)
model_price.fit(X_price, y_price)
joblib.dump(model_price, 'models/pricing_model.pkl')


# ---------------------------------------------------------
# 3. Predictive Maintenance Model
# ---------------------------------------------------------
print("Training Predictive Maintenance Model...")
# We need to simulate some vehicle data since we don't have it
np.random.seed(42)
vehicles = rides_df['ride_id'].unique()[:5000] # Use some ride_ids as vehicle_ids for simulation
maintenance_df = pd.DataFrame({
    'vehicle_id': vehicles,
    'battery_level': np.random.randint(10, 100, size=len(vehicles)),
    'mileage_since_last_service': np.random.randint(100, 10000, size=len(vehicles)),
    'sensor_warning_count': np.random.randint(0, 10, size=len(vehicles)),
    'tire_pressure_psi': np.random.uniform(28, 40, size=len(vehicles))
})

# Define a logic for needs_maintenance: low battery + high mileage + warnings
def check_maintenance(row):
    if row['sensor_warning_count'] > 7 or (row['mileage_since_last_service'] > 8000 and row['battery_level'] < 30):
        return 1
    return 0

maintenance_df['needs_maintenance'] = maintenance_df.apply(check_maintenance, axis=1)

X_maint = maintenance_df[['battery_level', 'mileage_since_last_service', 'sensor_warning_count', 'tire_pressure_psi']]
y_maint = maintenance_df['needs_maintenance']

model_maint = RandomForestClassifier(n_estimators=50, random_state=42)
model_maint.fit(X_maint, y_maint)
joblib.dump(model_maint, 'models/maintenance_model.pkl')


# ---------------------------------------------------------
# 4. Customer Churn Prediction
# ---------------------------------------------------------
print("Training Customer Churn Prediction Model...")
users_df['last_active_date'] = pd.to_datetime(users_df['last_active_date'])
max_date = users_df['last_active_date'].max()

# Define churn: inactive for > 180 days
users_df['days_inactive'] = (max_date - users_df['last_active_date']).dt.days
users_df['churned'] = (users_df['days_inactive'] > 180).astype(int)

# Features
features_churn = ['age', 'total_rides_taken', 'promo_credits_balance']
X_churn = users_df[features_churn].fillna(0)
y_churn = users_df['churned']

model_churn = RandomForestClassifier(n_estimators=50, random_state=42)
model_churn.fit(X_churn, y_churn)
joblib.dump(model_churn, 'models/churn_model.pkl')


# ---------------------------------------------------------
# 5. ETA Prediction Model
# ---------------------------------------------------------
print("Training ETA Prediction Model...")
# Filter out cancelled rides for ETA
eta_df = rides_df[rides_df['ride_status'] == 'Completed'].dropna(subset=['distance_miles', 'duration_minutes']).copy()
eta_df['city_encoded'] = le_city.transform(eta_df['pickup_city'])

X_eta = eta_df[['distance_miles', 'pickup_hour', 'city_encoded']]
y_eta = eta_df['duration_minutes']

model_eta = RandomForestRegressor(n_estimators=50, random_state=42)
model_eta.fit(X_eta, y_eta)
joblib.dump(model_eta, 'models/eta_model.pkl')

print("All 5 models have been successfully trained and saved in the 'models/' directory.")
