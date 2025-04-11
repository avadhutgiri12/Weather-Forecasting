
import pandas as pd
import joblib
from datetime import datetime

# Load models
model = joblib.load('./weather_model.pkl')
model2 = joblib.load('./humidity_model.pkl')
model3 = joblib.load('./wind_speed_model.pkl')
model4 = joblib.load('./precipitation_model.pkl')

# Load data
data = pd.read_csv('../data/large_weather_data.csv')

# Convert datetime
data['date_time'] = pd.to_datetime(data['date_time'], format='%d-%m-%Y %H:%M', errors='coerce')

# Extract features
data['hour'] = data['date_time'].dt.hour
data['month'] = data['date_time'].dt.month

# Prepare input features
features = ['wind_speed', 'humidity', 'hour', 'month']
X = data[features].fillna(data[features].mean())

# Predict weather metrics
data['predicted_temp'] = model.predict(X).round(0)
data['predicted_humidity'] = model2.predict(X).round(0)
data['predicted_wind_speed'] = model3.predict(X).round(1)
data['predicted_precipitation'] = model4.predict(X).round(1)

# --- Define forecasted weather label ---
def categorize_weather(temp, precip, clouds):
    if precip > 10:
        return "rainy"
    elif clouds > 70:
        return "cloudy"
    elif temp > 30:
        return "hot"
    elif temp < 5:
        return "cold"
    else:
        return "clear"

# Apply categorization
data['forecasted_weather'] = data.apply(
    lambda row: categorize_weather(
        row['predicted_temp'],
        row['predicted_precipitation'],
        row.get('cloud_coverage', 50)  # use default if cloud_coverage is missing
    ),
    axis=1
)

# Save to CSV
data.to_csv('../data/predicted_weather_data.csv', index=False)
print("✅ Forecasted data saved to predicted_weather_data.csv")
