import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load and prepare data
df = pd.read_csv('../data/large_weather_data.csv')
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')

# Feature engineering
df['hour'] = df['date_time'].dt.hour
df['month'] = df['date_time'].dt.month

# Common features
features = ['wind_speed', 'humidity', 'hour', 'month']

# Target columns
targets = ['temperature', 'humidity', 'wind_speed', 'precipitation']

# Train and evaluate models
for target in targets:
    print(f"\n🔧 Training model for: {target}")
    
    X = df[features]
    y = df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
    
    # Save model
    model_filename = f"{target}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"✅ Model saved as: {model_filename}")
