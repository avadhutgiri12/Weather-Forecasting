import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load and prepare data
df = pd.read_csv('../data/all_weather_data.csv')

df['Date'] = pd.to_datetime(df['Date'],format='%d-%m-%Y')
df['hour'] = df['Date'].dt.hour
df['month'] = df['Date'].dt.month

X = df[[ 'wind','Humidity', 'hour', 'month']]
y = df['temprature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, 'weather_model.pkl')
