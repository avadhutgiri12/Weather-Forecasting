## 🌦️ Weather Forecasting System with Supaboard Dashboard

## 📌 Problem Statement

Accurate and timely weather forecasts are essential for numerous sectors including agriculture, transportation, urban planning, and disaster management. Traditional methods often fall short in adapting to dynamic weather patterns, resulting in inaccurate predictions. This project leverages machine learning models to enhance forecasting accuracy and integrates the predictions into a real-time interactive dashboard using **Supaboard**.

---

## 🧠 Key Features

### 1. **Machine Learning Forecasting**
- Historical weather data is used to train ML models that forecast:
  - 🌡️ Temperature
  - 💧 Humidity
  - 🌬️ Wind Speed
  - 🌧️ Precipitation
  - 🌂 Rain Probability (Classification)

### 2. **Real-time Weather Data Ingestion**
- Uses **OpenWeatherMap API** to fetch current weather data for a specified city.
- Combines this with historical data (`weather.csv`) for improved prediction accuracy.

### 3. **Automated Scheduling**
- The main script is scheduled to run every **5 hours** using the `schedule` module.
- At each run:
  - Current weather is fetched.
  - ML models predict weather features for the next 7 days.
  - Results are stored in `weather_actual_vs_predicted.csv`.

### 4. **Dashboard Visualization (Supaboard)**
- The generated CSV file is automatically linked to a **Supaboard** dashboard.
- The following forecasted and real-time parameters are visualized:
  - 📈 Temperature
  - 💨 Wind Speed
  - 💦 Humidity
  - 🌧️ Precipitation
- Supaboard generates **interactive line graphs** for each parameter, making it easy to compare actual vs. predicted trends across days.

---

## 📊 Dataset Overview

### 📁 Input File: `weather.csv`
Historical weather data used for model training.

| Column Name     | Description                              |
|------------------|------------------------------------------|
| `MinTemp`        | Minimum temperature recorded (°C)        |
| `MaxTemp`        | Maximum temperature recorded (°C)        |
| `WindGustDir`    | Wind direction (compass)                 |
| `WindGustSpeed`  | Wind speed (km/h)                        |
| `Humidity`       | Relative humidity (%)                    |
| `Pressure`       | Atmospheric pressure (hPa)               |
| `Temp`           | Average temperature (°C)                 |
| `Precipitation`  | Rainfall in mm                           |
| `RainTomorrow`   | Target label - Will it rain tomorrow?    |

### 📁 Output File: `weather_actual_vs_predicted.csv`
Predictions saved for visualization.

| Column Name                | Description                                  |
|----------------------------|----------------------------------------------|
| `date_time`                | Forecasted date and time                     |
| `actual_temperature`       | Current temperature from API                 |
| `predicted_temperature`    | Predicted temperature using ML model         |
| `actual_humidity`          | Current humidity from API                    |
| `predicted_humidity`       | Predicted humidity                           |
| `actual_wind_speed`        | Wind speed from API                          |
| `predicted_wind_speed`     | Predicted wind speed                         |
| `precipitation (mm)`       | Observed rainfall from API                   |
| `predicted_precipitation`  | Predicted rainfall                           |
| `rain_prediction`          | Will it rain? (Yes/No)                       |
| `weather_prediction`       | Description (e.g., clear, cloudy, rain)      |

---

## ⚙️ How to Run

### Requirements
- Python 3.7+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `schedule`, `requests`, `pytz`

### Steps

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the main forecasting script
python weather_forecasting.py
```

> Make sure to replace the `API_KEY` with your valid OpenWeatherMap API key.

---

## 📈 Dashboard Integration (Supaboard)

1. After each script execution, `weather_actual_vs_predicted.csv` is updated.
2. This file is read by **Supaboard** to auto-update the dashboard.
3. Graphs are generated for each feature:
   - Actual vs Predicted Temperature
   - Actual vs Predicted Humidity
   - Actual vs Predicted Wind Speed
   - Actual vs Predicted Precipitation

Supaboard provides a clean, interactive UI to monitor and analyze weather trends effortlessly.

---

## ⚙️ Key Features & Technologies

### 🚀 Features
- Real-time data collection using OpenWeatherMap API
- Machine Learning models for:
  - Rain prediction (classification)
  - Forecasting temperature, humidity, wind speed, precipitation (regression)
- Predicts weather for the next **7 days**
- Automated scheduling every **5 hours**
- CSV output ready for dashboard integration
- Dashboard-ready format with Supaboard compatibility

### 🧰 Technologies Used
- **Python** (Core Language)
- **pandas**, **numpy** (Data Manipulation)
- **scikit-learn** (Machine Learning)
- **schedule** (Task Scheduling)
- **OpenWeatherMap API** (Real-time data)
- **Supaboard** (Dashboard Visualization)
- **pytz**, **datetime** (Timezone Handling)

## 📂 Project Structure

```
.
├── data/
│   ├── weather.csv                      # Historical weather data
│   └── weather_actual_vs_predicted.csv # Model predictions
├── weather_forecasting.py              # Main script for training and forecasting
├── README.md                           # Project documentation
```

---

## Drive Link :- 
https://drive.google.com/drive/folders/1MAeiqxCZWImd24Zccbw65tDF1t0RXQrW?usp=sharing
Dashboard : https://app.supaboard.ai/dashboard/weather-data
WebApp : https://weather-ap.streamlit.app/

## Contribution :-

 ### Avdhut giri
 ### Aditya Karanje
 ### Nishant.M
