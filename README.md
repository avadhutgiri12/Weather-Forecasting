
# Weather Forecasting System

Dashboard : https://app.supaboard.ai/dashboard/weather-data

## 📌 Problem Statement

Accurate weather forecasting is critical across industries such as agriculture, transportation, energy, and disaster planning. Traditional forecasting methods struggle with rapid changes in temperature, humidity, wind speed, and precipitation. This project develops a machine learning (ML)-based system that predicts weather parameters and visualizes forecasts on a dashboard.

---

## 🧠 Project Components

### 1. **ML-Based Weather Forecasting**
- Supervised ML models are trained using historical weather data to predict:
  - Temperature
  - Humidity
  - Wind speed
  - Precipitation
  - Rain (binary classification)

### 2. **Automated Data Ingestion**
- Current data is fetched using the OpenWeatherMap API.
- Historical data is sourced from a CSV file (`weather.csv`).
- The script automatically runs every 5 hours using `schedule`.

### 3. **Scheduler for Regular Updates**
- Script fetches current weather and generates predictions at scheduled intervals.
- Updated forecasts are saved in `weather_actual_vs_predicted.csv`.

### 4. **Dashboard Integration (optional)**
- Final CSV (`weather_actual_vs_predicted.csv`) can be visualized on a dashboard.
- Predictions include temperature, humidity, wind speed, and rain classification.

---

## 📊 Dataset Columns

The data includes the following columns:

| Column Name            | Description                                   |
|------------------------|-----------------------------------------------|
| `date`                | Timestamp of the reading                      |
| `city`                | City name                                     |
| `MinTemp`             | Minimum temperature (°C)                      |
| `MaxTemp`             | Maximum temperature (°C)                      |
| `WindGustDir`         | Wind direction (e.g., N, NW, etc.)            |
| `WindGustSpeed`       | Wind speed (km/h)                             |
| `Humidity`            | Relative humidity (%)                         |
| `Pressure`            | Atmospheric pressure (hPa)                    |
| `Temp`                | Current temperature (°C)                      |
| `Precipitation`       | Rainfall in mm                                |
| `RainTomorrow`        | Target label (Yes/No)                         |

---

## 🧪 Output

The prediction file (`weather_actual_vs_predicted.csv`) includes:
- Forecast for the next 7 days.
- Actual vs. predicted values for:
  - Temperature
  - Humidity
  - Wind speed
  - Precipitation
- Rain and weather classification per day.

---

## ⚙️ Usage

```bash
# Run the main script
python weather_forecasting.py
```

> Ensure the API key and internet access are working for real-time data.

---

## 🔁 Automation

- The script is scheduled to run every **5 hours** via `schedule`.
- Predicted outputs are saved as a CSV and can be pushed to a dashboard.

---

## 📂 File Structure

```
.
├── data/
│   ├── weather.csv                      # Historical weather data
│   └── weather_actual_vs_predicted.csv # Output predictions
├── weather_forecasting.py              # Main script
└── README.md                           # This file
```

---

## ✍️ Author

Developed as part of a machine learning weather forecasting project.

---

## 🔗 Submission Guidelines

Please upload your code and submission to the challenge platform as per instructions.
