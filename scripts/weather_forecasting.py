import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #to split data into tr
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta 
import pytz
import schedule
import time


API_KEY = "54227622caf10f3698a6c14399c42dc6"
BASE_URL = "https://api.openweathermap.org/data/2.5/"

def get_current_weather(city):
    '''Fetch current weather data for a given city using OpenWeatherMap API.
    Returns a dictionary with relevant weather information.'''
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching weather data: {response.status_code}")
    data = response.json()
    return {
        'city': data['name'],
        'current_temp':round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min':round(data['main']['temp_min']),
        'temp_max':round(data['main']['temp_max']),
        'humidity':round(data['main']['humidity']),
        'description':data['weather'][0]['description'],
        'country':data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed'],
        'cloud_coverage': data['clouds']['all'],
        'precipitation': data.get('rain', {}).get('1h', 0.0)
    }

def read_historical_data(filename):
    '''Read historical weather data from a CSV file.
    Cleans the data by dropping NaN values and duplicates.'''
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def prepare_data(data):
    '''Prepare the data for training the model.
    Encodes categorical variables and splits the data into features and target variable.'''
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    X = data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp','Precipitation']]
    y = data['RainTomorrow']

    return X ,y , le

def train_rain_model(X,y):
    '''Train a Random Forest Classifier to predict rain.
    Splits the data into training and testing sets, fits the model, and evaluates it.'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Mean Squared Error for Rain model")
    print(mean_squared_error(y_test, y_pred))

    return model

def prepare_regression_data(data, feature):
    '''Prepare the data for regression model.
    Splits the data into features and target variable for the specified feature.'''
    X,y = [], []

    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])



    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    return X, y

def train_regression_model(X,y):
    '''Train a Random Forest Regressor to predict the specified feature.
    Splits the data into training and testing sets, fits the model, and evaluates it.'''
    model =  RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X,y)
    return model

def predict_future(model, current_value):
    '''Predict future values using the trained regression model.
    Takes the current value and predicts the next 7 days.'''
    predictions = [current_value]

    for i in range(7):
        next_value = model.predict(np.array([[predictions[-1]]]))

        predictions.append(next_value[0])
    
    return predictions[1:]

def weather_view():
    '''Fetch current weather data for a specific city, train models on historical data,
    and predict future temperature and humidity.'''
    city = 'latur'
    current_weather = get_current_weather(city)

    # historical_data = read_historical_data('weather forcasting\\data\\weather.csv')
    historical_data = read_historical_data('./data/weather.csv')

    X, y, le = prepare_data(historical_data)

    rain_model = train_rain_model(X, y)

    wind_deg = current_weather['wind_gust_dir'] % 360
    compass_points = [
        ("N",0,11.25),("NNE",11.25,33.75),("NE",33.75,56.25),("ENE",56.25,78.75),("E",78.75,101.25),
        ("ESE",101.25,123.75),("SE",123.75,146.25),("SSE",146.25,168.75),("S",168.25,191.25),("SSW",191.25,213.75),
        ("SW",213.75,236.25),("WSW",236.25,258.75),("W",258.75,281.25),("WNW",281.25,303.75),("NW",303.75,326.25),
        ("NNW",326.25,348.75)
    ]

    compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)

    compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

    current_data = {
        'MinTemp':current_weather['temp_min'],
        'MaxTemp':current_weather['temp_max'],
        'WindGustDir':compass_direction_encoded,
        'WindGustSpeed':current_weather['Wind_Gust_Speed'],
        'Humidity': current_weather['humidity'],
        'Pressure': current_weather['pressure'],
        'Temp': current_weather['current_temp'],
        'Precipitation': current_weather['precipitation']
    }

    current_df = pd.DataFrame([current_data])

    rain_prediction = rain_model.predict(current_df)[0]

    X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')

    X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

    X_windspeed, y_windspeed = prepare_regression_data(historical_data, 'WindGustSpeed')

    X_pricip, y_pricip = prepare_regression_data(historical_data, 'Precipitation')


    temp_model = train_regression_model(X_temp, y_temp)
    hum_model = train_regression_model(X_hum, y_hum)
    windspeed_model = train_regression_model(X_windspeed, y_windspeed)
    precip_model = train_regression_model(X_pricip, y_pricip)


    future_temp = predict_future(temp_model, current_weather['temp_min'])
    future_humidity = predict_future(hum_model, current_weather['humidity'])
    future_windspeed = predict_future(windspeed_model, current_weather['Wind_Gust_Speed'])
    future_precip = predict_future(precip_model, current_weather['precipitation'])

    timezone = pytz.timezone('Asia/Kolkata')
    
    now = datetime.now(timezone)

    base_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(days=1)
    future_days = [(base_time + timedelta(days=i)).strftime("%Y-%m-%d %H:%M") for i in range(7)]
    


    print(f" City: {city}, {current_weather['country']}")
    print(f"Current Temprature: {current_weather['current_temp']}")
    print(f"Feels Like: {current_weather['feels_like']}")
    print(f"Minimum Temprature: {current_weather['temp_min']}")
    print(f"Maximum Temprature: {current_weather['temp_max']}")
    print(f"Wind Speed: {current_weather['Wind_Gust_Speed']} km/h")
    print(f"Humidity: {current_weather['humidity']}%")
    print(f"Weather Prediction: {current_weather['description']}")
    print(f"Rain Prediction: {'Yes' if rain_prediction else 'No'}")

    print("\nFuture Temprature Predictions:")

    for day, temp in zip(future_days,future_temp):
        print(f"{day}: {round(temp, 1)}deg C")

    print("\nFuture Humidity Predictions:")
    
    for day, humidity in zip(future_days,future_humidity):
        print(f"{day}: {round(humidity, 1)}%")


    print("\nFuture Wind Speed Predictions:")
    
    for day, windspeed in zip(future_days,future_windspeed):
        print(f"{day}: {round(windspeed, 1)} km/h")


    print("\nFuture precipitation Predictions:")
    
    for day, pricip in zip(future_days,future_precip):
        print(f"{day}: {round(pricip, 1)} mm")




    
    forecast_df = pd.DataFrame({
        'date_time': future_days,
        'actual_temperature': [current_weather['current_temp']] * 7,
        'predicted_temperature': [round(t, 1) for t in future_temp],
        'actual_humidity': [current_weather['humidity']] * 7,
        'predicted_humidity': [round(h, 1) for h in future_humidity],
        'actual_wind_speed': [current_weather['Wind_Gust_Speed']] * 7,
        'predicted_wind_speed': [round(w, 1) for w in future_windspeed],
        'actual_wind_direction': [current_weather['wind_gust_dir']] * 7,
        'cloud_coverage': [current_weather['cloud_coverage']] * 7,
        'precipitation (mm)': [current_weather['precipitation']] * 7,
        'predicted_precipitation': [round(p, 1) for p in future_precip],
        'rain_prediction': ['Yes' if rain_prediction else 'No'] * 7,
        'weather_prediction': [current_weather['description']] * 7
    })

    forecast_df.to_csv("./data/weather_actual_vs_predicted.csv", index=False)
    print("\n✅ Forecast saved to 'weather_actual_vs_predicted.csv'")



weather_view()




# Run the function every 5 hours
schedule.every(5).hours.do(weather_view)
print(f"✅ Scheduler started for city_name...\n")

while True:
    schedule.run_pending()
    time.sleep(60)
