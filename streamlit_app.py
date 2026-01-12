import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from datetime import date, timedelta

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="AQI Forecast Using LSTM", layout="wide")
st.title("🌫️ AQI Forecast Using LSTM")
st.write("City-based AQI prediction, classification, and evaluation")

# --------------------------------------------------
# AQI CATEGORY FUNCTION
# --------------------------------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Satisfactory 🟡"
    elif aqi <= 200:
        return "Moderate 🟠"
    elif aqi <= 300:
        return "Poor 🔴"
    elif aqi <= 400:
        return "Very Poor 🟣"
    else:
        return "Severe ⚫"

# --------------------------------------------------
# AQI VALUES & CLASSIFICATION TABLE
# --------------------------------------------------
aqi_values_table = pd.DataFrame({
    "AQI Range": ["0–50", "51–100", "101–200", "201–300", "301–400", "401–500"],
    "AQI Class": ["Good 🟢", "Satisfactory 🟡", "Moderate 🟠", "Poor 🔴", "Very Poor 🟣", "Severe ⚫"],
    "Health Impact": [
        "Minimal impact",
        "Minor discomfort to sensitive people",
        "Breathing discomfort to vulnerable groups",
        "Breathing discomfort to most people",
        "Respiratory illness on prolonged exposure",
        "Serious health effects"
    ]
})

# --------------------------------------------------
# CITY → LAT/LON
# --------------------------------------------------
def get_lat_lon(city):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    res = requests.get(url, params={"name": city, "count": 1}, timeout=20).json()
    if "results" not in res:
        return None, None
    return res["results"][0]["latitude"], res["results"][0]["longitude"]

# --------------------------------------------------
# FETCH PM2.5 DATA
# --------------------------------------------------
def fetch_pm25(lat, lon):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5",
        "past_days": 365
    }

    res = requests.get(url, params=params, timeout=20).json()

    df = pd.DataFrame({
        "time": pd.to_datetime(res["hourly"]["time"]),
        "pm25": res["hourly"]["pm2_5"]
    })

    df = df.set_index("time").resample("D").mean().reset_index()
    return df

# --------------------------------------------------
# PM2.5 → AQI CONVERSION
# --------------------------------------------------
def pm25_to_aqi(pm):
    if pm <= 12:
        return (50 / 12) * pm
    elif pm <= 35.4:
        return ((100 - 51) / (35.4 - 12)) * (pm - 12) + 51
    elif pm <= 55.4:
        return ((150 - 101) / (55.4 - 35.4)) * (pm - 35.4) + 101
    elif pm <= 150.4:
        return ((200 - 151) / (150.4 - 55.4)) * (pm - 55.4) + 151
    else:
        return ((300 - 201) / (250.4 - 150.4)) * (pm - 150.4) + 201

# --------------------------------------------------
# LSTM FORECAST + EVALUATION
# --------------------------------------------------
def lstm_forecast_with_evaluation(data, days=3):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))

    window = 14
    X, y = [], []

    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    # Train-Test Split (80-20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

    # Test Predictions
    y_pred = model.predict(X_test, verbose=0)

    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Evaluation Metrics
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    

    # Future Forecast
    last_seq = scaled[-window:]
    future = []

    for _ in range(days):
        pred = model.predict(last_seq.reshape(1, window, 1), verbose=0)
        future.append(pred[0][0])
        last_seq = np.vstack((last_seq[1:], pred))

    future = scaler.inverse_transform(np.array(future).reshape(-1, 1))

    return future.flatten(), y_test_inv.flatten(), y_pred_inv.flatten(), mae, rmse

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
city = st.text_input("Enter City Name", "Pune")

if st.button("Predict AQI"):
    with st.spinner("Analyzing AQI data..."):

        lat, lon = get_lat_lon(city)

        if lat is None:
            st.error("❌ City not found")
        else:
            df = fetch_pm25(lat, lon)
            df["AQI"] = df["pm25"].apply(pm25_to_aqi)
            df["Category"] = df["AQI"].apply(aqi_category)

            future_aqi, y_actual, y_pred, mae, rmse= lstm_forecast_with_evaluation(
                df["AQI"].values, 3
            )

            today = date.today()
            future_dates = [today + timedelta(days=i) for i in range(1, 4)]

            # --------------------------------------------------
            # FORECAST TABLE
            # --------------------------------------------------
            st.subheader("📅 AQI Forecast (Next 3 Days)")
            forecast_table = pd.DataFrame({
                "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
                "Predicted AQI": np.round(future_aqi, 1),
                "Category": [aqi_category(aqi) for aqi in future_aqi]
            })
            st.dataframe(forecast_table, use_container_width=True)

            # --------------------------------------------------
            # CHARTS
            # --------------------------------------------------
            left_col, right_col = st.columns(2)

            with left_col:
                st.subheader("📈 AQI Trend & Forecast")
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                ax1.plot(df["time"], df["AQI"], label="Historical AQI")
                ax1.plot(future_dates, future_aqi, "o--", label="Forecast")
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                fig1.autofmt_xdate(rotation=30)
                ax1.set_xlabel("Date")
                ax1.set_ylabel("AQI")
                ax1.legend()
                ax1.grid(True)
                st.pyplot(fig1)

            with right_col:
                st.subheader("📊 AQI Category Distribution")
                counts = df["Category"].value_counts()
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.bar(counts.index, counts.values)
                ax2.tick_params(axis="x", rotation=25)
                ax2.set_ylabel("Days")
                st.pyplot(fig2)
            
             # --------------------------------------------------
            # AQI CLASSIFICATION TABLE
            # --------------------------------------------------
            st.subheader("📘 AQI Values & Classification")
            st.table(aqi_values_table)


            # --------------------------------------------------
            # MODEL EVALUATION
            # --------------------------------------------------
            st.subheader("📊 Model Evaluation Results")
            c1, c2= st.columns(2)
            c1.metric("MAE", f"{mae:.2f}")
            c2.metric("RMSE", f"{rmse:.2f}")

            st.success("✅ AQI Forecast & Evaluation Completed Successfully")
            

            
           
