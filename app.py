
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title("📈 Gold Price Forecast with ARIMA")

uploaded_file = st.file_uploader("Upload your Gold CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    st.subheader("📊 Gold Price Data")
    st.write(df.tail())

    # Plot
    st.subheader("📉 Historical Gold Prices")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['price'], label="Gold Price", color='gold')
    ax.set_title("Gold Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # ARIMA model selection
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    best_aic = np.inf
    best_order = None

    train = df['price'][:int(len(df) * 0.8)]
    for order in pdq:
        try:
            model = sm.tsa.ARIMA(train, order=order).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = order
        except:
            continue

    st.write(f"✅ Best ARIMA Order: {best_order} (AIC: {best_aic:.2f})")
    model = sm.tsa.ARIMA(df['price'], order=best_order).fit()

    # Date selection
    st.subheader("📅 Select Start Date for 30-Day Forecast")
    selected_date = st.date_input("Select a date", df.index.max().date())

    if selected_date <= df.index.max().date():
        st.warning("⚠️ Please select a date after the latest date in the dataset.")
    else:
        future_days = 30
        forecast = model.forecast(steps=future_days)
        start_date = pd.to_datetime(selected_date)
        forecast_index = pd.date_range(start=start_date, periods=future_days)
        forecast_series = pd.Series(forecast, index=forecast_index)

        st.subheader("🔮 Forecast for Next 30 Days")
        fig, ax = plt.subplots()
        df['price'].plot(label="Historical", ax=ax)
        forecast_series.plot(label="Forecast", ax=ax, color='red')
        ax.set_title("Next 30-Day Gold Price Forecast")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(forecast_series.rename("Forecast Price"))
else:
    st.info("📤 Please upload a CSV file with 'date' and 'price' columns.")
