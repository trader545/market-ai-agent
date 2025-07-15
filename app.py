import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from difflib import get_close_matches

st.set_page_config(page_title="AI Market Price Predictor", layout="wide")
st.title("📈 AI Market Price Predictor")

# ✅ Real-time symbol mapping with fuzzy search
name_to_symbol = {
    "bitcoin": "BTC-USD",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "google": "GOOGL",
    "amazon": "AMZN",
    "ethereum": "ETH-USD",
    "nifty": "^NSEI",
    "banknifty": "^NSEBANK",
    "usd/inr": "USDINR=X",
    "eur/usd": "EURUSD=X",
    "tcs": "TCS.NS",
    "reliance": "RELIANCE.NS",
    "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS"
}

# 📥 Search input
query = st.text_input("🔍 Enter the name or symbol of a Stock, Indian Stock, Crypto, or Forex", value="bitcoin").strip().lower()

# ✅ Fuzzy match if not exact
symbol = name_to_symbol.get(query)
if not symbol:
    matches = get_close_matches(query, name_to_symbol.keys(), n=1, cutoff=0.6)
    if matches:
        symbol = name_to_symbol[matches[0]]
        st.success(f"✅ Found: *{matches[0].title()}* → {symbol}")
    else:
        symbol = query.upper()
        st.warning(f"⚠ Using as symbol: {symbol}")

# 📉 Download data
try:
    data = yf.download(symbol, period="7d", interval="15m", progress=False)
    if data.empty:
        st.error("❌ Could not find the symbol. Try typing a different name like 'Apple', 'TCS', 'Bitcoin'.")
    else:
        df = data[['Close']].dropna().copy()

        # ✅ Forecasting function
        def make_forecast(df, steps=1):
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df)
            X, y = [], []
            for i in range(60, len(scaled_data) - steps):
                X.append(scaled_data[i-60:i])
                y.append(scaled_data[i+steps-1])
            if not X:
                return np.nan
            X, y = np.array(X), np.array(y)
            model = LinearRegression()
            model.fit(X.reshape(X.shape[0], -1), y)
            pred_input = scaled_data[-60:].reshape(1, -1)
            forecast = model.predict(pred_input)
            return scaler.inverse_transform(forecast.reshape(-1, 1))[0][0]

        # 🔮 Predict next 15 mins & next day
        next_15 = make_forecast(df, 1)
        next_day = make_forecast(df, 96)

        # 💰 Display current + forecasted prices
        st.metric("💰 Current Price", f"${float(df['Close'].iloc[-1]):,.2f}")
        st.metric("⏱ Forecast (15 mins)", f"${next_15:,.2f}" if not np.isnan(next_15) else "N/A")
        st.metric("📅 Forecast (Tomorrow)", f"${next_day:,.2f}" if not np.isnan(next_day) else "N/A")

        st.markdown("---")
        st.markdown("### 📊 Interactive Charts")

        df['Time'] = df.index
        df['Forecast_15m'] = np.nan
        df.loc[df.index[-1], 'Forecast_15m'] = next_15

        # 📈 Line Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Time'], y=df['Close'], name='Price', mode='lines+markers'))
        fig.add_trace(go.Scatter(
            x=[df['Time'].iloc[-1] + timedelta(minutes=15)],
            y=[next_15],
            mode='markers+text',
            name='15-min Forecast',
            text=["Forecast"],
            textposition="top center"
        ))
        fig.update_layout(title=f"{symbol} Price Forecast", xaxis_title="Time", yaxis_title="Price")

        # 🕯 Candlestick Chart
        fig2 = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        fig2.update_layout(title="Candlestick Chart", xaxis_title="Time", yaxis_title="Price")

        # 📊 Bar Chart
        fig3 = go.Figure(data=[go.Bar(x=df['Time'], y=df['Close'], name='Bar')])
        fig3.update_layout(title="Bar Chart", xaxis_title="Time", yaxis_title="Price")

        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)

except Exception as e:
    st.error(f"❌ Error: {e}")
