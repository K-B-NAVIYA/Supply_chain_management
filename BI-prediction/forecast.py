import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Advanced Forecasting", layout="wide")

# -------------------------------
# 🎨 LIGHT BLUE UI STYLE
# -------------------------------
st.markdown("""
<style>

/* Main background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #eaf4ff, #f7fbff);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f0f7ff;
}

/* Title */
.big-title {
    font-size: 36px;
    font-weight: bold;
    color: #1e3a8a;
}

.subtitle {
    font-size: 18px;
    color: #475569;
}

/* Cards */
.card {
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
}

.blue { background: linear-gradient(135deg, #3b82f6, #2563eb); }
.green { background: linear-gradient(135deg, #10b981, #059669); }
.orange { background: linear-gradient(135deg, #f59e0b, #d97706); }

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='big-title'>📊 Intelligent Supply Chain Forecasting System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-driven Demand Prediction & Decision Support</div>", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("sales_orders.csv")
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df = df.dropna()

    product_map = {
        "PROD00135": "Smart Watch",
        "PROD00147": "Laptop",
        "PROD00172": "Headphones",
        "PROD00063": "Mobile Phone",
        "PROD00065": "Tablet"
    }

    df['Product_Name'] = df['Product_ID'].map(product_map)
    df = df[df['Product_Name'].notna()]

    return df

df = load_data()

# -------------------------------
# 🎛️ SIDEBAR
# -------------------------------
st.sidebar.markdown("## 🎛️ Control Panel")

st.sidebar.markdown("### 📦 Product Selection")
product = st.sidebar.selectbox(
    "Choose Product",
    ["All"] + list(df['Product_Name'].unique())
)

st.sidebar.markdown("### 📅 Forecast Settings")
forecast_horizon = st.sidebar.slider("Forecast Months", 6, 36, 24)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.write(f"✔ Total Records: {len(df)}")
st.sidebar.write(f"✔ Products: {df['Product_Name'].nunique()}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Current Selection")
st.sidebar.write(f"📦 Product: **{product}**")
st.sidebar.write(f"📅 Horizon: **{forecast_horizon} months**")

st.sidebar.markdown("---")
st.sidebar.info("Use controls to adjust forecasting and filter products.")

# -------------------------------
# FILTER DATA
# -------------------------------
filtered_df = df[df['Product_Name'] == product] if product != "All" else df.copy()

# -------------------------------
# TIME SERIES
# -------------------------------
data = filtered_df.groupby(
    pd.Grouper(key='Order_Date', freq='ME')
)['Order_Total'].sum().dropna()

# -------------------------------
# 📊 CARDS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='card blue'><h4>Total Sales</h4><h2>{int(data.sum())}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card green'><h4>Average Demand</h4><h2>{int(data.mean())}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card orange'><h4>Peak Demand</h4><h2>{int(data.max())}</h2></div>", unsafe_allow_html=True)

# -------------------------------
# INFO
# -------------------------------
st.subheader("🧠 Forecast Intelligence")
st.info("Future demand prediction using time-series and machine learning models.")

# -------------------------------
# ARIMA
# -------------------------------
if len(data) > 12:

    train = data[:-3]
    test = data[-3:]

    model = ARIMA(train, order=(5,1,0))
    fit = model.fit()

    pred = fit.forecast(3)
    mape = mean_absolute_percentage_error(test, pred)

    final_model = ARIMA(data, order=(5,1,0)).fit()
    forecast = final_model.get_forecast(steps=forecast_horizon)

    forecast_values = forecast.predicted_mean
    conf = forecast.conf_int()

    st.subheader("📈 Demand Forecast")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(data, label="Actual")
    ax.plot(forecast_values, label="Forecast", linestyle="--")

    ax.fill_between(
        forecast_values.index,
        conf.iloc[:,0],
        conf.iloc[:,1],
        alpha=0.2
    )

    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Model Performance")
    st.write(f"MAPE Error: {round(mape*100,2)}%")

else:
    st.error("Not enough data")

# -------------------------------
# ML COMPARISON
# -------------------------------
st.subheader("🤖 ML Comparison")

df_ml = data.reset_index()
df_ml['t'] = np.arange(len(df_ml))

lr = LinearRegression()
lr.fit(df_ml[['t']], df_ml['Order_Total'])

future_t = np.arange(len(df_ml), len(df_ml)+forecast_horizon).reshape(-1,1)
ml_forecast = lr.predict(future_t)

fig2, ax2 = plt.subplots(figsize=(12,5))
ax2.plot(data.values, label="Actual")
ax2.plot(range(len(data), len(data)+forecast_horizon), ml_forecast, label="ML Forecast")

ax2.legend()
st.pyplot(fig2)

# -------------------------------
# 🚨 ANOMALY
# -------------------------------
st.subheader("🚨 Anomaly Detection")

iso = IsolationForest(contamination=0.05)
anomaly = iso.fit_predict(data.values.reshape(-1,1))

fig3, ax3 = plt.subplots(figsize=(12,5))
ax3.plot(data.index, data.values)
ax3.scatter(data.index[anomaly==-1], data.values[anomaly==-1], color='red')

st.pyplot(fig3)

# -------------------------------
# 📦 CLASSIFICATION
# -------------------------------
st.subheader("📦 Demand Classification")

forecast_df = forecast_values.reset_index()
forecast_df.columns = ['Date','Sales']

forecast_df['Category'] = np.where(
    forecast_df['Sales'] > forecast_df['Sales'].mean()*1.2,
    "High Demand",
    np.where(
        forecast_df['Sales'] < forecast_df['Sales'].mean()*0.8,
        "Low Demand",
        "Normal"
    )
)

st.dataframe(forecast_df)

# -------------------------------
# 🎯 WHAT-IF
# -------------------------------
st.subheader("🎯 Scenario Simulation")

growth = st.slider("Increase Demand %", -20, 50, 10)

adjusted = forecast_df.copy()
adjusted['Adjusted_Sales'] = adjusted['Sales'] * (1 + growth/100)

st.line_chart(adjusted.set_index('Date')[['Sales','Adjusted_Sales']])

# -------------------------------
# SUMMARY
# -------------------------------
st.subheader("📋 Executive Summary")

trend = "increasing" if forecast_df['Sales'].iloc[-1] > forecast_df['Sales'].iloc[0] else "decreasing"

st.write(f"""
✔ Product: {product}  
✔ Trend: {trend}  
✔ Horizon: {forecast_horizon} months  
✔ Peak Demand: {forecast_df.loc[forecast_df['Sales'].idxmax(), 'Date']}  
✔ Accuracy: {round(mape*100,2)}%  
""")

# -------------------------------
# DOWNLOAD
# -------------------------------
st.subheader("📥 Download Forecast")

st.download_button(
    "Download CSV",
    forecast_df.to_csv(index=False).encode('utf-8'),
    "forecast.csv",
    "text/csv"
)