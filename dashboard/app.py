import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
from src.model import train_model, predict

# --- Page Configuration ---
st.set_page_config(
    page_title="ShuddhVayu - Gujarat Air Quality",
    page_icon="🌬️",
    layout="wide"
)

# --- File Paths ---
PROCESSED_DATA_PATH = os.path.join("data", "processed", "gujarat_aqi.csv")
MODEL_FEATURES_PATH = os.path.join("data", "processed", "gujarat_features_for_model.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "xgboost_pm25_model.joblib")

# --- Data Loading ---
@st.cache_data
def load_data(path):
    """Loads data from a given path."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['Date'])
    return df

# --- Main App Logic ---
st.title("🌬️ ShuddhVayu: Gujarat Air Quality Dashboard")

df_viz = load_data(PROCESSED_DATA_PATH)

if df_viz is None:
    st.error("Processed data not found. Please run the data pipeline first by executing `python src/data_pipeline.py`")
else:
    # --- Sidebar for User Inputs ---
    st.sidebar.header("Filters")
    
    cities = sorted(df_viz['City'].unique())
    selected_city = st.sidebar.selectbox("Select a City", cities, help="Choose a city to visualize its air quality trends.")
    
    pollutants = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    selected_pollutant = st.sidebar.selectbox("Select a Pollutant", pollutants, help="Select the pollutant to display on the chart.")

    # --- Main Panel: Historical Data Visualization ---
    st.header(f"Historical {selected_pollutant} Levels in {selected_city}")

    df_filtered = df_viz[df_viz['City'] == selected_city]

    fig = px.line(
        df_filtered,
        x='Date',
        y=selected_pollutant,
        title=f"Historical {selected_pollutant} Trend for {selected_city}",
        labels={'Date': 'Date', selected_pollutant: f'{selected_pollutant} Concentration'}
    )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(rangeselector=dict(buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])))
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Forecasting Section ---
    # --- FIX: Corrected city name spelling back to Ahmedabad ---
    st.header(f"🔮 PM2.5 Forecast for Ahmedabad")

    if selected_city != 'Ahmedabad':
        st.warning("Forecasting is currently only available for Ahmedabad.")
    else:
        df_model_features = load_data(MODEL_FEATURES_PATH)
        
        if df_model_features is None:
            st.error("Model features data not found. Please ensure the data pipeline has been run.")
        else:
            if st.button("Train/Retrain Forecasting Model"):
                with st.spinner("Training a new model... This may take a moment."):
                    train_model(df_model_features, save_path=MODEL_SAVE_PATH)
                st.success(f"Model trained and saved successfully to {MODEL_SAVE_PATH}")

            if not os.path.exists(MODEL_SAVE_PATH):
                st.info("Forecasting model not found. Please train the model using the button above.")
            else:
                model = joblib.load(MODEL_SAVE_PATH)
                
                last_data_point = df_model_features.tail(1)
                
                last_date = pd.to_datetime(last_data_point['Date'].iloc[0])
                future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, 8)])
                
                predictions = predict(model, last_data_point, future_dates)
                
                df_forecast = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted PM2.5': predictions
                })

                st.subheader("Next 7-Day PM2.5 Forecast")
                forecast_fig = px.line(
                    df_forecast,
                    x='Date',
                    y='Predicted PM2.5',
                    title="Forecasted PM2.5 Levels",
                    labels={'Predicted PM2.5': 'PM2.5 Concentration'},
                    markers=True
                )
                forecast_fig.update_layout(template="plotly_white")
                st.plotly_chart(forecast_fig, use_container_width=True)