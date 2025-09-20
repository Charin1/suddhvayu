import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import joblib

from src.model import train_model, predict
from src.eda import plot_trends, plot_distribution, plot_missing_values

# --- Page Configuration ---
st.set_page_config(page_title="ShuddhVayu - Air Quality Dashboard", page_icon="🌬️", layout="wide")

# --- File Paths ---
RAW_DATA_PATH = os.path.join("data", "raw", "city_day.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "gujarat_aqi.csv")
MODEL_FEATURES_PATH = os.path.join("data", "processed", "gujarat_features_for_model.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "xgboost_pm25_model.joblib")

# --- Helper Function for Alerts ---
def get_aqi_alert(pm25_value):
    if pm25_value <= 30: return "Good", "Air quality is excellent. Enjoy outdoor activities!", "success"
    if 31 <= pm25_value <= 60: return "Satisfactory", "Air quality is acceptable. Sensitive individuals may experience minor respiratory symptoms.", "info"
    if 61 <= pm25_value <= 90: return "Moderate", "Sensitive groups should reduce prolonged or heavy outdoor exertion.", "warning"
    if 91 <= pm25_value <= 120: return "Poor", "Everyone should reduce heavy outdoor exertion. People with heart or lung disease should avoid it.", "error"
    if 121 <= pm25_value <= 250: return "Very Poor", "Avoid all outdoor physical activity. Sensitive groups should remain indoors.", "error"
    return "Severe", "Remain indoors and keep activity levels low. A health advisory is in effect.", "error"

# --- Data Loading ---
@st.cache_data
def load_data(path, date_col='Date'):
    if not os.path.exists(path): return None
    return pd.read_csv(path, parse_dates=[date_col])

# --- Main App Logic ---
st.title("🌬️ ShuddhVayu: Gujarat Air Quality Dashboard")

tab_eda, tab_forecast = st.tabs(["📊 Exploratory Data Analysis", "🔮 Ahmedabad Forecast"])

with tab_eda:
    # This entire tab's code remains unchanged
    st.header("Explore Historical Air Quality Data")
    df_processed = load_data(PROCESSED_DATA_PATH)
    if df_processed is None:
        st.error("Processed data not found. Please run the data pipeline first by executing `python src/data_pipeline.py`")
    else:
        st.sidebar.header("EDA Controls")
        cities = sorted(df_processed['City'].unique())
        selected_city = st.sidebar.selectbox("Select a City for Analysis", cities)
        st.subheader(f"Analysis for {selected_city}")
        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Trend Analysis", "Pollutant Distribution", "Data Quality"])
        df_city_processed = df_processed[df_processed['City'] == selected_city]
        with eda_tab1:
            pollutants = ['AQI', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
            selected_pollutants = st.multiselect("Select pollutants to compare", options=pollutants, default=['PM2.5', 'AQI'])
            if not selected_pollutants: st.warning("Please select at least one pollutant.")
            else: st.plotly_chart(plot_trends(df_city_processed, selected_pollutants, selected_city), use_container_width=True)
        with eda_tab2:
            dist_pollutant = st.selectbox("Select a pollutant for distribution analysis", options=pollutants, index=1)
            st.plotly_chart(plot_distribution(df_city_processed, dist_pollutant, selected_city), use_container_width=True)
        with eda_tab3:
            df_raw = load_data(RAW_DATA_PATH)
            if df_raw is not None: st.plotly_chart(plot_missing_values(df_raw, pollutants, selected_city), use_container_width=True)
            else: st.error("Raw data file not found.")

with tab_forecast:
    st.header("🔮 PM2.5 Forecast for Ahmedabad")
    df_model_features = load_data(MODEL_FEATURES_PATH)
    
    if df_model_features is None:
        st.error("Model features data not found. Please ensure the data pipeline has been run.")
    else:
        if st.button("Train/Retrain Forecasting Model"):
            with st.spinner("Training a new model... This may take a moment."):
                train_model(df_model_features, save_path=MODEL_SAVE_PATH)
            st.success(f"Model trained and saved successfully!")
            st.rerun()

        if not os.path.exists(MODEL_SAVE_PATH):
            st.info("Forecasting model not found. Please train the model using the button above.")
        else:
            loaded_object = joblib.load(MODEL_SAVE_PATH)
            
            model, mae, start_date, end_date, train_time = (None,) * 5
            if isinstance(loaded_object, dict):
                model = loaded_object.get('model')
                mae = loaded_object.get('mae')
                start_date = loaded_object.get('train_start_date', 'N/A')
                end_date = loaded_object.get('train_end_date', 'N/A')
                train_time = loaded_object.get('train_timestamp', 'N/A')
            else:
                model = loaded_object
            
            st.subheader("Model Information")
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                col1.markdown("##### Algorithm"); col1.write("XGBoost Regressor")
                col2.markdown("##### Training Data Range"); col2.write(f"{start_date} to {end_date}")
                col3.markdown("##### Last Trained"); col3.write(train_time)

            # --- NEW: Recent Model Performance Section ---
            st.subheader("Recent Model Performance (Last 14 Days)")
            
            # Get the last 14 days of data with all necessary features
            df_recent_history = df_model_features.tail(14).copy()
            
            # Define the features the model expects
            FEATURES = [col for col in df_recent_history.columns if col not in ['PM2.5', 'Date', 'City', 'AQI_Bucket']]
            
            # Generate predictions for these known past days
            recent_predictions = model.predict(df_recent_history[FEATURES])
            
            # Create the performance DataFrame
            df_performance = pd.DataFrame({
                'Date': pd.to_datetime(df_recent_history['Date']).dt.strftime('%Y-%m-%d'),
                'Actual PM2.5': df_recent_history['PM2.5'],
                'Predicted PM2.5': recent_predictions
            })
            
            # Calculate the error column
            df_performance['Error'] = df_performance['Actual PM2.5'] - df_performance['Predicted PM2.5']
            
            # Style and display the table
            styler = df_performance.style.format({
                'Actual PM2.5': '{:.2f}',
                'Predicted PM2.5': '{:.2f}',
                'Error': '{:+.2f}'  # Show sign for positive/negative errors
            }).hide(axis="index")
            st.dataframe(styler, use_container_width=True)

            # --- Health Advisory and Forecast Plot ---
            last_data_point = df_model_features.tail(1)
            last_date = pd.to_datetime(last_data_point['Date'].iloc[0])
            future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, 8)])
            predictions = predict(model, last_data_point, future_dates)
            df_forecast = pd.DataFrame({'Date': future_dates, 'Predicted PM2.5': predictions})

            st.subheader("Health Advisory for Tomorrow")
            next_day_forecast = df_forecast['Predicted PM2.5'].iloc[0]
            category, recommendation, alert_type = get_aqi_alert(next_day_forecast)
            with st.container(border=True):
                st.metric(label=f"Tomorrow's Forecasted Category: {category}", value=f"{next_day_forecast:.2f} PM2.5")
                if alert_type == "success": st.success(f"**Recommendation:** {recommendation}")
                elif alert_type == "info": st.info(f"**Recommendation:** {recommendation}")
                elif alert_type == "warning": st.warning(f"**Recommendation:** {recommendation}")
                elif alert_type == "error": st.error(f"**Recommendation:** {recommendation}")

            st.subheader("7-Day Forecast Chart")
            if mae:
                df_forecast['Upper Bound'] = df_forecast['Predicted PM2.5'] + mae
                df_forecast['Lower Bound'] = df_forecast['Predicted PM2.5'] - mae
                forecast_fig = go.Figure([
                    go.Scatter(name='Upper Bound', x=df_forecast['Date'], y=df_forecast['Upper Bound'], mode='lines', line=dict(width=0.5, color='rgba(173, 216, 230, 0.5)')),
                    go.Scatter(name='Lower Bound', x=df_forecast['Date'], y=df_forecast['Lower Bound'], mode='lines', line=dict(width=0.5, color='rgba(173, 216, 230, 0.5)'), fillcolor='rgba(173, 216, 230, 0.2)', fill='tonexty'),
                    go.Scatter(name='Prediction', x=df_forecast['Date'], y=df_forecast['Predicted PM2.5'], mode='lines+markers', line=dict(color='rgb(0, 100, 80)'))
                ])
                forecast_fig.update_layout(title="Forecasted PM2.5 Levels with Confidence Interval", yaxis_title='PM2.5 Concentration', hovermode="x")
            else:
                st.warning("Retrain the model to see the confidence interval and full metadata.")
                forecast_fig = px.line(df_forecast, x='Date', y='Predicted PM2.5', title="Forecasted PM2.5 Levels", markers=True)
            st.plotly_chart(forecast_fig, use_container_width=True)