import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
RAW_DATA_PATH = os.path.join("data", "raw", "city_day.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "gujarat_aqi.csv")
MODEL_FEATURES_PATH = os.path.join("data", "processed", "gujarat_features_for_model.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "xgboost_pm25_model.joblib")

# --- Helper Function for Alerts ---
def get_aqi_alert(pm25_value):
    # ... (This function remains the same)
    if pm25_value <= 30:
        return "Good", "Air quality is excellent. Enjoy outdoor activities!", "success"
    elif 31 <= pm25_value <= 60:
        return "Satisfactory", "Air quality is acceptable. Sensitive individuals may experience minor respiratory symptoms.", "info"
    elif 61 <= pm25_value <= 90:
        return "Moderate", "Sensitive groups (children, elderly, asthmatics) should reduce prolonged or heavy outdoor exertion.", "warning"
    elif 91 <= pm25_value <= 120:
        return "Poor", "Everyone should reduce heavy outdoor exertion. People with heart or lung disease should avoid it entirely.", "error"
    elif 121 <= pm25_value <= 250:
        return "Very Poor", "Avoid all outdoor physical activity. Sensitive groups should remain indoors.", "error"
    else:
        return "Severe", "Remain indoors and keep activity levels low. A health advisory is in effect.", "error"

# --- Data Loading ---
@st.cache_data
def load_data(path, date_col='Date'):
    """Loads data from a given path."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=[date_col])
    return df

# --- Main App Logic ---
st.title("🌬️ ShuddhVayu: Gujarat Air Quality Dashboard")

df_processed = load_data(PROCESSED_DATA_PATH)

if df_processed is None:
    st.error("Processed data not found. Please run the data pipeline first by executing `python src/data_pipeline.py`")
else:
    # --- Sidebar for User Inputs ---
    st.sidebar.header("Filters")
    
    cities = sorted(df_processed['City'].unique())
    selected_city = st.sidebar.selectbox("Select a City", cities)
    
    pollutants = ['AQI', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    
    # --- Main Panel with Tabs ---
    st.header(f"Exploratory Data Analysis for {selected_city}")
    
    tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Pollutant Distribution", "Data Quality"])

    # Filter data once for all tabs
    df_city_processed = df_processed[df_processed['City'] == selected_city]

    # --- Tab 1: Trend Analysis ---
    with tab1:
        st.subheader("Compare Historical Pollutant Trends")
        
        selected_pollutants = st.multiselect(
            "Select pollutants to compare",
            options=pollutants,
            default=['PM2.5', 'AQI']
        )
        
        if not selected_pollutants:
            st.warning("Please select at least one pollutant to display the trend.")
        else:
            trend_fig = px.line(
                df_city_processed,
                x='Date',
                y=selected_pollutants,
                title=f"Historical Trends for {', '.join(selected_pollutants)} in {selected_city}"
            )
            trend_fig.update_layout(template="plotly_white", yaxis_title="Concentration / Index Value")
            st.plotly_chart(trend_fig, use_container_width=True)

    # --- Tab 2: Pollutant Distribution ---
    with tab2:
        st.subheader("Visualize Pollutant Variation")
        
        dist_pollutant = st.selectbox(
            "Select a pollutant to see its distribution",
            options=pollutants,
            index=1 # Default to PM2.5
        )
        
        df_city_processed['Year'] = df_city_processed['Date'].dt.year
        box_fig = px.box(
            df_city_processed,
            x='Year',
            y=dist_pollutant,
            title=f"Yearly Distribution of {dist_pollutant} in {selected_city}"
        )
        box_fig.update_layout(template="plotly_white")
        st.plotly_chart(box_fig, use_container_width=True)
        st.markdown("This box plot shows the median (middle line), the interquartile range (the box), and outliers (dots). It's useful for seeing the spread and skewness of the data each year.")

    # --- Tab 3: Data Quality ---
    with tab3:
        st.subheader("Original Missing Value Analysis")
        
        # Load raw data to show original missing values
        df_raw = load_data(RAW_DATA_PATH)
        if df_raw is not None:
            df_city_raw = df_raw[df_raw['City'] == selected_city]
            missing_values = df_city_raw[pollutants].isnull().sum().reset_index()
            missing_values.columns = ['Pollutant', 'Number of Missing Days']
            
            missing_fig = px.bar(
                missing_values,
                x='Pollutant',
                y='Number of Missing Days',
                title=f"Total Missing Data Points (Days) in Raw Data for {selected_city}"
            )
            st.plotly_chart(missing_fig, use_container_width=True)
            st.markdown("This chart shows the number of days for which data was not recorded for each pollutant in the original dataset. Our data pipeline fills these gaps using time-based interpolation.")
        else:
            st.error("Raw data file not found. Cannot perform missing value analysis.")

    # --- Forecasting Section (remains the same) ---
    st.header(f"🔮 PM2.5 Forecast for Ahmedabad")

    if selected_city != 'Ahmedabad':
        st.warning("Forecasting is currently only available for Ahmedabad.")
    else:
        # ... (The entire forecasting section code is unchanged)
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
                loaded_object = joblib.load(MODEL_SAVE_PATH)
                
                mae = None
                if isinstance(loaded_object, dict):
                    model = loaded_object['model']
                    mae = loaded_object['mae']
                else:
                    model = loaded_object
                
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

                st.subheader("Next 7-Day PM2.5 Forecast with Confidence Interval")

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
                    st.warning("Retrain the model to see the confidence interval.")
                    forecast_fig = px.line(df_forecast, x='Date', y='Predicted PM2.5', title="Forecasted PM2.5 Levels", markers=True)

                st.plotly_chart(forecast_fig, use_container_width=True)