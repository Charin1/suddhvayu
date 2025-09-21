import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import joblib

from src.models import train_and_save_model, dynamic_predict, MODELS
from src.eda import plot_trends, plot_distribution, plot_missing_values

st.set_page_config(page_title="ShuddhVayu - Air Quality Dashboard", page_icon="🌬️", layout="wide")
RAW_DATA_PATH = os.path.join("data", "raw", "city_day.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "gujarat_aqi.csv")
MODEL_FEATURES_PATH = os.path.join("data", "processed", "gujarat_features_for_model.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def get_aqi_alert(pm25_value):
    if pm25_value <= 30: return "Good", "Air quality is excellent. Enjoy outdoor activities!", "success"
    if 31 <= pm25_value <= 60: return "Satisfactory", "Air quality is acceptable. Sensitive individuals may experience minor respiratory symptoms.", "info"
    if 61 <= pm25_value <= 90: return "Moderate", "Sensitive groups should reduce prolonged or heavy outdoor exertion.", "warning"
    if 91 <= pm25_value <= 120: return "Poor", "Everyone should reduce heavy outdoor exertion. People with heart or lung disease should avoid it.", "error"
    if 121 <= pm25_value <= 250: return "Very Poor", "Avoid all outdoor physical activity. Sensitive groups should remain indoors.", "error"
    return "Severe", "Remain indoors and keep activity levels low. A health advisory is in effect.", "error"

@st.cache_data
def load_data(path, date_col='Date'):
    if not os.path.exists(path): return None
    return pd.read_csv(path, parse_dates=[date_col])

st.title("🌬️ ShuddhVayu: Gujarat Air Quality Dashboard")
tab_eda, tab_forecast = st.tabs(["📊 Exploratory Data Analysis", "🔮 Ahmedabad Forecast"])

with tab_eda:
    st.header("Explore Historical Air Quality Data")
    df_processed = load_data(PROCESSED_DATA_PATH)
    if df_processed is None: st.error("Processed data not found.")
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
    st.header("🔮 Forecast for Ahmedabad")
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_targets = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']
        selected_target = st.selectbox("Select a Pollutant to Forecast", forecast_targets)
    with col2:
        selected_model_name = st.selectbox("Select an Algorithm", list(MODELS.keys()), index=2)

    model_file_name = f"{selected_model_name.replace(' ', '_')}_{selected_target}_model.joblib"
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, model_file_name)

    df_model_features = load_data(MODEL_FEATURES_PATH)
    
    if df_model_features is None:
        st.error("Model features data not found. Please ensure the data pipeline has been run.")
    else:
        if st.button(f"Train/Retrain {selected_model_name} for {selected_target}"):
            with st.spinner(f"Training {selected_model_name} for {selected_target}..."):
                train_and_save_model(df_model_features, selected_target, selected_model_name, MODEL_SAVE_PATH)
            st.success(f"Model trained and saved successfully!")
            st.rerun()

        if not os.path.exists(MODEL_SAVE_PATH):
            st.info(f"Model not found. Please train the model using the button above.")
        else:
            loaded_object = joblib.load(MODEL_SAVE_PATH)
            
            model_wrapper = None 
            mae, start_date, end_date, train_time = (None,) * 4

            if isinstance(loaded_object, dict):
                model_wrapper = loaded_object.get('model_wrapper')
                mae = loaded_object.get('mae')
                start_date = loaded_object.get('train_start_date', 'N/A')
                end_date = loaded_object.get('train_end_date', 'N/A')
                train_time = loaded_object.get('train_timestamp', 'N/A')
            else:
                st.warning("Incompatible model file found. Please retrain the model to use the new format and see the forecast.")

            if model_wrapper:
                st.subheader("Model Information")
                with st.container(border=True):
                    col1, col2, col3 = st.columns(3)
                    col1.markdown("##### Algorithm"); col1.write(selected_model_name)
                    col2.markdown("##### Training Data Range"); col2.write(f"{start_date} to {end_date}")
                    col3.markdown("##### Last Trained"); col3.write(train_time)

                # --- RE-ADDED: The dynamic performance table ---
                st.subheader("Recent Model Performance (Last 14 Days)")
                df_recent_history = df_model_features.tail(14).copy()
                FEATURES = model_wrapper.get_feature_names()
                recent_predictions = model_wrapper.predict(df_recent_history[FEATURES])
                
                df_performance = pd.DataFrame({
                    'Date': pd.to_datetime(df_recent_history['Date']).dt.strftime('%Y-%m-%d'),
                    f'Actual {selected_target}': df_recent_history[selected_target],
                    f'Predicted {selected_target}': recent_predictions
                })
                df_performance['Error'] = df_performance[f'Actual {selected_target}'] - df_performance[f'Predicted {selected_target}']
                
                styler = df_performance.style.format({
                    f'Actual {selected_target}': '{:.2f}',
                    f'Predicted {selected_target}': '{:.2f}',
                    'Error': '{:+.2f}'
                }).hide(axis="index")
                st.dataframe(styler, use_container_width=True)

                last_data_point = df_model_features.tail(1)
                last_date = pd.to_datetime(last_data_point['Date'].iloc[0])
                future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, 8)])
                
                predictions = dynamic_predict(model_wrapper, last_data_point, future_dates, selected_target)
                df_forecast = pd.DataFrame({'Date': future_dates, f'Predicted {selected_target}': predictions})

                if selected_target == 'PM2.5':
                    st.subheader("Health Advisory for Tomorrow")
                    next_day_forecast_val = df_forecast[f'Predicted {selected_target}'].iloc[0]
                    category, recommendation, alert_type = get_aqi_alert(next_day_forecast_val)
                    with st.container(border=True):
                        st.metric(label=f"Tomorrow's Forecasted Category: {category}", value=f"{next_day_forecast_val:.2f} PM2.5")
                        if alert_type == "success": st.success(f"**Recommendation:** {recommendation}")
                        elif alert_type == "info": st.info(f"**Recommendation:** {recommendation}")
                        elif alert_type == "warning": st.warning(f"**Recommendation:** {recommendation}")
                        elif alert_type == "error": st.error(f"**Recommendation:** {recommendation}")

                st.subheader(f"7-Day Forecast Chart for {selected_target}")
                if mae:
                    df_forecast['Upper Bound'] = df_forecast[f'Predicted {selected_target}'] + mae
                    df_forecast['Lower Bound'] = df_forecast[f'Predicted {selected_target}'] - mae
                    forecast_fig = go.Figure([
                        go.Scatter(name='Upper Bound', x=df_forecast['Date'], y=df_forecast['Upper Bound'], mode='lines', line=dict(width=0.5, color='rgba(173, 216, 230, 0.5)')),
                        go.Scatter(name='Lower Bound', x=df_forecast['Date'], y=df_forecast['Lower Bound'], mode='lines', line=dict(width=0.5, color='rgba(173, 216, 230, 0.5)'), fillcolor='rgba(173, 216, 230, 0.2)', fill='tonexty'),
                        go.Scatter(name='Prediction', x=df_forecast['Date'], y=df_forecast[f'Predicted {selected_target}'], mode='lines+markers', line=dict(color='rgb(0, 100, 80)'))
                    ])
                    forecast_fig.update_layout(title=f"Forecasted {selected_target} Levels", yaxis_title='Concentration / Index', hovermode="x")
                else:
                    forecast_fig = px.line(df_forecast, x='Date', y=f'Predicted {selected_target}', title=f"Forecasted {selected_target} Levels", markers=True)
                st.plotly_chart(forecast_fig, use_container_width=True)