import pandas as pd
import numpy as np
import os
import sys

def process_data():
    # Define paths for all files
    raw_aqi_path = os.path.join("data", "raw", "city_day.csv")
    raw_weather_path = os.path.join("data", "raw", "ahmedabad_weather.csv") # New path
    processed_dir = os.path.join("data", "processed")
    cleaned_path = os.path.join(processed_dir, "gujarat_aqi.csv")
    features_path = os.path.join(processed_dir, "gujarat_features_for_model.csv")
    os.makedirs(processed_dir, exist_ok=True)

    # Load AQI data
    print("Loading raw AQI data...")
    df_aqi = pd.read_csv(raw_aqi_path, parse_dates=['Date'])
    
    # --- NEW: Load Weather Data ---
    print("Loading raw weather data...")
    if not os.path.exists(raw_weather_path):
        print(f"ERROR: Weather data not found at {raw_weather_path}", file=sys.stderr)
        print("Please run 'python scripts/01_fetch_weather_data.py' first.", file=sys.stderr)
        sys.exit(1)
    df_weather = pd.read_csv(raw_weather_path, parse_dates=['Date'])

    # Filter and clean AQI data
    gujarat_cities = ['Ahmedabad', 'Gandhinagar']
    df_gujarat = df_aqi[df_aqi['City'].isin(gujarat_cities)].copy()
    print(f"Filtered for Gujarat cities. Found {df_gujarat.shape[0]} rows.")
    df_gujarat.set_index('Date', inplace=True)
    df_gujarat.sort_values(by=['City', 'Date'], inplace=True)
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI']
    for col in pollutant_cols:
        df_gujarat[col] = df_gujarat.groupby('City')[col].transform(lambda x: x.interpolate(method='time', limit_direction='both'))
    df_gujarat.reset_index(inplace=True)
    df_gujarat.dropna(subset=['PM2.5', 'AQI'], inplace=True)
    print("Cleaned data and handled missing values.")
    df_gujarat.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to {cleaned_path}")

    # --- NEW: Merge AQI and Weather data for Ahmedabad ---
    df_ahmedabad = df_gujarat[df_gujarat['City'] == 'Ahmedabad'].copy()
    print("Merging AQI and weather data...")
    df_model = pd.merge(df_ahmedabad, df_weather, on='Date', how='left')
    
    # Interpolate any potential missing weather data points
    weather_cols = df_weather.columns.drop('Date')
    df_model[weather_cols] = df_model[weather_cols].interpolate(method='linear', limit_direction='both')

    # Dynamic Feature Engineering (this part is unchanged, but now acts on the merged data)
    df_model['day_of_week'] = df_model['Date'].dt.dayofweek
    df_model['month'] = df_model['Date'].dt.month
    df_model['year'] = df_model['Date'].dt.year
    df_model['day_of_year'] = df_model['Date'].dt.dayofyear
    potential_targets = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']
    for target in potential_targets:
        for i in range(1, 8):
            df_model[f'{target}_lag_{i}'] = df_model[target].shift(i)
        df_model[f'{target}_roll_mean_7'] = df_model[target].shift(1).rolling(window=7).mean()
        df_model[f'{target}_roll_std_7'] = df_model[target].shift(1).rolling(window=7).std()
    
    print(f"Shape before dropping NaNs from feature engineering: {df_model.shape}")
    df_model.dropna(subset=[f'{potential_targets[0]}_roll_mean_7'], inplace=True)
    print(f"Engineered features for all potential targets. Shape: {df_model.shape}")

    if df_model.empty:
        print("ERROR: Feature-engineered DataFrame is empty.", file=sys.stderr)
        sys.exit(1)

    df_model.to_csv(features_path, index=False)
    print(f"Feature-engineered data saved to {features_path}")

if __name__ == "__main__":
    process_data()