import pandas as pd
import numpy as np
import os
import sys

def process_data():
    """
    Loads raw city-level air quality data, filters for Gujarat,
    cleans it, and saves two files: one for visualization and
    one with features for modeling.
    """
    # Define paths
    raw_data_path = os.path.join("data", "raw", "city_day.csv")
    processed_dir = os.path.join("data", "processed")
    cleaned_path = os.path.join(processed_dir, "gujarat_aqi.csv")
    features_path = os.path.join(processed_dir, "gujarat_features_for_model.csv")

    os.makedirs(processed_dir, exist_ok=True)

    print("Loading raw data...")
    if not os.path.exists(raw_data_path):
        print(f"ERROR: Raw data file not found at {raw_data_path}", file=sys.stderr)
        sys.exit(1)
        
    df = pd.read_csv(raw_data_path, parse_dates=['Date'])

    # --- FIX: Corrected city name spelling back to Ahmedabad ---
    gujarat_cities = ['Ahmedabad', 'Gandhinagar'] 
    
    print(f"Filtering for cities: {gujarat_cities}")
    df_gujarat = df[df['City'].isin(gujarat_cities)].copy()

    if df_gujarat.empty:
        print("ERROR: No data found for the specified cities.", file=sys.stderr)
        sys.exit(1)

    print(f"Filtered for Gujarat cities. Found {df_gujarat.shape[0]} rows.")

    # Data Cleaning
    df_gujarat.set_index('Date', inplace=True)
    df_gujarat.sort_values(by=['City', 'Date'], inplace=True)

    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
    for col in pollutant_cols:
        df_gujarat[col] = df_gujarat.groupby('City')[col].transform(
            lambda x: x.interpolate(method='time', limit_direction='both')
        )
    
    df_gujarat.reset_index(inplace=True)
    df_gujarat.dropna(subset=['PM2.5', 'AQI'], inplace=True)
    print("Cleaned data and handled missing values.")

    df_gujarat.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to {cleaned_path}")

    # --- Feature Engineering for Modeling (Ahmedabad only) ---
    # --- FIX: Corrected city name spelling back to Ahmedabad ---
    df_model = df_gujarat[df_gujarat['City'] == 'Ahmedabad'].copy()

    df_model['day_of_week'] = df_model['Date'].dt.dayofweek
    df_model['month'] = df_model['Date'].dt.month
    df_model['year'] = df_model['Date'].dt.year
    df_model['day_of_year'] = df_model['Date'].dt.dayofyear

    target = 'PM2.5'
    for i in range(1, 8):
        df_model[f'{target}_lag_{i}'] = df_model[target].shift(i)

    df_model[f'{target}_roll_mean_7'] = df_model[target].shift(1).rolling(window=7).mean()
    df_model[f'{target}_roll_std_7'] = df_model[target].shift(1).rolling(window=7).std()
    
    print(f"Shape before dropping NaNs from feature engineering: {df_model.shape}")
    df_model.dropna(subset=[f'{target}_roll_mean_7'], inplace=True)
    print(f"Shape after dropping NaNs: {df_model.shape}")

    if df_model.empty:
        print("ERROR: Feature-engineered DataFrame is empty after dropping NaNs.", file=sys.stderr)
        sys.exit(1)

    df_model.to_csv(features_path, index=False)
    print(f"Feature-engineered data saved to {features_path}")

if __name__ == "__main__":
    process_data()