import pandas as pd
import numpy as np
import os
import sys

def process_data():
    raw_data_path = os.path.join("data", "raw", "city_day.csv")
    processed_dir = os.path.join("data", "processed")
    cleaned_path = os.path.join(processed_dir, "gujarat_aqi.csv")
    features_path = os.path.join(processed_dir, "gujarat_features_for_model.csv")
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading raw data...")
    df = pd.read_csv(raw_data_path, parse_dates=['Date'])
    
    gujarat_cities = ['Ahmedabad', 'Gandhinagar']
    df_gujarat = df[df['City'].isin(gujarat_cities)].copy()
    print(f"Filtered for Gujarat cities. Found {df_gujarat.shape[0]} rows.")

    df_gujarat.set_index('Date', inplace=True)
    df_gujarat.sort_values(by=['City', 'Date'], inplace=True)
    
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI']
    for col in pollutant_cols:
        df_gujarat[col] = df_gujarat.groupby('City')[col].transform(lambda x: x.interpolate(method='time', limit_direction='both'))
    
    df_gujarat.reset_index(inplace=True)
    
    # --- FIX: Reverted to a less aggressive dropna. ---
    # Instead of requiring all pollutants, we only require the most critical ones.
    # This prevents the entire dataframe from being deleted.
    df_gujarat.dropna(subset=['PM2.5', 'AQI'], inplace=True)
    
    print("Cleaned data and handled missing values.")
    df_gujarat.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to {cleaned_path}")

    # Dynamic Feature Engineering for Multiple Pollutants
    df_model = df_gujarat[df_gujarat['City'] == 'Ahmedabad'].copy()
    
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
    
    # --- FIX: Use a more robust dropna for the final feature set ---
    # This drops only the initial rows that are unusable for any model.
    print(f"Shape before dropping NaNs from feature engineering: {df_model.shape}")
    df_model.dropna(subset=[f'{potential_targets[0]}_roll_mean_7'], inplace=True)
    print(f"Engineered features for all potential targets. Shape: {df_model.shape}")

    if df_model.empty:
        print("ERROR: Feature-engineered DataFrame is still empty. Check the raw data for large gaps.", file=sys.stderr)
        sys.exit(1)

    df_model.to_csv(features_path, index=False)
    print(f"Feature-engineered data saved to {features_path}")

if __name__ == "__main__":
    process_data()