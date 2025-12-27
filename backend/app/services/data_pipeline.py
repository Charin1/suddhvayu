import pandas as pd
import numpy as np
import os
import sys

def process_data(city=None):
    """
    Runs the data processing pipeline.
    Uses city-specific AQI and weather files: {city}_aqi.csv, {city}_weather.csv
    Or combined file: {city}_combined.csv
    """
    if not city:
        city = "Ahmedabad"  # Default city
    
    city_lower = city.lower()
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Output paths
    cleaned_path = os.path.join(processed_dir, f"{city_lower}_aqi_cleaned.csv")
    features_path = os.path.join(processed_dir, "gujarat_features_for_model.csv")
    
    # Check for combined file first (preferred)
    combined_path = os.path.join(raw_dir, f"{city_lower}_combined.csv")
    aqi_path = os.path.join(raw_dir, f"{city_lower}_aqi.csv")
    weather_path = os.path.join(raw_dir, f"{city_lower}_weather.csv")
    legacy_path = os.path.join(raw_dir, "city_day.csv")
    
    df_combined = None
    
    # Try loading data in order of preference
    if os.path.exists(combined_path):
        print(f"Loading combined data from {combined_path}...")
        df_combined = pd.read_csv(combined_path, parse_dates=['Date'])
        df_combined['City'] = city
    elif os.path.exists(aqi_path) and os.path.exists(weather_path):
        print(f"Loading and merging AQI + Weather data...")
        df_aqi = pd.read_csv(aqi_path, parse_dates=['Date'])
        df_weather = pd.read_csv(weather_path, parse_dates=['Date'])
        df_combined = pd.merge(df_aqi, df_weather, on=['Date', 'City'], how='outer')
    elif os.path.exists(aqi_path):
        print(f"Loading AQI data only from {aqi_path}...")
        df_combined = pd.read_csv(aqi_path, parse_dates=['Date'])
        df_combined['City'] = city
    elif os.path.exists(weather_path):
        print(f"Loading weather data only from {weather_path}...")
        df_combined = pd.read_csv(weather_path, parse_dates=['Date'])
        df_combined['City'] = city
    elif os.path.exists(legacy_path):
        # Fallback to legacy city_day.csv
        print(f"Falling back to legacy city_day.csv...")
        df_all = pd.read_csv(legacy_path, parse_dates=['Date'])
        df_combined = df_all[df_all['City'] == city].copy()
        
        # Try to merge with weather if available
        if os.path.exists(weather_path):
            df_weather = pd.read_csv(weather_path, parse_dates=['Date'])
            df_combined = pd.merge(df_combined, df_weather, on='Date', how='left')
    else:
        print(f"Error: No data files found for {city}. Please run 'Fetch Weather' in Data Ops first.", file=sys.stderr)
        return
    
    if df_combined is None or df_combined.empty:
        print(f"Error: No data loaded for {city}", file=sys.stderr)
        return
    
    print(f"Loaded {len(df_combined)} rows for {city}")
    
    # Ensure City column exists
    if 'City' not in df_combined.columns:
        df_combined['City'] = city
    
    # Sort and set index
    df_combined = df_combined.sort_values('Date')
    
    # Interpolate pollutant columns
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI', 'pm25', 'pm10', 'no2', 'co', 'so2', 'o3']
    existing_pollutants = [col for col in pollutant_cols if col in df_combined.columns]
    
    for col in existing_pollutants:
        df_combined[col] = df_combined[col].interpolate(method='linear', limit_direction='both')
    
    # Save cleaned data
    df_combined.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to {cleaned_path}")
    
    # Legacy common file saving removed to prevent overlap
    # formatted_path now serves as the source of truth
    
    # ========== FEATURE ENGINEERING ==========
    df_model = df_combined.copy()
    
    # Interpolate weather columns if present
    weather_cols = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 
                    'precipitation_sum', 'rain_sum', 'wind_speed_10m_max', 
                    'shortwave_radiation_sum', 'wind_direction_10m_dominant']
    existing_weather = [col for col in weather_cols if col in df_model.columns]
    
    for col in existing_weather:
        df_model[col] = df_model[col].interpolate(method='linear', limit_direction='both')
    
    # Date Features
    df_model['day_of_week'] = df_model['Date'].dt.dayofweek
    df_model['month'] = df_model['Date'].dt.month
    df_model['year'] = df_model['Date'].dt.year
    df_model['day_of_year'] = df_model['Date'].dt.dayofyear
    
    # Lag Features for target columns
    # Standardized list based on fetch_weather_data.py mapping
    potential_targets = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'NO', 'AQI']
    existing_targets = [t for t in potential_targets if t in df_model.columns]
    
    for target in existing_targets:
        for i in range(1, 8):
            df_model[f'{target}_lag_{i}'] = df_model[target].shift(i)
        df_model[f'{target}_roll_mean_7'] = df_model[target].shift(1).rolling(window=7).mean()
        df_model[f'{target}_roll_std_7'] = df_model[target].shift(1).rolling(window=7).std()
    
    # Drop rows with NaN in rolling features
    rolling_cols = [f'{t}_roll_mean_7' for t in existing_targets if f'{t}_roll_mean_7' in df_model.columns]
    
    print(f"Shape before dropping NaNs: {df_model.shape}")
    if rolling_cols:
        df_model = df_model.dropna(subset=rolling_cols)
    
    if df_model.empty:
        print("Warning: Feature-engineered DataFrame is empty after dropping NaNs", file=sys.stderr)
        return
    
    print(f"Feature-engineered data shape: {df_model.shape}")
    
    # Updated filename to be city-specific to avoid overwriting
    features_filename = f"{city.lower()}_features.csv"
    features_path = os.path.join(processed_dir, features_filename)
    
    df_model.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")
    print(f"Pipeline complete for {city}!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="Ahmedabad")
    args = parser.parse_args()
    process_data(args.city)