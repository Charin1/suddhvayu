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
    
    # --- Consolidate duplicate columns ---
    # When raw CSV has duplicate column names (PM2.5, PM2.5), pandas renames to PM2.5, PM2.5.1
    # We merge these by taking the mean of non-null values
    def consolidate_duplicate_columns(df):
        """Merge duplicate sensor columns (e.g., PM2.5 and PM2.5.1) by taking mean."""
        cols_to_drop = []
        base_cols_seen = set()
        
        for col in df.columns:
            # Check for pandas-renamed duplicate columns (e.g., "PM2.5.1", "CO.1")
            if '.' in str(col):
                parts = str(col).rsplit('.', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_col = parts[0]
                    if base_col in df.columns:
                        # Merge: take mean of base and duplicate, preserving non-null values
                        df[base_col] = df[[base_col, col]].mean(axis=1, skipna=True)
                        cols_to_drop.append(col)
                        base_cols_seen.add(base_col)
        
        if cols_to_drop:
            print(f"Consolidated duplicate columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop, errors='ignore')
        
        return df
    
    df_combined = consolidate_duplicate_columns(df_combined)
    
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
    
    # --- AQI Calculation Fallback ---
    # Convert standardized names to what CPCB expects if needed, or map them directly
    # Our data uses: PM2.5, PM10, NO2, SO2, CO, O3, NH3
    
    def calculate_sub_index(concentration, breakpoints):
        """
        Calculates sub-index for a single pollutant value.
        breakpoints: list of tuples (low_conc, high_conc, low_index, high_index)
        """
        if pd.isna(concentration):
            return np.nan
        
        for low_c, high_c, low_i, high_i in breakpoints:
            if low_c <= concentration <= high_c:
                return low_i + (high_i - low_i) * (concentration - low_c) / (high_c - low_c)
        
        # If concentration is higher than the max range, cap at 500 (Severe+)
        # We generally do not want to extrapolate to infinity
        return 500

    def get_aqi_cpcb(row):
        """
        Calculates AQI based on CPCB standards.
        Requires at least PM2.5 or PM10 to be present for a valid AQI (standard rule),
        but we will be lenient and calculate if any valid sub-index exists to fill gaps.
        """
        # Breakpoints (Conc Low, Conc High, AQI Low, AQI High)
        # 24-hr avg for PM, NO2, SO2, NH3. 8-hr max for CO, O3.
        # Assuming input data 'PM2.5', 'PM10' etc represent these averages.
        
        bp_pm25 = [
            (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
            (91, 120, 201, 300), (121, 250, 301, 400), (251, 380, 401, 500)
        ]
        bp_pm10 = [
            (0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
            (251, 350, 201, 300), (351, 430, 301, 400), (431, 510, 401, 500)
        ]
        bp_no2 = [
            (0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
            (181, 280, 201, 300), (281, 400, 301, 400), (401, 520, 401, 500)
        ]
        bp_so2 = [
            (0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
            (381, 800, 201, 300), (801, 1600, 301, 400), (1600, 2400, 401, 500) # Estimated upper
        ]
        bp_co = [
            (0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200),
            (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 51, 401, 500)
        ]
        bp_o3 = [
            (0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
            (169, 208, 201, 300), (209, 748, 301, 400), (748, 1000, 401, 500) # Estimated
        ]
        bp_nh3 = [
            (0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200),
            (801, 1200, 201, 300), (1200, 1800, 301, 400), (1801, 2400, 401, 500)
        ]

        sub_indices = []
        
        # Helper to safely get value and handle unit issues
        def get_val(col_name):
            if col_name not in row: return None
            val = row[col_name]
            if pd.isna(val): return None
            
            try:
                val = float(val)
            except:
                return None

            # Heuristic for CO: CPCB uses mg/m3 (0-50 range roughly).
            # Historical data often uses ug/m3 (values in thousands).
            # Use threshold > 100 to distinguish (typical mg/m3 range is 0-50).
            if col_name == 'CO':
                 if val > 100:
                     val = val / 1000.0
            
            return val

        # Calculate indices
        si_pm25 = calculate_sub_index(get_val('PM2.5'), bp_pm25)
        si_pm10 = calculate_sub_index(get_val('PM10'), bp_pm10)
        si_no2 = calculate_sub_index(get_val('NO2'), bp_no2)
        si_so2 = calculate_sub_index(get_val('SO2'), bp_so2)
        si_co = calculate_sub_index(get_val('CO'), bp_co)
        si_o3 = calculate_sub_index(get_val('O3'), bp_o3)
        si_nh3 = calculate_sub_index(get_val('NH3'), bp_nh3)

        sub_indices = [si_pm25, si_pm10, si_no2, si_so2, si_co, si_o3, si_nh3]
        
        valid_indices = [si for si in sub_indices if pd.notna(si)]
        
        if not valid_indices:
            return np.nan
        
        final_aqi = max(valid_indices)
        # if final_aqi == 500:
        #      names = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3']
        #      try:
        #          culprit_idx = sub_indices.index(500)
        #          print(f"DEBUG: Row triggered 500 due to {names[culprit_idx]} with val {get_val(names[culprit_idx])}")
        #      except ValueError:
        #          pass
             
        return final_aqi

    # Always calculate AQI from pollutants to ensure consistency and fix potential data gaps or default values (like 90)
    print("Recalculating AQI from pollutants...")
    
    # Calculate AQI for ALL rows, not just missing ones
    # This overrides any existing 'AQI' column which might contain garbage or default values
    computed_aqi = df_combined.apply(get_aqi_cpcb, axis=1)
    
    # If we have a computed AQI, use it. If computed is NaN (missing pollutants), fallback to existing AQI if available.
    if 'AQI' in df_combined.columns:
        df_combined['AQI'] = computed_aqi.fillna(df_combined['AQI'])
    else:
        df_combined['AQI'] = computed_aqi
    
    # Final cleanup for any remaining NaNs
    df_combined['AQI'] = df_combined['AQI'].interpolate(method='linear', limit_direction='both')

    # --- End AQI Calculation ---
    
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