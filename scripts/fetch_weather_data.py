"""
ShuddhVayu Unified Data Fetcher
================================
Fetches both:
1. Weather data from Open-Meteo API (free, no key required)
2. AQI/Pollutant data from OpenAQ API v3 (free key from openaq.org)

To get OpenAQ API key (FREE):
1. Register at https://explore.openaq.org/register
2. After login, go to Profile -> API Keys
3. Create a new key and add it to .env as OPENAQ_API_KEY

Without an API key, only weather data will be fetched.
"""
import requests
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "").strip()


# Gujarat city coordinates
CITY_CONFIG = {
    "Ahmedabad": {
        "coords": (23.02, 72.58),
        "state": "Gujarat"
    },
    "Gandhinagar": {
        "coords": (23.22, 72.68),
        "state": "Gujarat"
    },
    "Surat": {
        "coords": (21.17, 72.83),
        "state": "Gujarat"
    },
    "Rajkot": {
        "coords": (22.30, 70.80),
        "state": "Gujarat"
    },
    "Vadodara": {
        "coords": (22.30, 73.18),
        "state": "Gujarat"
    }
}


def fetch_weather_data(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical daily weather data from Open-Meteo API.
    Free, no API key required.
    """
    print(f"[Weather] Fetching for {city} ({start_date} to {end_date})...")
    
    # Validation: Open-Meteo Archive is strictly historical (up to yesterday/2 days lag)
    try:
        dt_end = datetime.strptime(end_date, "%Y-%m-%d")
        dt_start = datetime.strptime(start_date, "%Y-%m-%d")
        dt_yesterday = datetime.now() - timedelta(days=2) # Safe lag
        
        if dt_end > dt_yesterday:
            print(f"[Weather] ⚠️  Adjusting end_date from {end_date} to {dt_yesterday.strftime('%Y-%m-%d')} (Archive delay)")
            end_date = dt_yesterday.strftime("%Y-%m-%d")
            
        if dt_start > dt_end: # If adjustment made start > end
             dt_start = dt_end - timedelta(days=1)
             start_date = dt_start.strftime("%Y-%m-%d")
             
    except Exception as e:
        print(f"[Weather] Date parsing warning: {e}")

    config = CITY_CONFIG.get(city, CITY_CONFIG["Ahmedabad"])
    lat, lon = config["coords"]
    
    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "shortwave_radiation_sum",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant"
        ],
        "timezone": "Asia/Kolkata"
    }

    try:
        response = requests.get(API_URL, params=params, timeout=60)
        
        if response.status_code != 200:
            print(f"[Weather] Error: Status {response.status_code}")
            return pd.DataFrame()

        data = response.json()
        if 'daily' not in data:
            print("[Weather] Error: No daily data in response")
            return pd.DataFrame()

        df = pd.DataFrame(data['daily'])
        df.rename(columns={'time': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df['City'] = city
        
        print(f"[Weather] ✅ Retrieved {len(df)} days of data")
        return df

    except Exception as e:
        print(f"[Weather] Exception: {e}")
        return pd.DataFrame()


def fetch_openaq_aqi_data(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches AQI data from OpenAQ API v3.
    Requires free API key from https://explore.openaq.org/register
    Fetches daily historical data for ALL parameters found at the station.
    """
    if not OPENAQ_API_KEY:
        print("[AQI] ⚠️  No OpenAQ API key found. Set OPENAQ_API_KEY in .env")
        print("[AQI] Get free API key at: https://explore.openaq.org/register")
        return pd.DataFrame()
    
    print(f"[AQI] Fetching for {city} from OpenAQ...")
    
    config = CITY_CONFIG.get(city, CITY_CONFIG["Ahmedabad"])
    lat, lon = config["coords"]
    
    BASE_URL = "https://api.openaq.org/v3"
    headers = {}
    if OPENAQ_API_KEY:
        headers["X-API-Key"] = OPENAQ_API_KEY
    
    try:
        # Step 1: Find locations near city coordinates
        print(f"[AQI] Searching for stations near {city}...")
        
        params = {
            "coordinates": f"{lat},{lon}",
            "radius": 25000,  # 25km radius (API Limit)
            "limit": 10
        }
        
        resp = requests.get(f"{BASE_URL}/locations", params=params, headers=headers, timeout=30)
        
        if resp.status_code == 401:
            print("[AQI] ❌ Invalid API key. Check your OPENAQ_API_KEY in .env")
            return pd.DataFrame()
        
        if resp.status_code != 200:
            print(f"[AQI] Error finding locations: {resp.status_code}")
            return pd.DataFrame()
        
        loc_data = resp.json()
        locations = loc_data.get("results", [])
        
        if not locations:
            print(f"[AQI] No stations found near {city}")
            return pd.DataFrame()
        
        # Get the closest location
        location = locations[0]
        location_id = location.get("id")
        location_name = location.get("name", "Unknown")
        print(f"[AQI] Found station: {location_name} (ID: {location_id})")
        
        # Step 2: Get sensors for this location
        search_sensors = location.get("sensors", [])
        
        if not search_sensors:
             # Fetch full location details if sensors missing
             detail_resp = requests.get(f"{BASE_URL}/locations/{location_id}", headers=headers, timeout=30)
             if detail_resp.status_code == 200:
                 search_sensors = detail_resp.json().get("results", [])[0].get("sensors", [])
        
        if not search_sensors:
            print(f"[AQI] No sensors found for station {location_id}")
            return pd.DataFrame()
            
        print(f"[AQI] Found {len(search_sensors)} sensors. Fetching historical data ({start_date} to {end_date})...")
        
        all_frames = []
        
        for sensor in search_sensors:
            try:
                sensor_id = sensor.get("id")
                # Standardize parameter names
                raw_param = sensor.get("parameter", {}).get("name", "unknown")
                
                # Mapping dictionary for common pollutants
                PARAM_MAP = {
                    "pm25": "PM2.5", "pm2.5": "PM2.5",
                    "pm10": "PM10",
                    "no2": "NO2",
                    "so2": "SO2",
                    "co": "CO",
                    "o3": "O3",
                    "no": "NO"
                }
                
                # Normalize: lowercase check -> Mapped or Uppercase
                parameter = PARAM_MAP.get(raw_param.lower(), raw_param.upper())
                
                # Fetch DAILY averages
                hist_params = {
                    "date_from": start_date,
                    "date_to": end_date,
                    "limit": 1000
                }
                
                hist_resp = requests.get(
                    f"{BASE_URL}/sensors/{sensor_id}/measurements/daily", 
                    headers=headers, 
                    params=hist_params,
                    timeout=30
                )
                
                if hist_resp.status_code != 200:
                    continue
                    
                hist_data = hist_resp.json().get("results", [])
                if not hist_data:
                    continue
                    
                records = []
                for h in hist_data:
                    dt_str = h.get("period", {}).get("datetimeFrom", {}).get("local", "")
                    if not dt_str:
                        continue
                        
                    val = h.get("value")
                    records.append({
                        "Date": dt_str.split("T")[0], # YYYY-MM-DD
                        parameter: val
                    })
                
                if records:
                    df_sensor = pd.DataFrame(records)
                    df_sensor['Date'] = pd.to_datetime(df_sensor['Date'])
                    df_sensor.set_index('Date', inplace=True)
                    df_sensor = df_sensor.groupby('Date').mean()
                    all_frames.append(df_sensor)
                    print(f"[AQI] > Fetched {len(df_sensor)} days for {parameter}")
                    
            except Exception as e:
                print(f"[AQI] Error fetching sensor {sensor.get('id')}: {e}")
                continue

        if not all_frames:
            print("[AQI] No historical data found for any sensor.")
            return pd.DataFrame()
            
        # Step 3: Merge all sensor dataframes
        df_final = pd.concat(all_frames, axis=1).reset_index()
        df_final['City'] = city
        
        print(f"[AQI] ✅ Retrieved historical data with {len(df_final.columns) - 2} pollutants")
        return df_final

    except Exception as e:
        print(f"[AQI] Exception: {e}")
        return pd.DataFrame()


def fetch_aqi_data(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Main AQI fetch function using OpenAQ.
    """
    return fetch_openaq_aqi_data(city, start_date, end_date)


def save_data(weather_df: pd.DataFrame, aqi_df: pd.DataFrame, city: str):
    """
    Saves weather and AQI data to files.
    """
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    city_lower = city.lower()
    
    # Save weather data
    if not weather_df.empty:
        weather_path = os.path.join(output_dir, f"{city_lower}_weather.csv")
        weather_df.to_csv(weather_path, index=False)
        print(f"[Save] ✅ Weather -> {weather_path}")
    
    # Save AQI data
    if not aqi_df.empty:
        aqi_path = os.path.join(output_dir, f"{city_lower}_aqi.csv")
        aqi_df.to_csv(aqi_path, index=False)
        print(f"[Save] ✅ AQI -> {aqi_path}")
    
    # Create combined file if both available
    if not weather_df.empty and not aqi_df.empty:
        try:
            merged = pd.merge(weather_df, aqi_df, on=["Date", "City"], how="outer")
            merged_path = os.path.join(output_dir, f"{city_lower}_combined.csv")
            merged.to_csv(merged_path, index=False)
            print(f"[Save] ✅ Combined -> {merged_path}")
        except Exception as e:
            print(f"[Save] Could not merge: {e}")
    
    print(f"\n[Done] Data fetch complete for {city}!")


def main():
    parser = argparse.ArgumentParser(description="Fetch weather and AQI data for Gujarat cities.")
    parser.add_argument("--city", type=str, default="Ahmedabad", help="City name")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--weather_only", action="store_true", help="Fetch only weather data")
    parser.add_argument("--aqi_only", action="store_true", help="Fetch only AQI data")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  ShuddhVayu Data Fetcher")
    print(f"  City: {args.city} | {args.start_date} to {args.end_date}")
    print(f"{'='*60}\n")
    
    weather_df = pd.DataFrame()
    aqi_df = pd.DataFrame()
    
    if not args.aqi_only:
        weather_df = fetch_weather_data(args.city, args.start_date, args.end_date)
    
    if not args.weather_only:
        aqi_df = fetch_aqi_data(args.city, args.start_date, args.end_date)
    
    save_data(weather_df, aqi_df, args.city)


if __name__ == "__main__":
    main()