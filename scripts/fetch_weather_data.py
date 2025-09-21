import requests
import pandas as pd
import os

def fetch_weather_data():
    """
    Downloads historical daily weather data for Ahmedabad (2015-2020)
    from the Open-Meteo API and saves it to a CSV file.
    """
    print("Fetching historical weather data for Ahmedabad...")

    # Define API parameters
    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 23.02,
        "longitude": 72.58,
        "start_date": "2015-01-01",
        "end_date": "2020-12-31",
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

    # Make the API request
    response = requests.get(API_URL, params=params)

    if response.status_code != 200:
        print(f"Error: Failed to fetch data. Status code: {response.status_code}")
        print(response.json())
        return

    # Process the data
    data = response.json()
    df = pd.DataFrame(data['daily'])
    
    # Rename 'time' to 'Date' to match our existing data
    df.rename(columns={'time': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define the output path
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ahmedabad_weather.csv")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully downloaded and saved weather data to {output_path}")

if __name__ == "__main__":
    fetch_weather_data()