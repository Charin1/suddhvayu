import pandas as pd
import os

def load_kaggle_data(filepath: str) -> pd.DataFrame:
    """
    Load Kaggle India Air Quality dataset.
    Expected columns: ['City', 'Date', 'PM2.5', 'PM10', 'NO2, 'SO2', ...]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    
    df = pd.read_csv(filepath, parse_dates=['Date'])
    return df
