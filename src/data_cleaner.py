import pandas as pd

def clean_air_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter Gujarat air quality data"""
    gujarat_cities = [
        "Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar", 
        "Bhavnagar", "Jamnagar", "Junagadh", "Anand"
    ]
    
    # Filter Gujarat cities
    df = df[df["City"].isin(gujarat_cities)].copy()
    
    # Parse Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])  # drop rows where Date failed
    
    # Fill missing values forward/backward
    df = df.fillna(method="ffill").fillna(method="bfill")
    
    return df
