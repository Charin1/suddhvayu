import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from src.data_loader import load_kaggle_data
from src.data_cleaner import clean_air_quality

DATA_PATH = "data/raw/city_day.csv"

st.title("🌍 ShuddhVayu - Gujarat Air Quality Dashboard")

# Load + clean
df = load_kaggle_data(DATA_PATH)
df_clean = clean_air_quality(df)

# Sidebar city selection
city = st.sidebar.selectbox("Select City", df_clean["City"].unique())

# Filter city data
city_df = df_clean[df_clean["City"] == city]

st.subheader(f"Air Quality in {city}")

# Show last few rows
st.write(city_df.tail())

# Let user select pollutants
pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", 
              "Benzene", "Toluene", "Xylene", "AQI"]

selected = st.multiselect("Select pollutants to plot", pollutants, default=["PM2.5", "PM10"])

if selected:
    plot_df = city_df.set_index("Date")[selected]

    # Keep only numeric cols (avoid AQI_Bucket, City, etc.)
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")

    st.line_chart(plot_df, use_container_width=True)
else:
    st.warning("Please select at least one pollutant to plot.")

st.write("Debug preview:", plot_df.head())

