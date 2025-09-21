# ShuddhVayu - Gujarat Air Quality Dashboard

An interactive dashboard for air quality analysis and forecasting in Gujarat, India. This tool provides a comprehensive platform for exploring historical air quality data and generating multi-model, multi-pollutant forecasts for Ahmedabad.


---

## Features

This application is divided into two main sections, accessible via tabs:

### 📊 Exploratory Data Analysis (EDA)

This tab provides a suite of tools to analyze historical air quality data for major cities in Gujarat:
*   **City Selection:** Dynamically filter the entire analysis for a specific city (e.g., Ahmedabad, Gandhinagar).
*   **Trend Analysis:** Superimpose and compare historical trends for multiple pollutants (PM2.5, PM10, AQI, etc.) on a single interactive chart.
*   **Pollutant Distribution:** Visualize the yearly variation and distribution of any selected pollutant using box plots to identify medians, ranges, and outliers.
*   **Data Quality Analysis:** View a summary of missing values from the original raw dataset to understand the initial data quality.

### 🔮 Ahmedabad Forecast

A dedicated forecasting playground for Ahmedabad's air quality, featuring:
*   **Dynamic Pollutant Forecasting:** Select any major pollutant (PM2.5, PM10, AQI, etc.) to be the target for forecasting.
*   **Multi-Model Selection:** Choose from multiple machine learning algorithms to train and generate forecasts:
    *   Linear Regression
    *   Support Vector Regressor (SVR)
    *   XGBoost Regressor
*   **On-Demand Model Training:** Train a new model with a single click, with all preprocessing and feature engineering handled automatically.
*   **Model Transparency:** View key metadata for the currently loaded model, including the algorithm, training data date range, and the exact time it was last trained.
*   **Performance Evaluation:** A clear table shows the model's performance on the 14 most recent days, comparing `Actual` vs. `Predicted` values and showing the `Error`.
*   **7-Day Forecast Chart:** An interactive chart displays the forecast for the next 7 days, complete with a confidence interval based on the model's Mean Absolute Error (MAE).
*   **PM2.5 Health Advisory:** When forecasting for PM2.5, the dashboard provides a clear, color-coded health advisory and recommendation for the next day.

---

## Technology Stack

*   **Backend & Modeling:** Python, Pandas, Scikit-learn, XGBoost
*   **Frontend & UI:** Streamlit
*   **Data Visualization:** Plotly
*   **Data Fetching:** Requests

---

## Project Structure

```
.
├── dashboard/
│   └── app.py              # Main Streamlit application UI and logic
├── data/
│   ├── processed/          # Cleaned & feature-engineered data
│   └── raw/                # Raw data (from Kaggle and weather API)
├── models/                 # Saved (trained) model files
├── scripts/
│   └── fetch_weather_data.py # One-time script to download weather data
├── src/
│   ├── eda.py              # Functions for generating EDA plots
│   └── models.py           # Classes and functions for multi-model training & prediction
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ShuddhVayu
```

### 2. Create and Activate a Virtual Environment
Using `venv` (standard):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Or using `uv` (faster):
```bash
uv venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# Or using uv: uv pip install -r requirements.txt
```

---

## How to Run the Application

The application requires a three-step process to run for the first time.

### Step 1: Fetch Weather Data (One-Time Setup)
Run this script once to download the necessary historical weather data.
```bash
python scripts/fetch_weather_data.py
```
This will create `ahmedabad_weather.csv` in the `data/raw/` directory.

### Step 2: Run the Data Pipeline
This script cleans the raw data, merges it with the weather data, and creates the final feature set for the models.
```bash
python src/data_pipeline.py
```
**Note:** You should re-run this script whenever you update the raw data or change the feature engineering logic.

### Step 3: Launch the Streamlit App
Run the application from the root project directory.
```bash
python -m streamlit run dashboard/app.py
```
