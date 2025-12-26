# ShuddhVayu - Technical Documentation & Developer Manual

This document provides a comprehensive deep-dive into the ShuddhVayu codebase. It is intended for developers who wish to understand the internal workings, extend the forecasting engine, or modify the dashboard.

---

## 1. System Architecture

The application follows a modular architecture separating data processing, model logic, and the presentation layer.

```mermaid
graph TD
    RawData[Raw Data (CSV)] --> Pipeline[src/data_pipeline.py]
    Pipeline --> ProcessedData[Processed Data & Features]
    
    subgraph "Model Layer (src/)"
        ProcessedData --> Classical[classical_models.py]
        ProcessedData --> DL[dl_models.py]
        Classical --> Joblib[Saved Models (.joblib)]
        DL --> Joblib
    end
    
    subgraph "Dashboard Layer (dashboard/)"
        ProcessedData --> EDA[eda.py]
        Joblib --> App[app.py]
        App --> UserUI[Streamlit UI]
    end
```

---

## 2. Module Reference

### 2.1 Data Pipeline (`src/data_pipeline.py`)

This module handles the Extract-Transform-Load (ETL) process. It is responsible for cleaning raw data and engineering features for the machine learning models.

#### Functions

**`process_data()`**
*   **Purpose**: Main execution function triggered when running the script.
*   **Workflow**:
    1.  **Ingestion**: Reads `data/raw/city_day.csv` and `data/raw/ahmedabad_weather.csv`.
    2.  **Filtering**: Selects data only for specific Gujarat cities (currently 'Ahmedabad' and 'Gandhinagar').
    3.  **Imputation (Stage 1)**: Fills missing pollutant values using time-based interpolation per city.
    4.  **Weather Merge**: Merges external weather data for Ahmedabad (Temperature, Humidity, Wind Speed, etc.).
    5.  **Feature Engineering**:
        *   **Date Features**: Extracts day of week, month, year, day of year.
        *   **Lag Features**: Creates lags 1 through 7 for every target pollutant (e.g., `PM2.5_lag_1`).
        *   **Rolling Stats**: Calculates 7-day rolling mean and standard deviation.
    6.  **Cleaning (Stage 2)**: Drops rows containing NaNs in essential engineered features to ensure model stability.
    7.  **Output**: Saves `gujarat_aqi.csv` (for EDA) and `gujarat_features_for_model.csv` (for training).

---

### 2.2 Classical Models (`src/classical_models.py`)

Handles traditional machine learning algorithms using Scikit-Learn and XGBoost.

#### Classes

**`BaseModel`**
*   **Description**: A generic wrapper for Scikit-Learn regressors.
*   **`__init__(self, model)`**: Initializes with a base sklearn estimator.
*   **`train(self, X_train, y_train, X_val=None, y_val=None)`**: Fits the pipeline (Imputer + Regressor). validation data is accepted for API consistency but not used by standard sklearn `fit`.
*   **`predict(self, X)`**: Returns predictions.

**`XGBoostModel`**
*   **Description**: A specialized wrapper for XGBoost to utilize early stopping.
*   **`__init__(self)`**: Configures `XGBRegressor` with `n_estimators=1000`, `learning_rate=0.05`, and `early_stopping_rounds=50`.
*   **`train(...)`**: Fits the model using the validation set (`eval_set`) to stop training when performance plateaus.

#### Global Variables
*   **`CLASSICAL_MODELS`**: Dictionary registry of available models: `Linear Regression`, `SVR`, `XGBoost`.

#### Helper Functions

**`train_classical_model(df, target_pollutant, model_wrapper, model_name, save_path)`**
*   **Purpose**: Orchestrates the training process.
*   **Steps**:
    1.  Splits data (90% train, 10% validation).
    2.  Trains the model wrapper.
    3.  Calculates Mean Absolute Error (MAE).
    4.  Saves a dictionary containing the model object, MAE, and training metadata to disk using `joblib`.

**`dynamic_predict(model_wrapper, last_known_data, future_dates, target_pollutant)`**
*   **Purpose**: Performs recursive multi-step forecasting.
*   **Logic**:
    *   Predicts day T+1.
    *   Updates the feature set for day T+2 by treating the T+1 prediction as a "lagged" actual value.
    *   Recalculates rolling means/stds for the new window.
    *   Repeats for T+3... T+7.

---

### 2.3 Deep Learning Models (`src/dl_models.py`)

Handles Neural Network architectures using Keras/TensorFlow.

#### Classes

**`ANNModel`**
*   **Description**: A standard Feed-Forward Artificial Neural Network.
*   **Architecture**:
    *   Input Layer -> Dense(64, relu) -> Dense(32, relu) -> Output(1)
*   **Preprocessing**: built-in `SimpleImputer` (median) and `MinMaxScaler`.
*   **`train(...)`**: Fits scaler on training data, then trains the network.
*   **`predict(...)`**: Imputes/Scales input -> Predicts -> Inverse Scales output.

**`LSTMModel`**
*   **Inheritance**: Inherits from `ANNModel`.
*   **Description**: Long Short-Term Memory network for sequence data.
*   **Architecture**:
    *   LSTM(50, relu) -> Dense(25, relu) -> Output(1)
*   **Special Logic**: Reshapes input data into 3D array `(samples, time_steps, features)` required by LSTM layers.

#### Global Variables
*   **`DL_MODELS`**: Dictionary registry: `ANN`, `LSTM`.

---

### 2.4 Visualization Modules (`src/eda.py` & `src/visualize.py`)

Helper functions to generate Plotly figures for the dashboard.

*   **`plot_trends`**: Creates multi-line time-series charts. Used in EDA tab.
*   **`plot_distribution`**: Creates yearly box plots to visualize statistical spread.
*   **`plot_missing_values`**: Visualizes data gaps.
*   **`src/visualize.py`**: Contains `plot_monthly_heatmap` using Seaborn (currently secondary/unused in main UI).

---

## 3. Dashboard Application (`dashboard/app.py`)

The Streamlit entry point.

### Key Logic Flow

1.  **Initialization**: Sets page config, constants, and creates `models/` directory if missing.
2.  **`load_data(path)`**: Cached function to load CSVs efficiently.
3.  **Tab Structure**:
    *   **Tab 1 (EDA)**:
        *   Filters data by City.
        *   Calls functions from `src.eda` based on user selection.
    *   **Tab 2 (Forecast)**:
        *   **Inputs**: User selects Target Pollutant (e.g., PM2.5) and Algorithm.
        *   **Training Trigger**: User clicks "Train/Retrain". The app routes the request to either `train_classical_model` or `train_dl_model` based on the algorithm type.
        *   **Inference**: Loads the saved `.joblib` model. Displays metadata (Training timestamp, MAE).
        *   **Forecasting**: Uses `dynamic_predict` to generate the next 7 days of values.
        *   **Health Advisory**: Custom logic (`get_aqi_alert`) maps predicted PM2.5 values to health categories (Good, Moderate, Poor, etc.).

---

## 4. Extending the Project

### Adding a New Model
1.  **Define the Class**: Create a new class in `src/classical_models.py` or new file. It must implement `train` and `predict` methods.
2.  **Register it**: Add the instance to the `CLASSICAL_MODELS` dictionary.
3.  **Update UI**: The `dashboard/app.py` automatically picks up keys from the dictionary, so no UI changes are needed for classical models.

### Adding a New City
1.  **Data Requirement**: You need historical weather data for the new city.
2.  **Update Pipeline**: Modify `src/data_pipeline.py` to:
    *   Load the new weather CSV.
    *   Merge it when `df['City'] == 'NewCity'`.
3.  **Rerun Pipeline**: Execute `python src/data_pipeline.py`.
