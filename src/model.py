import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

def train_model(df, save_path=os.path.join("models", "xgboost_pm25_model.joblib")):
    """
    Trains an XGBoost model on the feature-engineered data and saves it.
    
    Args:
        df (pd.DataFrame): The feature-engineered dataframe.
        save_path (str): Path to save the trained model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    TARGET = 'PM2.5'
    FEATURES = [col for col in df.columns if col not in [TARGET, 'Date', 'City', 'AQI_Bucket']]
    
    X = df[FEATURES]
    y = df[TARGET]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )
    
    split_index = int(len(X) * 0.9)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    joblib.dump(model, save_path)
    print(f"Model trained and saved to {save_path}")

def predict(model, last_known_data, future_dates):
    """
    Makes multi-step predictions for future dates.

    Args:
        model: The trained XGBoost model.
        last_known_data (pd.DataFrame): A dataframe with the last row of known data.
        future_dates (pd.DatetimeIndex): The dates to forecast.

    Returns:
        list: A list of predicted PM2.5 values.
    """
    # --- FIX: Define the target variable to ensure case consistency ---
    target = 'PM2.5'
    
    predictions = []
    current_features = last_known_data.copy()

    for date in future_dates:
        current_features['day_of_week'] = date.dayofweek
        current_features['month'] = date.month
        current_features['year'] = date.year
        current_features['day_of_year'] = date.dayofyear
        
        features_for_pred = [col for col in model.feature_names_in_ if col in current_features.columns]
        
        pred = model.predict(current_features[features_for_pred])[0]
        predictions.append(pred)
        
        # --- FIX: Use the 'target' variable in f-strings ---
        # Shift lags
        for i in range(7, 1, -1):
            current_features[f'{target}_lag_{i}'] = current_features[f'{target}_lag_{i-1}']
        current_features[f'{target}_lag_1'] = pred
        
        # Recalculate rolling features
        previous_lags = [current_features[f'{target}_lag_{i}'].iloc[0] for i in range(1, 8)]
        current_features[f'{target}_roll_mean_7'] = np.mean(previous_lags)
        current_features[f'{target}_roll_std_7'] = np.std(previous_lags)

    return predictions