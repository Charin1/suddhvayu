import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_absolute_error
from datetime import datetime

def train_model(df, target_pollutant, save_path):
    """
    Trains an XGBoost model for a specific target pollutant and saves it with metadata.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Features are all columns that are NOT the target or identifiers
    FEATURES = [col for col in df.columns if col not in [target_pollutant, 'Date', 'City', 'AQI_Bucket']]
    
    X = df[FEATURES]
    y = df[target_pollutant]

    split_index = int(len(X) * 0.9)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, objective='reg:squarederror', n_jobs=-1,
        random_state=42, early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    print(f"Model for {target_pollutant} validation MAE: {mae:.2f}")

    model_data = {
        'model': model, 'mae': mae,
        'train_start_date': df['Date'].min().strftime('%Y-%m-%d'),
        'train_end_date': df['Date'].max().strftime('%Y-%m-%d'),
        'train_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    joblib.dump(model_data, save_path)
    print(f"Model and metadata for {target_pollutant} saved to {save_path}")

def predict(model, last_known_data, future_dates, target_pollutant):
    """
    Makes multi-step predictions for a specific target pollutant.
    """
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
        
        # Update lag and rolling features for the specific target
        for i in range(7, 1, -1):
            current_features[f'{target_pollutant}_lag_{i}'] = current_features[f'{target_pollutant}_lag_{i-1}']
        current_features[f'{target_pollutant}_lag_1'] = pred
        previous_lags = [current_features[f'{target_pollutant}_lag_{i}'].iloc[0] for i in range(1, 8)]
        current_features[f'{target_pollutant}_roll_mean_7'] = np.mean(previous_lags)
        current_features[f'{target_pollutant}_roll_std_7'] = np.std(previous_lags)
    return predictions
