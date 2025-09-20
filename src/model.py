import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_absolute_error

def train_model(df, save_path=os.path.join("models", "xgboost_pm25_model.joblib")):
    """
    Trains an XGBoost model, calculates its MAE on a validation set,
    and saves both the model and the MAE to a file.
    
    Args:
        df (pd.DataFrame): The feature-engineered dataframe.
        save_path (str): Path to save the trained model and metrics.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    TARGET = 'PM2.5'
    FEATURES = [col for col in df.columns if col not in [TARGET, 'Date', 'City', 'AQI_Bucket']]
    
    X = df[FEATURES]
    y = df[TARGET]

    # Split data for training and validation
    split_index = int(len(X) * 0.9)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

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
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # --- NEW: Calculate the MAE on the validation set ---
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    print(f"Model validation MAE: {mae:.2f}")

    # --- NEW: Save the model and the MAE in a dictionary ---
    model_data = {
        'model': model,
        'mae': mae
    }
    
    joblib.dump(model_data, save_path)
    print(f"Model and MAE saved to {save_path}")

def predict(model, last_known_data, future_dates):
    """
    Makes multi-step predictions for future dates.
    (This function does not need to be changed)
    """
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
        
        for i in range(7, 1, -1):
            current_features[f'{target}_lag_{i}'] = current_features[f'{target}_lag_{i-1}']
        current_features[f'{target}_lag_1'] = pred
        
        previous_lags = [current_features[f'{target}_lag_{i}'].iloc[0] for i in range(1, 8)]
        current_features[f'{target}_roll_mean_7'] = np.mean(previous_lags)
        current_features[f'{target}_roll_std_7'] = np.std(previous_lags)

    return predictions