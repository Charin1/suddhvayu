import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import joblib
import os
from datetime import datetime

# ... (The BaseModel and XGBoostModel classes are unchanged) ...
class BaseModel:
    def __init__(self, model):
        self.pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('regressor', model)])
        self.feature_names = []
    def train(self, X_train, y_train, X_val=None, y_val=None): self.pipeline.fit(X_train, y_train)
    def predict(self, X): return self.pipeline.predict(X)
    def get_feature_names(self): return self.feature_names

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror', n_jobs=-1, random_state=42, early_stopping_rounds=50)
        self.feature_names = []
    def train(self, X_train, y_train, X_val=None, y_val=None): self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    def predict(self, X): return self.model.predict(X)
    def get_feature_names(self): return self.feature_names

CLASSICAL_MODELS = {
    "Linear Regression": BaseModel(LinearRegression()),
    "SVR": BaseModel(SVR()),
    "XGBoost": XGBoostModel()
}

# --- FIX: Renamed function to be specific ---
def train_classical_model(df, target_pollutant, model_wrapper, model_name, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    FEATURES = [col for col in df.columns if col not in [target_pollutant, 'Date', 'City', 'AQI_Bucket']]
    X = df[FEATURES]
    y = df[target_pollutant]
    split_index = int(len(X) * 0.9)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    
    model_wrapper.feature_names = FEATURES
    print(f"Training {model_name} for {target_pollutant}...")
    model_wrapper.train(X_train, y_train, X_val, y_val)
    
    predictions = model_wrapper.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    print(f"Model validation MAE: {mae:.2f}")

    model_data = {
        'model_wrapper': model_wrapper, 'mae': mae,
        'train_start_date': df['Date'].min().strftime('%Y-%m-%d'),
        'train_end_date': df['Date'].max().strftime('%Y-%m-%d'),
        'train_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    joblib.dump(model_data, save_path)
    print(f"Model and metadata saved to {save_path}")

# ... (dynamic_predict function is unchanged) ...
def dynamic_predict(model_wrapper, last_known_data, future_dates, target_pollutant):
    predictions = []
    current_features = last_known_data.copy()
    feature_names = model_wrapper.get_feature_names()
    for date in future_dates:
        current_features['day_of_week'] = date.dayofweek
        current_features['month'] = date.month
        current_features['year'] = date.year
        current_features['day_of_year'] = date.dayofyear
        features_for_pred = [col for col in feature_names if col in current_features.columns]
        pred = model_wrapper.predict(current_features[features_for_pred])[0]
        predictions.append(pred)
        for i in range(7, 1, -1):
            current_features[f'{target_pollutant}_lag_{i}'] = current_features[f'{target_pollutant}_lag_{i-1}']
        current_features[f'{target_pollutant}_lag_1'] = pred
        previous_lags = [current_features[f'{target_pollutant}_lag_{i}'].iloc[0] for i in range(1, 8)]
        current_features[f'{target_pollutant}_roll_mean_7'] = np.mean(previous_lags)
        current_features[f'{target_pollutant}_roll_std_7'] = np.std(previous_lags)
    return predictions