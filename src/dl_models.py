import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error
import joblib
import os
from datetime import datetime

# --- FIX: Import the SimpleImputer ---
from sklearn.impute import SimpleImputer

class ANNModel:
    def __init__(self):
        self.model = None
        # --- FIX: Add an imputer to the class ---
        self.imputer = SimpleImputer(strategy='median')
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_names = []

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # --- FIX: First, impute the data to remove NaNs ---
        X_train_imputed = self.imputer.fit_transform(X_train)
        # Then, scale the imputed data
        X_train_scaled = self.scaler_X.fit_transform(X_train_imputed)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_absolute_error')
        self.model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, verbose=0)

    def predict(self, X):
        # --- FIX: Apply the same imputation and scaling ---
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler_X.transform(X_imputed)
        
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred.flatten()

    def get_feature_names(self): return self.feature_names

class LSTMModel(ANNModel): # Inherits the imputer and scaling logic from ANNModel
    def _reshape_data(self, X):
        return np.reshape(X, (X.shape[0], 1, X.shape[1]))
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Impute and scale data
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_scaled = self.scaler_X.fit_transform(X_train_imputed)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        
        # Reshape for LSTM
        X_train_reshaped = self._reshape_data(X_train_scaled)
        
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_absolute_error')
        self.model.fit(X_train_reshaped, y_train_scaled, epochs=50, batch_size=32, verbose=0)

    def predict(self, X):
        # Impute and scale data
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler_X.transform(X_imputed)
        
        # Reshape for LSTM
        X_reshaped = self._reshape_data(X_scaled)
        
        y_pred_scaled = self.model.predict(X_reshaped, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred.flatten()

DL_MODELS = {
    "ANN": ANNModel(),
    "LSTM": LSTMModel()
}

def train_dl_model(df, target_pollutant, model_wrapper, model_name, save_path):
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