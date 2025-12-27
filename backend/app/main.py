from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uvicorn
from groq import Groq
from dotenv import load_dotenv
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import subprocess
import sys
from backend.app.services.data_pipeline import process_data as run_data_pipeline

# ... existing code ...

from backend.app.services.prompts import HEALTH_ANALYSIS_SYSTEM_PROMPT, GOVT_POLICY_SYSTEM_PROMPT
from backend.app.services.classical_models import CLASSICAL_MODELS, train_classical_model, dynamic_predict
from backend.app.services.dl_models import DL_MODELS, train_dl_model

load_dotenv()

# Define Pydantic models for request/response
class TrainRequest(BaseModel):
    model_type: str = "xgboost"
    city: str = "Ahmedabad"
    target: str = "AQI"  # Added target field

class PredictRequest(BaseModel):
    model_type: str="xgboost"
    city: str = "Ahmedabad"
    days: int = 7
    target: str = "AQI" # Added target field

class HealthAnalysisRequest(BaseModel):
    current_aqi: float
    predicted_aqi: float
    dominant_pollutant: str

class PolicyAnalysisRequest(BaseModel):
    city: str
    current_aqi: float
    dominant_pollutant: str

class FetchWeatherRequest(BaseModel):
    city: str
    start_date: str
    end_date: str

class ProcessFeaturesRequest(BaseModel):
    city: str = None

# Initialize FastAPI app
app = FastAPI(
    title="ShuddhVayu API",
    description="Backend API for Air Quality Prediction and Health Analysis",
    version="1.0.0"
)

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Paths
DATA_DIR = os.path.join("data", "processed")
RAW_DATA_DIR = os.path.join("data", "raw")
MODELS_DIR = os.path.join("models")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Service is healthy"}


@app.get("/api/v1/history/{city}")
async def get_history(city: str):
    try:
        # Try city-specific file first (new format)
        city_lower = city.lower()
        specific_file = os.path.join(DATA_DIR, f"{city_lower}_aqi_cleaned.csv")
        legacy_file = os.path.join(DATA_DIR, "gujarat_aqi.csv")
        
        df = None
        if os.path.exists(specific_file):
            df = pd.read_csv(specific_file)
            # Ensure filtering if multiple cities somehow exist
            if 'City' in df.columns:
                df = df[df['City'].str.lower() == city_lower]
        elif os.path.exists(legacy_file):
            df_all = pd.read_csv(legacy_file)
            df = df_all[df_all['City'].str.lower() == city_lower]
        
        if df is None or df.empty:
             # Just return empty list instead of 404/500 to allow UI to render empty state
             return {"city": city, "data": []}

        # Convert to list of dicts for JSON response
        # Returning last 30 days to avoid huge payload
        # Replace NaN with None for valid JSON serialization
        data = df.tail(30).replace({np.nan: None}).to_dict(orient="records")
        return {"city": city, "data": data}
    except Exception as e:
        print(f"Error fetching history for {city}: {e}")
        # Return empty data structure on error preventing frontend crash
        return {"city": city, "data": []}

@app.get("/api/v1/distribution/{city}")
async def get_distribution(city: str, pollutant: str = "AQI"):
    """
    Get distribution data for a specific pollutant in a city (for Histogram)
    """
    try:
        file_path = os.path.join(DATA_DIR, "gujarat_aqi.csv")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Data file not found")
            
        df = pd.read_csv(file_path)
        df_city = df[df['City'].str.lower() == city.lower()]
        
        if df_city.empty:
             raise HTTPException(status_code=404, detail=f"No data found for city: {city}")
             
        if pollutant not in df_city.columns:
             raise HTTPException(status_code=400, detail=f"Pollutant {pollutant} not found in data")

        # Get data for histogram
        values = df_city[pollutant].dropna().tolist()
        return {"city": city, "pollutant": pollutant, "values": values}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_training_pipeline(model_type: str, city: str, target: str):
    # This runs in background
    try:
        model_type_key = None
        # Map input string to dictionary keys
        for key in CLASSICAL_MODELS.keys():
            if key.lower().replace(" ", "") == model_type.lower().replace(" ", ""):
                model_type_key = key
                break
        
        is_dl = False
        if not model_type_key:
             for key in DL_MODELS.keys():
                if key.lower() == model_type.lower():
                    model_type_key = key
                    is_dl = True
                    break
        
        if not model_type_key:
            print(f"Error: Invalid model type {model_type}")
            return

        feature_file = os.path.join(DATA_DIR, f"{city.lower()}_features.csv")
        if not os.path.exists(feature_file):
            print(f"Error: Feature file not found at {feature_file}")
            return
            
        df = pd.read_csv(feature_file, parse_dates=['Date'])
        # Filter for city if needed, but current feature file might be specific to Ahmedabad essentially
        # The data_pipeline.py focuses on Ahmedabad for features currently.
        
        save_path = os.path.join(MODELS_DIR, f"{model_type_key.lower().replace(' ', '_')}_{city.lower()}_{target}.pkl")
        
        if is_dl:
             train_dl_model(df, target, DL_MODELS[model_type_key], model_type_key, save_path)
        else:
             train_classical_model(df, target, CLASSICAL_MODELS[model_type_key], model_type_key, save_path)
             
    except Exception as e:
        print(f"Training failed: {str(e)}")

@app.post("/api/v1/train")
async def train_model(request: TrainRequest):
    try:
        # Run synchronously to ensure model is ready before prediction
        print(f"DEBUG: Starting synchronous training for {request.model_type} on {request.city}")
        run_training_pipeline(request.model_type, request.city, request.target)
        return {"status": "training_completed", "model": request.model_type, "city": request.city, "target": request.target}
    except Exception as e:
        print(f"ERROR: Training endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict")
async def predict_aqi(request: PredictRequest):
    try:
        # Resolve model filename
        model_name_clean = request.model_type.lower().replace(" ", "_")
        # Use simple lowercase for filename construction to match training
        model_file = f"{model_name_clean}_{request.city.lower()}_{request.target}.pkl"
        model_path = os.path.join(MODELS_DIR, model_file)
        
        if not os.path.exists(model_path):
             raise HTTPException(status_code=404, detail=f"Model not found. Please train {request.model_type} for target {request.target} first.")
             
        model_data = joblib.load(model_path)
        # Check integrity
        if not isinstance(model_data, dict) or 'model_wrapper' not in model_data:
             # Legacy or corrupt support
             raise HTTPException(status_code=500, detail="Invalid model file format. Please retrain.")

        model_wrapper = model_data['model_wrapper']
        mae = model_data.get('mae')
        train_timestamp = model_data.get('train_timestamp')
        
        # Load latest data for prediction context
        feature_file = os.path.join(DATA_DIR, f"{request.city.lower()}_features.csv")
        if not os.path.exists(feature_file):
             raise HTTPException(status_code=500, detail="Feature data file unavailable")
             
        df_full = pd.read_csv(feature_file, parse_dates=['Date'])
        last_known_data = df_full.iloc[[-1]].copy() # Last row as DataFrame
        
        last_date = last_known_data['Date'].iloc[0]
        future_dates = [last_date + timedelta(days=i) for i in range(1, request.days + 1)]
        
        predictions = dynamic_predict(model_wrapper, last_known_data, future_dates, request.target)
        
        forecast = []
        for date, pred in zip(future_dates, predictions):
            forecast.append({"date": date.strftime('%Y-%m-%d'), "value": float(round(pred, 2))})
            
        # Also return recent performance if available (last 14 days)
        recent_performance = []
        try:
             df_recent = df_full.tail(14).copy()
             features = model_wrapper.get_feature_names()
             # Only predict if we have all features
             valid_features = [f for f in features if f in df_recent.columns]
             if len(valid_features) == len(features):
                 recent_preds = model_wrapper.predict(df_recent[features])
                 for idx, row in df_recent.reset_index().iterrows():
                     recent_performance.append({
                         "date": row['Date'].strftime('%Y-%m-%d'),
                         "actual": row.get(request.target),
                         "predicted": float(round(recent_preds[idx], 2))
                      })
        except Exception as e:
            print(f"Warning: Could not calculate recent performance: {e}")

        return {
            "city": request.city, 
            "model": request.model_type, 
            "target": request.target,
            "forecast": forecast,
            "metadata": {
                "mae": float(mae) if mae is not None else None,
                "last_trained": train_timestamp
            },
            "recent_performance": recent_performance
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze-health")
async def analyze_health(request: HealthAnalysisRequest):
    try:
        prompt = HEALTH_ANALYSIS_SYSTEM_PROMPT.format(
            current_aqi=request.current_aqi,
            predicted_aqi=request.predicted_aqi,
            dominant_pollutant=request.dominant_pollutant
        )
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
        )
        
        result = chat_completion.choices[0].message.content
        return json.loads(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze-policy")
async def analyze_policy(request: PolicyAnalysisRequest):
    try:
        prompt = GOVT_POLICY_SYSTEM_PROMPT.format(
            city=request.city,
            current_aqi=request.current_aqi,
            dominant_pollutant=request.dominant_pollutant
        )
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
        )
        
        result = chat_completion.choices[0].message.content
        return json.loads(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data/fetch-weather")
async def fetch_weather_endpoint(request: FetchWeatherRequest, background_tasks: BackgroundTasks):
    """
    Triggers the fetch_weather_data.py script in the background with dynamic arguments.
    """
    print(f"DEBUG: Received fetch request for city={request.city}, start={request.start_date}, end={request.end_date}")
    
    def run_script(city, start, end):
        script_path = os.path.join("scripts", "fetch_weather_data.py")
        abs_script_path = os.path.abspath(script_path)
        venv_python = sys.executable
        
        print(f"DEBUG: Starting background task for {city}")
        print(f"DEBUG: Using Python: {venv_python}")
        print(f"DEBUG: Script Path: {abs_script_path}")
        
        if not os.path.exists(abs_script_path):
             print(f"ERROR: Script not found at {abs_script_path}")
             return

        cmd = [
            venv_python, abs_script_path,
            "--city", city,
            "--start_date", start,
            "--end_date", end
        ]
        
        print(f"DEBUG: Executing command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(f"DEBUG: Subprocess return code: {result.returncode}")
            print("DEBUG: STDOUT:", result.stdout)
            print("DEBUG: STDERR:", result.stderr)
        except Exception as e:
            print(f"ERROR: Subprocess failed with exception: {e}")

    background_tasks.add_task(run_script, request.city, request.start_date, request.end_date)
    return {"status": "started", "message": f"Weather data fetch for {request.city} initiated in background."}

@app.post("/api/v1/data/process-features")
async def process_features_endpoint(request: ProcessFeaturesRequest, background_tasks: BackgroundTasks):
    """
    Triggers the feature engineering pipeline (data_pipeline.py).
    """
    def task_wrapper(city):
        print(f"Starting feature processing for {city}...")
        try:
             run_data_pipeline(city=city)
             print(f"Feature processing for {city} completed.")
        except Exception as e:
             print(f"Feature processing failed: {e}")

    background_tasks.add_task(task_wrapper, request.city)
    return {"status": "started", "message": f"Feature engineering for {request.city or 'ALL'} initiated in background."}

@app.get("/api/v1/system/status")
async def get_system_status():
    status = []
    
    # helper to process a directory
    def scan_dir(directory, category):
        if not os.path.exists(directory):
            return
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                path = os.path.join(directory, filename)
                info = {
                    "key": filename,
                    "category": category, # "raw" or "processed"
                    "exists": True,
                    "path": path,
                    "size_bytes": os.path.getsize(path),
                    "last_modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
                }
                status.append(info)

    scan_dir(RAW_DATA_DIR, "raw")
    scan_dir(DATA_DIR, "processed")
        
    return status

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
