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

from backend.app.services.prompts import HEALTH_ANALYSIS_SYSTEM_PROMPT
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
    current_aqi: int
    predicted_aqi: int
    dominant_pollutant: str

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

@app.get("/api/v1/system/status")
async def get_system_status():
    """
    Returns the status (existence, modification time) of critical data files.
    """
    def get_file_info(path):
        if os.path.exists(path):
            return {
                "exists": True,
                "modified": datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S'),
                "size_kb": round(os.path.getsize(path) / 1024, 2)
            }
        return {"exists": False, "modified": None, "size_kb": 0}

    return {
        "raw_data": {
            "city_day.csv": get_file_info(os.path.join(RAW_DATA_DIR, "city_day.csv")),
            "ahmedabad_weather.csv": get_file_info(os.path.join(RAW_DATA_DIR, "ahmedabad_weather.csv"))
        },
        "processed_data": {
            "gujarat_aqi.csv": get_file_info(os.path.join(DATA_DIR, "gujarat_aqi.csv")),
            "gujarat_features_for_model.csv": get_file_info(os.path.join(DATA_DIR, "gujarat_features_for_model.csv"))
        }
    }

def run_script_task(script_name: str):
    try:
        # Assuming script is in root/scripts/
        script_path = os.path.join("scripts", script_name)
        if not os.path.exists(script_path):
            print(f"Script not found: {script_path}")
            return
        
        print(f"Executing {script_name}...")
        # running in the same python environment
        subprocess.run(["python", script_path], check=True)
        print(f"Finished executing {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

@app.post("/api/v1/data/fetch-weather")
async def fetch_weather(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_script_task, "fetch_weather_data.py")
    return {"status": "started", "message": "Weather data fetch started in background"}

@app.get("/api/v1/history/{city}")
async def get_history(city: str):
    try:
        file_path = os.path.join(DATA_DIR, "gujarat_aqi.csv")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Data file not found")
        
        df = pd.read_csv(file_path)
        # Filter by city (assuming case-insensitive match for robustness)
        df_city = df[df['City'].str.lower() == city.lower()]
        
        if df_city.empty:
             raise HTTPException(status_code=404, detail=f"No data found for city: {city}")

        # Convert to list of dicts for JSON response
        # Returning last 30 days to avoid huge payload
        # Replace NaN with None for valid JSON serialization
        data = df_city.tail(30).replace({np.nan: None}).to_dict(orient="records")
        return {"city": city, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

        feature_file = os.path.join(DATA_DIR, "gujarat_features_for_model.csv")
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
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training_pipeline, request.model_type, request.city, request.target)
    return {"status": "training_started", "model": request.model_type, "city": request.city, "target": request.target}

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
        feature_file = os.path.join(DATA_DIR, "gujarat_features_for_model.csv")
        if not os.path.exists(feature_file):
             raise HTTPException(status_code=500, detail="Feature data file unavailable")
             
        df_full = pd.read_csv(feature_file, parse_dates=['Date'])
        last_known_data = df_full.iloc[[-1]].copy() # Last row as DataFrame
        
        last_date = last_known_data['Date'].iloc[0]
        future_dates = [last_date + timedelta(days=i) for i in range(1, request.days + 1)]
        
        predictions = dynamic_predict(model_wrapper, last_known_data, future_dates, request.target)
        
        forecast = []
        for date, pred in zip(future_dates, predictions):
            forecast.append({"date": date.strftime('%Y-%m-%d'), "value": round(pred, 2)})
            
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
                         "predicted": round(recent_preds[idx], 2)
                     })
        except Exception as e:
            print(f"Warning: Could not calculate recent performance: {e}")

        return {
            "city": request.city, 
            "model": request.model_type, 
            "target": request.target,
            "forecast": forecast,
            "metadata": {
                "mae": mae,
                "last_trained": train_timestamp
            },
            "recent_performance": recent_performance
        }
        
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
