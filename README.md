# ShuddhVayu - Gujarat Air Quality Intelligence Platform

An end-to-end air quality analysis and forecasting platform for Gujarat, India. Features a React frontend with a FastAPI backend, powered by machine learning models for multi-pollutant forecasting.

---

## Features

### 📊 Exploratory Data Analysis (EDA)
- **City Selection:** Filter analysis for major Gujarat cities (Ahmedabad, Gandhinagar, Surat, etc.)
- **Trend Analysis:** Compare historical trends for PM2.5, PM10, NO2, SO2, CO, O3, and AQI
- **Pollutant Distribution:** Yearly box plots showing medians, ranges, and outliers
- **Data Quality Insights:** Missing value analysis for raw datasets
- **KPI Dashboard:** Real-time key performance indicators grid

### 🔮 Multi-Model Forecasting
- **Dynamic Pollutant Selection:** Forecast any major pollutant (PM2.5, PM10, AQI, etc.)
- **ML Algorithms:** Linear Regression, SVR, XGBoost, ANN, LSTM
- **7-Day Forecasts:** Interactive charts with confidence intervals
- **Model Dashboard:** Training status, performance metrics, model comparison
- **Forecast Viewer:** Interactive visualization of predictions

### 🤖 AI-Powered Analysis (Groq LLM)
- **Health Analysis:** Personalized health recommendations for:
  - Children, Seniors, Outdoor Athletes
  - Risk level assessment (Low/Moderate/High/Severe)
  - Based on current and predicted AQI
- **Policy Engine:** Government policy recommendations with:
  - Short-term (3 months), Medium-term (6 months), Long-term (12 months) roadmaps
  - Projected impact estimates
  - City-specific action plans

### 🔄 Data Operations
- **OpenAQ Integration:** Fetch real-time pollutant data via OpenAQ API v3
- **Open-Meteo Weather:** Historical weather data (temperature, humidity, wind)
- **Automated Pipeline:** Feature engineering with lag features, rolling statistics
- **Data Console:** Interactive data management interface
- **Pipeline Status:** Real-time monitoring of data processing
- **System Status:** Monitor available data files and trained models

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React, TypeScript, Tailwind CSS, Recharts |
| **Backend** | FastAPI, Python, Pandas, NumPy |
| **ML/AI** | Scikit-learn, XGBoost, TensorFlow/Keras |
| **LLM** | Groq API (Llama 3.3) |
| **Data Sources** | OpenAQ API v3, Open-Meteo Archive API |

---

## Project Structure

```
.
├── frontend/               # React application
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/          # Route pages
│   │   └── services/       # API clients
│   └── package.json
├── backend/                # FastAPI application
│   ├── app/
│   │   ├── main.py         # API routes
│   │   └── services/
│   │       ├── data_pipeline.py    # ETL & feature engineering
│   │       ├── classical_models.py # ML model training
│   │       └── dl_models.py        # Neural network models
│   └── requirements.txt
├── data/
│   ├── raw/                # Raw CSV files (AQI, weather)
│   └── processed/          # Cleaned & feature-engineered data
├── models/                 # Saved trained models (.pkl, .joblib)
├── scripts/
│   └── fetch_weather_data.py   # Data fetching utility
├── documentation/
│   └── PROJECT_MANUAL.md   # Technical documentation
└── README.md
```

---

## Setup and Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- OpenAQ API Key (free from [openaq.org](https://explore.openaq.org/register))

### 1. Clone and Setup Environment

```bash
git clone <your-repo-url>
cd suddhvayu

# Backend setup
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

### 2. Configure Environment Variables

Create `.env` in the project root:
```bash
OPENAQ_API_KEY=your_openaq_api_key_here
```

### 3. Fetch Initial Data

```bash
# Fetch weather and AQI data for Ahmedabad
python scripts/fetch_weather_data.py --city Ahmedabad --start_date 2023-01-01 --end_date 2025-12-25
```

### 4. Run Data Pipeline

```bash
# From project root
cd backend
python -c "from app.services.data_pipeline import process_data; process_data('Ahmedabad')"
```

---

## Running the Application

### Development Mode

```bash
# Terminal 1: Backend
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

Access the app at: `http://localhost:5173`

---

## Data Pipeline Details

### AQI Calculation (CPCB Standards)

The pipeline calculates AQI using Central Pollution Control Board (CPCB) breakpoints:

| Pollutant | Good (0-50) | Satisfactory (51-100) | Moderate (101-200) |
|-----------|-------------|----------------------|-------------------|
| PM2.5 | 0-30 µg/m³ | 31-60 µg/m³ | 61-90 µg/m³ |
| PM10 | 0-50 µg/m³ | 51-100 µg/m³ | 101-250 µg/m³ |
| NO2 | 0-40 µg/m³ | 41-80 µg/m³ | 81-180 µg/m³ |
| CO | 0-1 mg/m³ | 1.1-2 mg/m³ | 2.1-10 mg/m³ |

### Data Quality Handling

- **Duplicate Sensors:** Automatically consolidated (PM2.5 + PM2.5.1 → mean)
- **Missing Values:** Linear interpolation with bidirectional fill
- **Unit Conversion:** CO auto-converted from µg/m³ to mg/m³ when needed

---

## API Endpoints

### Core Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/history/{city}` | GET | Historical pollutant trends |
| `/api/distribution/{city}` | GET | Pollutant distribution data |
| `/api/system-status` | GET | Available data files and models |

### Model Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/train` | POST | Train ML model (background task) |
| `/api/predict` | POST | Generate 7-day forecast |

### AI Analysis Endpoints (Groq LLM)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze-health` | POST | Health recommendations by demographic |
| `/api/analyze-policy` | POST | Government policy roadmap |

### Data Operations Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/fetch-weather` | POST | Fetch external data (background) |
| `/api/process-features` | POST | Run data pipeline (background) |

---

## License

MIT License - See LICENSE file for details.
