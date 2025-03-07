from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import numpy as np
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="Air Quality API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data model for air quality inputs
class AirQualityInput(BaseModel):
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    o3: float
    temperature: float
    humidity: float
    wind_speed: float

# Define the data model for weather inputs
class WeatherInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float

# Define the data model for prediction results
class AirQualityPrediction(BaseModel):
    predicted_aqi: float
    category: str
    color: str
    description: str

class WeatherPrediction(BaseModel):
    predicted_temperature: float
    predicted_humidity: float
    predicted_wind_speed: float

# Define the data model for historical data
class HistoricalData(BaseModel):
    dates: List[str]
    aqi_values: List[float]
    pm25_values: List[float]
    pm10_values: List[float]

# Define the data model for model performance
class ModelPerformance(BaseModel):
    mse: float
    rmse: float
    mae: float
    r2: float
    feature_importance: Dict[str, float]

# # Try to load the model
# MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model.joblib")
# logger.info(f"🔍 Checking model path: {MODEL_PATH}")

# try:
#     model = joblib.load(MODEL_PATH)
#     logger.info("✅ Model loaded successfully!")
#     model_loaded = True
# except FileNotFoundError:
#     logger.warning("⚠️ Model file not found! Using dummy predictions.")
#     model_loaded = False
# except Exception as e:
#     logger.error(f"❌ Error loading model: {e}")
#     model_loaded = False

# Define the Google Drive file ID and the path to save the model
MODEL_FILE_ID = "1abcdEFgHijkLMNOPqrsTUvwxyz"  # Replace this with your Google Drive file ID
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model.joblib")

logger.info(f"🔍 Checking model path: {MODEL_PATH}")

# Try to load the model
try:
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully!")
    model_loaded = True
except FileNotFoundError:
    logger.warning("⚠️ Model file not found! Downloading from Google Drive...")
    try:
        # Download the model from Google Drive if it doesn't exist locally
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        download_model_from_google_drive(MODEL_FILE_ID, MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Model downloaded and loaded successfully!")
        model_loaded = True
    except Exception as e:
        logger.error(f"❌ Error downloading or loading model: {e}")
        model_loaded = False
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model_loaded = False


def get_aqi_category(aqi_value: float) -> tuple:
    """Get AQI category, color, and description based on AQI value."""
    if aqi_value <= 50:
        return "Good", "green", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi_value <= 100:
        return "Moderate", "yellow", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "orange", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi_value <= 200:
        return "Unhealthy", "red", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi_value <= 300:
        return "Very Unhealthy", "purple", "Health alert: The risk of health effects is increased for everyone."
    else:
        return "Hazardous", "maroon", "Health warning of emergency conditions: everyone is more likely to be affected."

@app.get("/")
def read_root():
    return {"message": "Welcome to the Air Quality API"}

@app.post("/predict/", response_model=AirQualityPrediction)
def predict_aqi(data: AirQualityInput):
    """Predict AQI based on input parameters."""
    if model_loaded:
        try:
            # Prepare input data for the model (order matters)
            input_data = np.array([
                [data.pm25, data.pm10, data.no2, data.so2, data.co, 
                data.o3, data.temperature, data.humidity, data.wind_speed]
            ])
            
            # Make prediction
            aqi_value = float(model.predict(input_data)[0])
            
            # Get category information
            category, color, description = get_aqi_category(aqi_value)
            
            return {
                "predicted_aqi": aqi_value,
                "category": category,
                "color": color,
                "description": description
            }
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    else:
        # Generate dummy prediction if model is not loaded
        base_aqi = 30
        
        # Add weighted contributions from each pollutant
        aqi = base_aqi + (data.pm25 * 0.5) + (data.pm10 * 0.3) + (data.no2 * 0.4) + \
              (data.so2 * 0.3) + (data.co * 20) + (data.o3 * 0.4)
        
        # Apply some randomness
        aqi = max(10, min(450, aqi + np.random.normal(0, 5)))
        
        # Get category info
        category, color, description = get_aqi_category(aqi)
        
        return {
            "predicted_aqi": round(aqi, 1),
            "category": category,
            "color": color,
            "description": description
        }

@app.post("/predict-weather/", response_model=WeatherPrediction)
def predict_weather(data: WeatherInput):
    """Predict weather parameters (temperature, humidity, and wind speed) based on input."""
    # For simplicity, using dummy models
    predicted_temperature = data.temperature + np.random.normal(0, 2)
    predicted_humidity = data.humidity + np.random.normal(0, 5)
    predicted_wind_speed = data.wind_speed + np.random.normal(0, 1)

    # Ensure predictions are within reasonable bounds
    predicted_temperature = max(-20, min(50, predicted_temperature))
    predicted_humidity = max(0, min(100, predicted_humidity))
    predicted_wind_speed = max(0, min(30, predicted_wind_speed))

    return {
        "predicted_temperature": round(predicted_temperature, 2),
        "predicted_humidity": round(predicted_humidity, 2),
        "predicted_wind_speed": round(predicted_wind_speed, 2)
    }

@app.get("/historical-data/", response_model=HistoricalData)
def get_historical_data():
    """Get historical AQI data."""
    # Generate dummy historical data (you would replace this with a database query)
    dates = [f"2023-01-{i+1:02d}" for i in range(14)]
    
    # Create a base pattern with some weekly cyclicity
    base_pattern = np.array([80, 85, 95, 100, 90, 70, 60])  # Higher on weekdays, lower on weekends
    
    # Repeat the pattern for 2 weeks and add some noise
    base_values = np.tile(base_pattern, 2)
    aqi_values = base_values + np.random.normal(0, 15, size=14)
    aqi_values = np.clip(aqi_values, 20, 250)  # Clip to realistic AQI range
    
    # Generate PM2.5 and PM10 values
    pm25_values = aqi_values * 0.4 + np.random.normal(0, 5, size=14)
    pm25_values = np.clip(pm25_values, 5, 100)
    
    pm10_values = aqi_values * 0.7 + np.random.normal(0, 10, size=14)
    pm10_values = np.clip(pm10_values, 10, 150)
    
    return {
        "dates": dates,
        "aqi_values": aqi_values.tolist(),
        "pm25_values": pm25_values.tolist(),
        "pm10_values": pm10_values.tolist(),
    }

@app.get("/model-performance/", response_model=ModelPerformance)
def get_model_performance():
    """Get model performance metrics."""
    # In a real app, these would come from model evaluation
    return {
        "mse": 15.23,
        "rmse": 3.90,
        "mae": 3.12,
        "r2": 0.85,
        "feature_importance": {
            "PM2.5": 0.35,
            "PM10": 0.20,
            "NO2": 0.15,
            "SO2": 0.10,
            "CO": 0.08,
            "O3": 0.07,
            "Temperature": 0.02,
            "Humidity": 0.02,
            "Wind_speed": 0.01
        }
    }

# For direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
