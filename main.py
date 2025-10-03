# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict
import logging
import rasterio
from shapely.geometry import Point, Polygon as ShapelyPolygon, MultiPoint
import joblib
import json
import pandas as pd
from datetime import datetime
import xgboost as xgb
import numpy as np
import random 
# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KaalNetra API",
    description="Geospatial analysis for ecology prediction",
    version="3.0.0" # Production-ready feature engineering
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000","http://localhost:3001","http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Constants ---

# Define the exact feature order the ecology model expects
MODEL_FEATURE_ORDER = [
    'latitude', 'longitude', 'T2M', 'PRECTOTCORR', 'SWGDN', 'biome_cat_Af', 
    'biome_cat_Am', 'biome_cat_Aw', 'biome_cat_BSh', 'biome_cat_BSk', 
    'biome_cat_BWh', 'biome_cat_BWk', 'biome_cat_Cfa', 'biome_cat_Cfb', 
    'biome_cat_Cfc', 'biome_cat_Csa', 'biome_cat_Csb', 'biome_cat_Cwa', 
    'biome_cat_Cwb', 'biome_cat_Dfa', 'biome_cat_Dfc', 'biome_cat_Dfd', 
    'biome_cat_Dsa', 'biome_cat_Dsb', 'biome_cat_Dsc', 'biome_cat_Dsd', 
    'biome_cat_Dwa', 'biome_cat_Dwc', 'biome_cat_Dwd', 'biome_cat_ET', 
    'biome_cat_Unknown', 'doy_sin', 'doy_cos', 'temp_x_radiation', 'drought_index'
]

# Extract all possible biome codes from the feature list for one-hot encoding
ALL_BIOME_CODES = [code for col in MODEL_FEATURE_ORDER if (code := col.replace('biome_cat_', '')) != col]

# Standard Köppen Classification Dictionary
KOPPEN_CLASSES = {
    1: ("Af", "Tropical rainforest"), 2: ("Am", "Tropical monsoon"), 3: ("Aw", "Tropical savanna"),
    4: ("BWh", "Hot desert"), 5: ("BWk", "Cold desert"), 6: ("BSh", "Hot semi-arid"),
    7: ("BSk", "Cold semi-arid"), 8: ("Csa", "Hot-summer Mediterranean"), 9: ("Csb", "Warm-summer Mediterranean"),
    10: ("Cwa", "Monsoon-influenced humid subtropical"), 11: ("Cwb", "Subtropical highland"), 12: ("Cfa", "Humid subtropical"),
    13: ("Cfb", "Temperate oceanic"), 14: ("Cfc", "Subpolar oceanic"), 15: ("Dsa", "Hot-summer humid continental"),
    16: ("Dsb", "Warm-summer humid continental"), 17: ("Dsc", "Subarctic"), 18: ("Dsd", "Severely cold subarctic"),
    19: ("Dwa", "Monsoon-influenced hot-summer humid continental"), 20: ("Dwb", "Monsoon-influenced warm-summer humid continental"),
    21: ("Dwc", "Monsoon-influenced subarctic"), 22: ("Dwd", "Monsoon-influenced severely cold subarctic"),
    23: ("Dfa", "Hot-summer humid continental"), 24: ("Dfb", "Warm-summer humid continental"), 25: ("Dfc", "Subarctic"),
    26: ("Dfd", "Severely cold subarctic"), 27: ("ET", "Tundra"), 28: ("EF", "Ice cap")
}


# --- Load ML Models and Lookups on Startup ---
try:
    logger.info("Loading machine learning assets...")
    # Ecology Models
    combined_model = joblib.load('models/combined_model.joblib')
    combined_encoder = joblib.load('models/combined_encoder.joblib')

    # Climate Models
    temp_model = xgb.XGBRegressor()
    temp_model.load_model("models/temp_model.json")
    precip_model = xgb.XGBRegressor()
    precip_model.load_model("models/precip_model.json")
    rad_model = xgb.XGBRegressor()
    rad_model.load_model("models/rad_model.json")

    # JSON Lookups
    with open('models/scenario_to_plants_lookup.json') as f:
        scenario_to_plants = json.load(f)
    with open('models/name_lookups.json') as f:
        name_lookups = json.load(f)
    logger.info("All assets loaded successfully.")
except Exception as e:
    logger.error(f"FATAL: A required model file was not found or failed to load. Please check the 'models' directory. Details: {e}")
    combined_model = None # Prevent app from running if models are missing


# --- ML Prediction Functions ---

def predict_temperature_k(lat: float, lng: float, month: int, year: int) -> float:
    """Predicts temperature and returns it in Kelvin."""
    input_df = pd.DataFrame([[lat, lng, month, year]], columns=['lat', 'lon', 'month', 'year'])
    prediction_k = temp_model.predict(input_df)[0]
    return float(prediction_k)

def predict_precipitation_raw(lat: float, lng: float, month: int, year: int) -> float:
    """Predicts precipitation and returns it in the model's raw unit (kg m-2 s-1)."""
    input_df = pd.DataFrame([[lat, lng, month, year]], columns=['lat', 'lon', 'month', 'year'])
    prediction_raw = precip_model.predict(input_df)[0]
    return float(prediction_raw)

def predict_radiation(lat: float, lng: float, month: int, year: int) -> float:
    """Predicts radiation and returns it in W m-2."""
    input_df = pd.DataFrame([[lat, lng, month, year]], columns=['lat', 'lon', 'month', 'year'])
    prediction = rad_model.predict(input_df)[0]
    return round(float(prediction), 1)

def predict_ecology_from_data(
    biome_code: str, lat: float, lng: float, doy: int, 
    temp_k: float, precip_raw: float, rad: float
) -> Dict:
    """Performs full feature engineering and runs the ecology prediction."""
    if not combined_model:
        raise RuntimeError("ML models are not loaded. Cannot perform prediction.")

    # 1. Feature Engineering
    doy_sin = np.sin(2 * np.pi * doy / 365.25)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)
    temp_x_radiation = temp_k * rad
    drought_index = temp_k / (precip_raw + 1) 

    data = {
        'latitude': lat,
        'longitude': lng,
        'T2M': temp_k,
        'PRECTOTCORR': precip_raw,
        'SWGDN': rad,
        'doy_sin': doy_sin,
        'doy_cos': doy_cos,
        'temp_x_radiation': temp_x_radiation,
        'drought_index': drought_index,
    }

    for code in ALL_BIOME_CODES:
        data[f'biome_cat_{code}'] = 1 if code == biome_code else 0
    
    # 2. Create and Order DataFrame
    input_df = pd.DataFrame([data])
    input_df = input_df[MODEL_FEATURE_ORDER] # Enforce exact column order

    # 3. Run Prediction
    encoded_result = combined_model.predict(input_df)[0]
    scenario_text = combined_encoder.inverse_transform([encoded_result])[0]
    
    # 4. Format Output
    try:
        phenophase, pest_scientific_name = [part.strip() for part in scenario_text.split('|')]
    except ValueError:
        logger.error(f"Could not parse scenario text: {scenario_text}")
        return {"species": [], "pests": []}
    
    plant_scientific_names = scenario_to_plants.get(scenario_text, [])
    species_list = []
    
    # <-- CORRECTED: Safely access the nested 'plant_names' dictionary
    plant_name_dict = name_lookups.get('plant_names', {})
    for plant_name in plant_scientific_names:
        species_list.append({
            'scientific_name': plant_name,
            # <-- CORRECTED: Look up the stripped name in the nested dictionary
            'common_name': plant_name_dict.get(plant_name.strip(), "Unknown"),
            'phenophase': phenophase
        })
        
    # <-- CORRECTED: Safely access the nested 'pest_names' dictionary
    pest_name_dict = name_lookups.get('pest_names', {})
    pests_list = [{
        'scientific_name_pest': pest_scientific_name,
        # <-- CORRECTED: Look up the stripped name in the nested dictionary
        'common_name_pest': pest_name_dict.get(pest_scientific_name.strip(), "Unknown")
    }]
    return {"species": species_list, "pests": pests_list}


# --- Geospatial Helper Function ---

def get_koppen_biome(lat: float, lng: float, raster_file="koppen.tif") -> Tuple[int, str, str]:
    """Reads the Köppen-Geiger classification from a GeoTIFF file for a given point."""
    try:
        with rasterio.open(raster_file) as src:
            for val in src.sample([(lng, lat)]):
                class_id = int(val[0])
                code, name = KOPPEN_CLASSES.get(class_id, ("Unknown", "Unknown"))
                return class_id, code, name
    except Exception as e:
        logger.error(f"Could not read from raster file: {e}")
        return 0, "Error", "Error"


# --- API Models ---

class AnalysisRequest(BaseModel):
    shape_points: List[Tuple[float, float]] = Field(..., min_items=3)
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=1980, le=2099)


# --- API Endpoints ---

@app.get("/")
async def root():
    return {"status": "KaalNetra API is online"}

@app.post("/analyze-by-biome")
async def analyze_area_by_biome(request: AnalysisRequest):
    """
    Analyzes a user-defined area by identifying biomes, calculating their centroids,
    and running climate and ecology predictions for each distinct biome.
    """
    try:
        shape_polygon = ShapelyPolygon([(lng, lat) for lat, lng in request.shape_points])
        min_lng, min_lat, max_lng, max_lat = shape_polygon.bounds

        # 1. Sample points within the polygon to identify present biomes
        sample_points = 500
        biome_point_groups = {}
        for _ in range(sample_points):
            point_lng, point_lat = (random.uniform(min_lng, max_lng), random.uniform(min_lat, max_lat))
            if shape_polygon.contains(Point(point_lng, point_lat)):
                biome_id, biome_code, biome_name = get_koppen_biome(point_lat, point_lng)
                if biome_code not in ["Unknown", "Error"]:
                    if biome_code not in biome_point_groups:
                        biome_point_groups[biome_code] = {"name": biome_name, "points": []}
                    biome_point_groups[biome_code]["points"].append((point_lng, point_lat))
        
        if not biome_point_groups:
            raise HTTPException(status_code=400, detail="No valid biomes found in the selected area.")

        # 2. Process each identified biome
        analysis_results = []
        day_of_year = datetime(request.year, request.month, 15).timetuple().tm_yday

        for biome_code, data in biome_point_groups.items():
            if not data["points"]: continue
            
            # a) Find the representative center point (centroid) of the biome
            biome_multipoint = MultiPoint(data["points"])
            biome_centroid = biome_multipoint.centroid
            rep_lat, rep_lng = biome_centroid.y, biome_centroid.x

            # b) Get raw climate predictions for the centroid
            temp_k = predict_temperature_k(rep_lat, rep_lng, request.month, request.year)
            precip_raw = predict_precipitation_raw(rep_lat, rep_lng, request.month, request.year)
            rad = predict_radiation(rep_lat, rep_lng, request.month, request.year)
            
            # c) Run the full ecology prediction with engineered features
            species_result = predict_ecology_from_data(
                biome_code=biome_code,
                lat=rep_lat,
                lng=rep_lng,
                doy=day_of_year,
                temp_k=temp_k,
                precip_raw=precip_raw,
                rad=rad
            )
            
            # d) Convert units for user-facing display ONLY
            temp_c_display = round(temp_k - 273.15, 1)
            precip_mm_day_display = round(precip_raw * 86400, 1)

            analysis_results.append({
                "biome": biome_code,
                "biome_name": data["name"],
                "location": {"lat": rep_lat, "lng": rep_lng},
                "climate_data": {
                    "temperature": temp_c_display, 
                    "precipitation": precip_mm_day_display, 
                    "radiation": rad
                },
                "species": species_result["species"],
                "pests": species_result["pests"]
            })

        return {"results": analysis_results}
    except Exception as e:
        logger.exception(f"Analysis failed: {e}") 
        raise HTTPException(status_code=500, detail="An error occurred during analysis.")