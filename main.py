# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Tuple
import random
import logging
import rasterio
from shapely.geometry import Point, Polygon as ShapelyPolygon

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KaalNetra API",
    description="Geospatial analysis for ecology prediction",
    version="1.4.0" # Updated to standard Köppen
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mock ML Models (Unchanged) ---
def mock_temp_model(lat: float, lng: float, month: int, year: int) -> float:
    return round(random.uniform(15, 30), 1)

def mock_precip_model(lat: float, lng: float, month: int, year: int) -> float:
    return round(random.uniform(50, 200), 1)

def mock_rad_model(lat: float, lng: float, month: int, year: int) -> float:
    return round(random.uniform(150, 250), 1)

def mock_final_species_model(temp: float, precip: float, rad: float, biome: str) -> dict:
    species_list = [
        {'scientific_name': 'Rhamnus cathartica', 'common_name': 'common buckthorn', 'phenophase': 'Fruiting'},
        {'scientific_name': 'Chamaenerion angustifolium', 'common_name': 'fireweed', 'phenophase': 'Fruiting'},
        {'scientific_name': 'Acer negundo', 'common_name': 'box elder', 'phenophase': 'Fruiting'},
    ]
    pests_list = [
        {'scientific_name_pest': 'Agrilus planipennis', 'common_name_pest': 'Emerald Ash Borer'}
    ]
    return {
        "species": random.sample(species_list, k=random.randint(2, 3)),
        "pests": random.sample(pests_list, k=1)
    }

# --- Standard Köppen Classification Dictionary ---
# This dictionary maps the numeric code from the .tif file to the detailed 3-letter code and name.
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

# --- UPDATED Helper to use the standard dictionary ---
def get_koppen_biome(lat: float, lng: float, raster_file="koppen.tif") -> Tuple[str, str]:
    """
    Extracts the numeric value from the raster file and looks it up in the standard KOPPEN_CLASSES dictionary.
    """
    try:
        with rasterio.open(raster_file) as src:
            # Rasterio uses (x, y) which corresponds to (lng, lat)
            for val in src.sample([(lng, lat)]):
                class_id = int(val[0])
                # Use the dictionary for a direct lookup, with a fallback for unknown codes
                return KOPPEN_CLASSES.get(class_id, ("Unknown", "Unknown"))
    except Exception as e:
        logger.error(f"Could not read from raster file: {e}")
        return ("Error", "Error")

# --- API Models (Unchanged) ---
class AnalysisRequest(BaseModel):
    shape_points: List[Tuple[float, float]] = Field(..., min_items=3)
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=1980, le=2099)

# --- Endpoints (The logic is unchanged, but now uses the detailed biome data) ---
@app.get("/")
async def root():
    return {"status": "KaalNetra API is online"}

@app.post("/analyze-by-biome")
async def analyze_area_by_biome(request: AnalysisRequest):
    try:
        shape_polygon = ShapelyPolygon([(lng, lat) for lat, lng in request.shape_points])
        centroid = shape_polygon.centroid
        min_lng, min_lat, max_lng, max_lat = shape_polygon.bounds

        sample_points = 200
        points_by_biome = {}
        for _ in range(sample_points):
            point_lng = random.uniform(min_lng, max_lng)
            point_lat = random.uniform(min_lat, max_lat)
            if shape_polygon.contains(Point(point_lng, point_lat)):
                # This call now uses the standard get_koppen_biome function
                biome_code, biome_name = get_koppen_biome(point_lat, point_lng)
                if biome_code not in points_by_biome and biome_code not in ["Unknown", "Error"]:
                    points_by_biome[biome_code] = {
                        "name": biome_name,
                        "representative_point": (point_lat, point_lng)
                    }
        
        if not points_by_biome:
            raise HTTPException(status_code=400, detail="No valid biomes found in the selected area.")

        analysis_results = []
        for biome_code, data in points_by_biome.items():
            rep_lat, rep_lng = data["representative_point"]
            
            temp = mock_temp_model(rep_lat, rep_lng, request.month, request.year)
            precip = mock_precip_model(rep_lat, rep_lng, request.month, request.year)
            rad = mock_rad_model(rep_lat, rep_lng, request.month, request.year)
            
            species_result = mock_final_species_model(temp, precip, rad, biome_code)
            
            analysis_results.append({
                "biome": biome_code,
                "biome_name": data["name"],
                "climate_data": { "temperature": temp, "precipitation": precip, "radiation": rad },
                "species": species_result["species"],
                "pests": species_result["pests"]
            })

        return {
            "centroid": {"lat": centroid.y, "lng": centroid.x},
            "results": analysis_results
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during analysis.")