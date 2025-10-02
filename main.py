"""
KaalNetra Backend API Server
FastAPI server for generating random geographical points within shapes
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Tuple
import random
import math
from shapely.geometry import Point, Polygon as ShapelyPolygon
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KaalNetra API",
    description="Generate random geographical points within shapes",
    version="1.0.0"
)

# CORS Configuration - Allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        # Add your production domain here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Data Models
# ============================================================================

class Rectangle(BaseModel):
    """Rectangle defined by northeast and southwest corners"""
    ne: Tuple[float, float] = Field(..., description="Northeast corner (lat, lng)")
    sw: Tuple[float, float] = Field(..., description="Southwest corner (lat, lng)")
    
    @validator('ne', 'sw')
    def validate_coordinates(cls, v):
        lat, lng = v
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        if not (-180 <= lng <= 180):
            raise ValueError(f"Longitude must be between -180 and 180, got {lng}")
        return v


class Circle(BaseModel):
    """Circle defined by center point and radius"""
    center: Tuple[float, float] = Field(..., description="Center point (lat, lng)")
    radius: float = Field(..., gt=0, description="Radius in meters")
    
    @validator('center')
    def validate_center(cls, v):
        lat, lng = v
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        if not (-180 <= lng <= 180):
            raise ValueError(f"Longitude must be between -180 and 180, got {lng}")
        return v


class Polygon(BaseModel):
    """Polygon defined by a list of vertices"""
    points: List[Tuple[float, float]] = Field(..., min_items=3, description="List of vertices (lat, lng)")
    
    @validator('points')
    def validate_points(cls, v):
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 points")
        for lat, lng in v:
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
            if not (-180 <= lng <= 180):
                raise ValueError(f"Longitude must be between -180 and 180, got {lng}")
        return v


class PointsResponse(BaseModel):
    """Response containing generated points"""
    points: List[Tuple[float, float]]
    count: int
    shape_type: str


# ============================================================================
# Helper Functions
# ============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max"""
    return max(min_val, min(max_val, value))


def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians"""
    return degrees * math.pi / 180


def rad_to_deg(radians: float) -> float:
    """Convert radians to degrees"""
    return radians * 180 / math.pi


# ============================================================================
# Point Generation Algorithms
# ============================================================================

def generate_points_in_rectangle(ne: Tuple[float, float], sw: Tuple[float, float], count: int) -> List[Tuple[float, float]]:
    """
    Generate random points uniformly distributed within a rectangle.
    
    Args:
        ne: Northeast corner (lat, lng)
        sw: Southwest corner (lat, lng)
        count: Number of points to generate
        
    Returns:
        List of (lat, lng) tuples
    """
    lat_min = min(ne[0], sw[0])
    lat_max = max(ne[0], sw[0])
    lng_min = min(ne[1], sw[1])
    lng_max = max(ne[1], sw[1])
    
    points = []
    for _ in range(count):
        lat = random.uniform(lat_min, lat_max)
        lng = random.uniform(lng_min, lng_max)
        points.append((lat, lng))
    
    logger.info(f"Generated {len(points)} points in rectangle")
    return points


def generate_points_in_circle(center: Tuple[float, float], radius: float, count: int) -> List[Tuple[float, float]]:
    """
    Generate random points uniformly distributed within a circle using spherical geometry.
    
    Args:
        center: Center point (lat, lng)
        radius: Radius in meters
        count: Number of points to generate
        
    Returns:
        List of (lat, lng) tuples
    """
    lat_center, lng_center = center
    lat_rad = deg_to_rad(lat_center)
    lng_rad = deg_to_rad(lng_center)
    
    # Earth's radius in meters
    EARTH_RADIUS = 6371000
    
    points = []
    for _ in range(count):
        # Random angle (0 to 2π)
        angle = random.uniform(0, 2 * math.pi)
        
        # Random distance with sqrt for uniform distribution
        distance = radius * math.sqrt(random.random())
        
        # Angular distance
        angular_distance = distance / EARTH_RADIUS
        
        # Calculate new point using spherical geometry
        new_lat = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance) +
            math.cos(lat_rad) * math.sin(angular_distance) * math.cos(angle)
        )
        
        new_lng = lng_rad + math.atan2(
            math.sin(angle) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat)
        )
        
        # Convert back to degrees
        lat = rad_to_deg(new_lat)
        lng = rad_to_deg(new_lng)
        
        # Normalize longitude to [-180, 180]
        lng = ((lng + 180) % 360) - 180
        
        points.append((lat, lng))
    
    logger.info(f"Generated {len(points)} points in circle (radius: {radius}m)")
    return points


def generate_points_in_polygon(vertices: List[Tuple[float, float]], count: int) -> List[Tuple[float, float]]:
    """
    Generate random points within a polygon using rejection sampling.
    
    Args:
        vertices: List of polygon vertices (lat, lng)
        count: Number of points to generate
        
    Returns:
        List of (lat, lng) tuples
    """
    # Convert to Shapely polygon (note: Shapely uses (lng, lat) order)
    shapely_vertices = [(lng, lat) for lat, lng in vertices]
    polygon = ShapelyPolygon(shapely_vertices)
    
    # Get bounding box
    min_lng, min_lat, max_lng, max_lat = polygon.bounds
    
    points = []
    attempts = 0
    max_attempts = count * 1000  # Safety limit
    
    while len(points) < count and attempts < max_attempts:
        attempts += 1
        
        # Generate random point in bounding box
        lat = random.uniform(min_lat, max_lat)
        lng = random.uniform(min_lng, max_lng)
        point = Point(lng, lat)
        
        # Check if point is inside polygon
        if polygon.contains(point):
            points.append((lat, lng))
    
    if len(points) < count:
        logger.warning(f"Only generated {len(points)}/{count} points after {attempts} attempts")
    else:
        logger.info(f"Generated {len(points)} points in polygon after {attempts} attempts")
    
    return points


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "KaalNetra API",
        "version": "1.0.0",
        "endpoints": [
            "/random-points/rectangle",
            "/random-points/circle",
            "/random-points/polygon"
        ]
    }


@app.post("/random-points/rectangle", response_model=PointsResponse)
async def random_points_rectangle(
    rect: Rectangle,
    count: int = Query(default=100, ge=1, le=5000, description="Number of points to generate")
):
    """
    Generate random points within a rectangle.
    
    - **ne**: Northeast corner coordinates (lat, lng)
    - **sw**: Southwest corner coordinates (lat, lng)
    - **count**: Number of points to generate (1-5000)
    """
    try:
        points = generate_points_in_rectangle(rect.ne, rect.sw, count)
        return PointsResponse(
            points=points,
            count=len(points),
            shape_type="rectangle"
        )
    except Exception as e:
        logger.error(f"Error generating rectangle points: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/random-points/circle", response_model=PointsResponse)
async def random_points_circle(
    circle: Circle,
    count: int = Query(default=100, ge=1, le=5000, description="Number of points to generate")
):
    """
    Generate random points within a circle.
    
    - **center**: Center point coordinates (lat, lng)
    - **radius**: Radius in meters
    - **count**: Number of points to generate (1-5000)
    """
    try:
        points = generate_points_in_circle(circle.center, circle.radius, count)
        return PointsResponse(
            points=points,
            count=len(points),
            shape_type="circle"
        )
    except Exception as e:
        logger.error(f"Error generating circle points: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/random-points/polygon", response_model=PointsResponse)
async def random_points_polygon(
    poly: Polygon,
    count: int = Query(default=100, ge=1, le=5000, description="Number of points to generate")
):
    """
    Generate random points within a polygon.
    
    - **points**: List of polygon vertices (lat, lng)
    - **count**: Number of points to generate (1-5000)
    """
    try:
        points = generate_points_in_polygon(poly.points, count)
        return PointsResponse(
            points=points,
            count=len(points),
            shape_type="polygon"
        )
    except Exception as e:
        logger.error(f"Error generating polygon points: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
