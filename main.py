from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
import random
import math
from shapely.geometry import Point, Polygon as ShapelyPolygon
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend (localhost:3000) to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # allow all headers
)

#Defining data models

class Rectangle(BaseModel):
    ne: Tuple[float, float]  # (lat, lng) for northeast corner
    sw: Tuple[float, float]  # (lat, lng) for southwest corner

class Circle(BaseModel):
    center: Tuple[float, float]  # (lat, lng) for circle center
    radius: float  # radius in meters

class Polygon(BaseModel):
    points: List[Tuple[float, float]]  # list of (lat, lng) points

@app.post("/random-points/rectangle")
def random_points_rectangle(rect: Rectangle, count: int = 5):
    points = []
    lat_min, lng_min = rect.sw
    lat_max, lng_max = rect.ne

    for _ in range(count):
        lat = random.uniform(lat_min, lat_max)
        lng = random.uniform(lng_min, lng_max)
        points.append((lat, lng))

    return {"points": points}


@app.post("/random-points/circle")
def random_points_circle(circle: Circle, count: int = 5):
    points = []
    lat_center, lng_center = circle.center

    for _ in range(count):
        # Pick a random angle (0 to 360 degrees in radians)
        angle = random.uniform(0, 2 * math.pi)

        # Pick a random distance (but sqrt for uniform distribution)
        r = circle.radius * math.sqrt(random.random())

        # Convert meters to degrees (approx, works for small areas)
        delta_lat = (r / 111320) * math.cos(angle)   # 111,320 meters ≈ 1 degree latitude
        delta_lng = (r / (40075000 * math.cos(math.radians(lat_center)) / 360)) * math.sin(angle)

        # New random point
        lat = lat_center + delta_lat
        lng = lng_center + delta_lng
        points.append((lat, lng))

    return {"points": points}

@app.post("/random-points/polygon")
def random_points_polygon(poly: Polygon, count: int = 5):
    # flip lat/lng → (lng, lat) for Shapely
    shapely_poly = ShapelyPolygon([(lng, lat) for lat, lng in poly.points])
    minx, miny, maxx, maxy = shapely_poly.bounds
    points = []

    while len(points) < count:
        # Pick random point in bounding box
        lng = random.uniform(minx, maxx)
        lat = random.uniform(miny, maxy)
        p = Point(lng, lat)

        if shapely_poly.contains(p):
            points.append((lat, lng))  # return back as (lat, lng)

    return {"points": points}