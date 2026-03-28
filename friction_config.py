# -*- coding: utf-8 -*-
"""friction_config.py

Centralized lookup tables, weights, thresholds, and seasonal multipliers
for the Alaska fuel delivery friction surface system.

All friction values are unitless multipliers applied to a base traversal
cost.  A value of 1.0 represents ideal conditions; higher values indicate
greater difficulty.  A value of 999 indicates an impassable cell.

Sources:
    - Trochim et al. (review) -- land cover and permafrost friction values
    - Atkinson et al., 2005 -- slope classification thresholds
    - FAA 14 CFR 91.177 -- minimum safe altitude (MSA) for planes
    - Obu et al., 2019 -- permafrost zonation classification
"""

# ---------------------------------------------------------------------------
# Coordinate Reference System
# ---------------------------------------------------------------------------
CRS_TARGET = "EPSG:3413"  # WGS 84 / NSIDC Sea Ice Polar Stereographic North
TARGET_RESOLUTION = 150    # metres, following Trochim et al.

# ---------------------------------------------------------------------------
# Impassable sentinel
# ---------------------------------------------------------------------------
IMPASSABLE = 999

# ---------------------------------------------------------------------------
# Slope friction (Road delivery only)
# ---------------------------------------------------------------------------
# Thresholds in degrees
SLOPE_THRESHOLDS = (2.0, 8.0)  # flat < 2, rolling 2-8, mountain > 8

SLOPE_FRICTION = {
    "flat": 1.0,      # < 2 degrees
    "rolling": 1.4,   # 2 - 8 degrees
    "mountain": 1.75,  # > 8 degrees
}

# ---------------------------------------------------------------------------
# Dynamic World land-use / land-cover (LULC) class codes
# ---------------------------------------------------------------------------
# Dynamic World v1 class codes (0-8)
LULC_CLASSES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub",
    6: "built",
    7: "bare_ground",
    8: "snow_ice",
}

# ---------------------------------------------------------------------------
# Base LULC friction for Road delivery
# ---------------------------------------------------------------------------
# Water (class 0) is handled separately as impassable for road
LULC_FRICTION_ROAD = {
    0: IMPASSABLE,  # water
    1: 1.46,        # trees
    2: 1.15,        # grass
    3: 1.63,        # flooded vegetation
    4: 1.1,         # crops
    5: 1.15,        # shrub (same as grass per Trochim)
    6: 1.0,         # built-up area
    7: 1.16,        # bare ground / tundra
    8: 5.0,         # snow & ice
}

# ---------------------------------------------------------------------------
# Permafrost zones (Obu et al., 2019 reclassified)
# ---------------------------------------------------------------------------
PERMAFROST_ZONES = {
    0: "none",
    1: "sporadic",
    2: "discontinuous",
    3: "continuous",
}

# ---------------------------------------------------------------------------
# Combined Permafrost x LULC friction matrix (Road delivery)
# ---------------------------------------------------------------------------
# For LULC classes that interact with permafrost, the combined value
# replaces the base LULC friction.  Classes not listed here (water,
# crops, snow_ice, built) use their base LULC friction unchanged.
PERMAFROST_LULC_MATRIX = {
    # lulc_class: {permafrost_zone: combined_friction}
    2: {  # grass
        0: 1.15, 1: 1.79, 2: 1.83, 3: 1.88,
    },
    5: {  # shrub (same values as grass per Trochim)
        0: 1.15, 1: 1.79, 2: 1.83, 3: 1.88,
    },
    1: {  # trees
        0: 1.46, 1: 1.87, 2: 1.92, 3: 1.96,
    },
    3: {  # flooded vegetation
        0: 1.63, 1: 2.11, 2: 2.16, 3: 2.21,
    },
    7: {  # bare ground / tundra
        0: 1.16, 1: 1.79, 2: 1.83, 3: 1.88,
    },
}

# ---------------------------------------------------------------------------
# Road surface type friction (AK DOT + USGS NTD classification)
# ---------------------------------------------------------------------------
# Rasterized road type values in roads_type_alaska.tif
ROAD_TYPE_FRICTION = {
    1: 1.0,   # paved
    2: 1.1,   # gravel
    3: 1.6,   # dirt
}

# ---------------------------------------------------------------------------
# River friction
# ---------------------------------------------------------------------------
# Rasterized river classes in rivers_alaska.tif (1=major, 2=minor)
RIVER_FRICTION_ROAD = {
    1: IMPASSABLE,  # major river -- impassable for road
    2: IMPASSABLE,  # minor river -- impassable for road
}

RIVER_FRICTION_BARGE = {
    1: 1.0,   # major navigable river
    2: 1.5,   # minor river
}

# ---------------------------------------------------------------------------
# Port access friction (Barge delivery)
# ---------------------------------------------------------------------------
PORT_FRICTION = {
    "port": 1.0,           # full port access
    "beach_landing": 1.4,  # beach landing site
    "no_port": IMPASSABLE, # no port access
}

# ---------------------------------------------------------------------------
# Airport access friction (Plane delivery)
# ---------------------------------------------------------------------------
AIRPORT_FRICTION = {
    "access": 1.0,           # airstrip available
    "no_access": IMPASSABLE, # no airstrip
}

# ---------------------------------------------------------------------------
# Plane MSA (Minimum Safe Altitude) -- FAA 14 CFR 91.177
# ---------------------------------------------------------------------------
# Terrain cells below MSA are impassable; above MSA get a flat friction
PLANE_MSA_FEET = {
    "non_mountainous": 1000,  # 1000 ft AGL
    "mountainous": 2000,      # 2000 ft AGL (most of Alaska)
}
PLANE_FRICTION_ABOVE_MSA = 10.0   # flat cost for traversable airspace
PLANE_FRICTION_BELOW_MSA = IMPASSABLE

# ---------------------------------------------------------------------------
# LULC friction for Barge delivery
# ---------------------------------------------------------------------------
# All land cells are impassable for barge (except at port transitions)
LULC_FRICTION_BARGE = {k: IMPASSABLE for k in LULC_CLASSES}
LULC_FRICTION_BARGE[0] = 1.0  # water is navigable

# ---------------------------------------------------------------------------
# Seasonal multipliers
# ---------------------------------------------------------------------------
# Applied to avg_friction after path computation.  Keyed by
# (feature_type, season) -> multiplier.
# Seasons: "summer" (Jun-Aug), "shoulder" (May & Oct), "winter" (Nov-Apr)
SEASONS = ["summer", "shoulder", "winter"]

SEASONAL_MULTIPLIERS = {
    # Major river navigability
    ("major_river", "summer"):   1.0,
    ("major_river", "shoulder"): 1.3,
    ("major_river", "winter"):   IMPASSABLE,

    # Minor river navigability
    ("minor_river", "summer"):   1.0,
    ("minor_river", "shoulder"): 1.5,
    ("minor_river", "winter"):   IMPASSABLE,

    # Sea ice (coastal / ocean routes)
    ("sea_ice", "summer"):       1.0,
    ("sea_ice", "shoulder"):     1.5,
    ("sea_ice", "winter"):       IMPASSABLE,

    # Marginal sea ice zones
    ("marginal_sea_ice", "summer"):   1.0,
    ("marginal_sea_ice", "shoulder"): 2.0,
    ("marginal_sea_ice", "winter"):   IMPASSABLE,
}

# ---------------------------------------------------------------------------
# Baseline delivery cost rates ($ per friction-mile)
# ---------------------------------------------------------------------------
# These are initial estimates; the Validation Agent will derive
# regional calibration multipliers from ISER / AEA benchmarks.
BASELINE_RATES = {
    "Road":  3.5,   # mid-range of $2-5/mi
    "Barge": 2.0,   # mid-range of $1-3/mi
    "Plane": 11.5,  # mid-range of $8-15/mi
}

# ---------------------------------------------------------------------------
# GRIP4 road presence (binary raster values)
# ---------------------------------------------------------------------------
# roads_presence_alaska.tif: 1 = road present, 0/NoData = no road
GRIP4_ROAD_PRESENT = 1

# ---------------------------------------------------------------------------
# Data source paths (default, overridable via environment variables)
# ---------------------------------------------------------------------------
import os

RASTER_DIR = os.getenv("RASTER_DIR", "./rasters")
VECTOR_DIR = os.getenv("VECTOR_DIR", "./vectors")

RASTER_FILES = {
    "lulc":              os.path.join(RASTER_DIR, "lulc_alaska_modal.tif"),
    "slope":             os.path.join(RASTER_DIR, "slope_alaska.tif"),
    "permafrost":        os.path.join(RASTER_DIR, "permafrost_alaska.tif"),
    "roads_presence":    os.path.join(RASTER_DIR, "roads_presence_alaska.tif"),
    "roads_type":        os.path.join(RASTER_DIR, "roads_type_alaska.tif"),
    "rivers":            os.path.join(RASTER_DIR, "rivers_alaska.tif"),
    "dem":               os.path.join(RASTER_DIR, "dem_alaska.tif"),
}

VECTOR_FILES = {
    "airports":   os.path.join(VECTOR_DIR, "airports_alaska.geojson"),
    "ports":      os.path.join(VECTOR_DIR, "ports_alaska.geojson"),
    "facilities": os.path.join(VECTOR_DIR, "facilities_alaska.geojson"),
}
