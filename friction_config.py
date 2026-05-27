# -*- coding: utf-8 -*-
"""friction_config.py

Single source of truth for the deterministic friction-surface pipeline.

The agent-driven model (sentinel `IMPASSABLE = 999`, combined
`PERMAFROST_LULC_MATRIX`, 3-season multipliers, regional overrides) has
been removed. Impassability is expressed as NoData; LULC and permafrost
are independent factors; sea ice and river ice arrive as monthly rasters.
"""

import os

# ---------------------------------------------------------------------------
# Coordinate Reference System
# ---------------------------------------------------------------------------
CRS_TARGET = "EPSG:3413"      # WGS 84 / NSIDC Sea Ice Polar Stereographic North
TARGET_RESOLUTION = 150       # metres, following Trochim et al.

# ---------------------------------------------------------------------------
# NoData and modes
# ---------------------------------------------------------------------------
FRICTION_NODATA = -9999.0
MODES = ("overland", "barge", "ice_road")

# ---------------------------------------------------------------------------
# Slope friction (degrees -> unitless friction)
# ---------------------------------------------------------------------------
# flat (<2): 1.0  |  rolling (2-8): 1.4  |  mountain (>=8): 1.75
SLOPE_THRESHOLDS = (2.0, 8.0)
SLOPE_FRICTION = (1.0, 1.4, 1.75)

# ---------------------------------------------------------------------------
# Land use / land cover (Dynamic World v1 modal class codes)
# ---------------------------------------------------------------------------
# value=None marks NoData in the overland base; mode-specific surfaces
# handle water themselves (barge / ice_road).
LULC_WATER_CLASS = 0
LULC_FRICTION = {
    0: None,    # water -> NoData in overland base
    1: 1.46,    # trees
    2: 1.15,    # grass
    3: 1.63,    # flooded_vegetation
    4: 1.10,    # crops
    5: 1.15,    # shrub_scrub
    6: 1.43,    # built_area
    7: 1.16,    # bare_ground
    8: 5.00,    # snow_ice
}

# ---------------------------------------------------------------------------
# Permafrost seasonal modifier
# ---------------------------------------------------------------------------
# Winter (Nov-Apr): modifier = 1.0 everywhere (frozen ground is trafficable).
# Shoulder (May, Oct):  1.0 + permafrost * (PERMAFROST_MAX_SHOULDER - 1.0)
# Summer (Jun-Sep):     1.0 + permafrost * (PERMAFROST_MAX_SUMMER   - 1.0)
PERMAFROST_WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
PERMAFROST_SHOULDER_MONTHS = {5, 10}
PERMAFROST_SUMMER_MONTHS = {6, 7, 8, 9}
PERMAFROST_MAX_SHOULDER = 1.15      # at 100% permafrost in May / Oct
PERMAFROST_MAX_SUMMER = 1.40        # at 100% permafrost in Jun-Sep

# ---------------------------------------------------------------------------
# Ice and water friction (per-mode)
# ---------------------------------------------------------------------------
ICE_PROB_THRESHOLD = 0.5            # combined sea/river ice probability
WATER_FRICTION_BARGE = 0.5          # ice-free water under barge mode
WATER_FRICTION_ICEROAD = 0.8        # high-ice water under ice_road mode

# ---------------------------------------------------------------------------
# Region ID mapping (integer codes in regions_alaska.tif -> region names)
# ---------------------------------------------------------------------------
# Retained for `notebooks/gee_preprocessing.ipynb`. The new friction
# pipeline does not consume this dictionary.
REGION_IDS = {
    1: "Southeast",
    2: "Kodiak",
    3: "Aleutians",
    4: "Copper River Chugach",
    5: "Bristol Bay",
    6: "Yukon-Kuskokwim Delta",
    7: "Interior",
    8: "Northwest Arctic",
    9: "North Slope",
    10: "Railbelt",
}

# ---------------------------------------------------------------------------
# Baseline delivery cost rates ($ per friction-mile)
# ---------------------------------------------------------------------------
BASELINE_RATES = {
    "Road":  3.5,    # mid-range of $2-5/mi
    "Barge": 2.0,    # mid-range of $1-3/mi
    "Plane": 11.5,   # mid-range of $8-15/mi
}

# ---------------------------------------------------------------------------
# Data source paths (default, overridable via environment variables)
# ---------------------------------------------------------------------------
RASTER_DIR = os.getenv("RASTER_DIR", "./rasters")
VECTOR_DIR = os.getenv("VECTOR_DIR", "./vectors")

RASTER_FILES = {
    "slope":      os.path.join(RASTER_DIR, "slope.tif"),
    "lulc":       os.path.join(RASTER_DIR, "lulc.tif"),
    "permafrost": os.path.join(RASTER_DIR, "permafrost.tif"),
    "sea_ice":    os.path.join(RASTER_DIR, "sea_ice"),    # directory of sea_ice_{01..12}.tif
    "river_ice":  os.path.join(RASTER_DIR, "river_ice"),  # directory of river_ice_{01..12}.tif
}

VECTOR_FILES = {
    "roads":        os.path.join(VECTOR_DIR, "roads.gpkg"),
    "waterways":    os.path.join(VECTOR_DIR, "waterways.gpkg"),
    "flight_paths": os.path.join(VECTOR_DIR, "flight_paths.gpkg"),
}
