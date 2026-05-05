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
    - Pastick et al., 2015 -- near-surface permafrost probability (30m)
      DOI: 10.5066/F7C53HX6
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
# Friction nodata sentinel (true barrier for WhiteboxTools cost_distance)
# ---------------------------------------------------------------------------
FRICTION_NODATA = -9999.0

# ---------------------------------------------------------------------------
# Water type codes (pixel-level classification for seasonal surfaces)
# ---------------------------------------------------------------------------
WATER_TYPE_LAND              = 0
WATER_TYPE_RIVER             = 1
WATER_TYPE_SEASONALLY_FROZEN = 2   # >70% climatological winter ice concentration
WATER_TYPE_SEASONALLY_MARGINAL = 3 # 20-70% climatological winter ice concentration
WATER_TYPE_OPEN              = 4   # <20% ice, navigable year-round by default

# Water type -> feature name for seasonal multiplier lookup
WATER_TYPE_FEATURES = {
    WATER_TYPE_RIVER:              "major_river",
    WATER_TYPE_SEASONALLY_FROZEN:  "sea_ice",
    WATER_TYPE_SEASONALLY_MARGINAL: "marginal_sea_ice",
    WATER_TYPE_OPEN:               "open_water",
}

# ---------------------------------------------------------------------------
# Sea ice classification thresholds (climatological winter concentration %)
# ---------------------------------------------------------------------------
SEA_ICE_THRESHOLDS = {
    "sea_ice":          70.0,   # >70% -> seasonally frozen
    "marginal_sea_ice": 20.0,   # 20-70% -> seasonally marginal
}

# ---------------------------------------------------------------------------
# Region ID mapping (integer codes in regions_alaska.tif -> region names)
# ---------------------------------------------------------------------------
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
# Permafrost zones (reclassified from Pastick et al., 2015 probability)
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
# ---------------------------------------------------------------------------
# Road presence friction (GRIP4 binary)
# ---------------------------------------------------------------------------
# Friction value applied wherever a GRIP4 road is present.
# Was previously a paved/gravel/dirt lookup from AK DOT data, but the
# AK DOT surface type data was unreliable, so we treat all roads
# uniformly as ideal traversal (1.0).
ROAD_PRESENT_FRICTION = 1.0

# Number of pixels to dilate the GRIP4 road mask before applying the
# friction override.  1 pixel at 150 m absorbs vector-to-raster snap
# error and stabilises WhiteboxTools cost-distance paths along narrow
# road corridors.  Set to 0 to disable.
ROAD_BUFFER_PIXELS = 1

# ---------------------------------------------------------------------------
# River friction
# ---------------------------------------------------------------------------
# Rasterized waterway classes in rivers_alaska.tif (1=navigable)
# All USACE NWN features are treated as navigable; no minor class.
RIVER_FRICTION_ROAD = {
    1: IMPASSABLE,  # navigable waterway -- impassable for road
}

RIVER_FRICTION_BARGE = {
    1: 1.0,   # navigable waterway (all NWN features)
}

# ---------------------------------------------------------------------------
# Port access friction (Barge delivery)
# ---------------------------------------------------------------------------
PORT_FRICTION = {
    "port": 1.0,           # full port access
    "beach_landing": 1.4,  # beach landing site
    "no_port": IMPASSABLE, # no port access
}

PORT_BUFFER = {
    "port":          3,    # 7x7 kernel, ~1 km diameter at 150m
    "beach_landing": 1,    # 3x3 kernel, ~450m diameter
}

PORT_DECAY_PER_PIXEL = {
    "port":          0.1,  # 1.0 -> 1.1 -> 1.2 -> 1.3
    "beach_landing": 0.0,  # uniform 1.4 across buffer
}

# ---------------------------------------------------------------------------
# Plane delivery
# ---------------------------------------------------------------------------
# Plane routes use direct Haversine distance (airport-to-airport) rather
# than a cell-by-cell friction raster.  Cost = distance_miles × BASELINE_RATE.
# No terrain friction surface is built for plane delivery.

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
    # Navigable waterway (USACE NWN) navigability
    ("major_river", "summer"):   1.0,
    ("major_river", "shoulder"): 1.3,
    ("major_river", "winter"):   IMPASSABLE,

    # Sea ice (coastal / ocean routes)
    ("sea_ice", "summer"):       1.0,
    ("sea_ice", "shoulder"):     1.5,
    ("sea_ice", "winter"):       IMPASSABLE,

    # Marginal sea ice zones
    ("marginal_sea_ice", "summer"):   1.0,
    ("marginal_sea_ice", "shoulder"): 2.0,
    ("marginal_sea_ice", "winter"):   IMPASSABLE,

    # Open water (<20% climatological ice, navigable year-round by default)
    ("open_water", "summer"):   1.0,
    ("open_water", "shoulder"): 1.0,
    ("open_water", "winter"):   1.0,
}

# Regional overrides: (region, feature_type, season) -> multiplier
# Overrides SEASONAL_MULTIPLIERS when a facility is in the given region.
# Regions without overrides fall back to the global defaults above.
REGIONAL_SEASONAL_OVERRIDES = {
    # Southeast: temperate maritime climate, ice-free waters year-round
    ("Southeast", "major_river", "shoulder"):        1.1,
    ("Southeast", "major_river", "winter"):          1.3,
    ("Southeast", "sea_ice", "shoulder"):            1.1,
    ("Southeast", "sea_ice", "winter"):              1.4,
    ("Southeast", "marginal_sea_ice", "shoulder"):   1.2,
    ("Southeast", "marginal_sea_ice", "winter"):     1.5,

    # Kodiak: mild maritime climate, limited ice impact
    ("Kodiak", "major_river", "shoulder"):           1.2,
    ("Kodiak", "major_river", "winter"):             1.8,
    ("Kodiak", "sea_ice", "shoulder"):               1.2,
    ("Kodiak", "sea_ice", "winter"):                 2.0,
    ("Kodiak", "marginal_sea_ice", "shoulder"):      1.3,
    ("Kodiak", "marginal_sea_ice", "winter"):        2.5,

    # Aleutians: maritime but exposed, some winter navigability
    ("Aleutians", "sea_ice", "shoulder"):            1.3,
    ("Aleutians", "sea_ice", "winter"):              2.0,
    ("Aleutians", "marginal_sea_ice", "shoulder"):   1.5,
    ("Aleutians", "marginal_sea_ice", "winter"):     3.0,

    # Copper River Chugach: coastal portions navigable longer
    ("Copper River Chugach", "sea_ice", "shoulder"): 1.3,
    ("Copper River Chugach", "sea_ice", "winter"):   2.5,

    # Bristol Bay: short ice-free window, harsh shoulder/winter
    ("Bristol Bay", "sea_ice", "shoulder"):           1.8,
    ("Bristol Bay", "sea_ice", "winter"):             IMPASSABLE,
    ("Bristol Bay", "major_river", "shoulder"):       1.5,
    ("Bristol Bay", "major_river", "winter"):         IMPASSABLE,
}

# ---------------------------------------------------------------------------
# Baseline delivery cost rates ($ per friction-mile)
# ---------------------------------------------------------------------------
# Initial estimates; the Validation Agent derives regional calibration
# multipliers by comparing computed costs against community fuel prices from:
#   - Alaska Energy Data Gateway (AEDG): akenergygateway.alaska.edu
#   - DCRA Alaska Fuel Price Reports: storymaps.arcgis.com (semi-annual surveys)
#   - ISER / AEA published Alaska energy cost studies
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
    "rivers":            os.path.join(RASTER_DIR, "rivers_alaska.tif"),
    "dem":               os.path.join(RASTER_DIR, "dem_alaska.tif"),
    "sea_ice":           os.path.join(RASTER_DIR, "sea_ice_concentration_alaska.tif"),
    "regions":           os.path.join(RASTER_DIR, "regions_alaska.tif"),
}

VECTOR_FILES = {
    "airports":   os.path.join(VECTOR_DIR, "airports_alaska.geojson"),
    "ports":      os.path.join(VECTOR_DIR, "ports_alaska.geojson"),
    "facilities": os.path.join(VECTOR_DIR, "facilities_alaska.geojson"),
}
