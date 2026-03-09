# -*- coding: utf-8 -*-
"""friction_layer_graph.py

Friction Layer Module for Alaska Fuel Delivery.

Builds delivery-method-specific cost surfaces over Alaska incorporating
roads, waterways, airports, terrain, land cover, and permafrost.
Friction-adjusted distances replace Haversine in the connects_to graph
edges, improving regionalization assignments and TSP route optimization.

Pipeline Position:
    Step 1: regionalization_graph.main()  (creates DB)
    Step 2: friction_layer_graph.main()   (THIS MODULE)
    Step 3: market_cost_analysis.main()
    Step 4: tsp_model_graph.main()        (uses friction costs)

Data Sources:
    - DEM: USGS 3DEP (AWS Open Data) - slope computation
    - Land Cover: NLCD 2021 Alaska - terrain type friction
    - Roads: Census TIGER (FIPS 02) - road network connectivity
    - Waterways: USGS NHD Alaska - barge navigability
    - Airports: FAA NASR - plane access points
    - Ports: USACE - barge access points
    - Permafrost: NSIDC (Brown et al.) - ground stability

CrewAI Agents:
    - Data Acquisition Agent: Downloads and validates geospatial datasets
    - Friction Modeler Agent: Computes/adjusts friction surfaces and costs
    - Validation Agent: Validates costs and recommends fixes

Task Flow:
    Task 1: Data Acquisition Agent -> download & validate datasets
    Task 2: Friction Modeler Agent -> compute initial surfaces & costs
    Task 3: Validation Agent -> validate, identify issues, recommend fixes
    Task 4: Friction Modeler Agent -> apply fixes, recompute affected costs
    Task 5: Validation Agent -> final validation pass
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import json
import math
import hashlib
import zipfile
import warnings
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import requests
import duckdb
import geopandas as gpd
import pandas as pd
from pyproj import Transformer

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

import pipeline

warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path('data/friction')
DB_PATH = 'regionalization.duckdb'

# Alaska bounding box in EPSG:3338 (Alaska Albers)
AK_BOUNDS_3338 = {
    'xmin': -2400000, 'xmax': 1600000,
    'ymin': 200000, 'ymax': 2800000,
}
CELL_SIZE_M = 1000  # 1 km grid cells
CRS_ALASKA = 'EPSG:3338'  # Alaska Albers Equal Area

# Coordinate transformer: WGS84 (lon/lat) -> Alaska Albers
TRANSFORMER_4326_TO_3338 = Transformer.from_crs('EPSG:4326', CRS_ALASKA, always_xy=True)

# ---------------------------------------------------------------------------
# Road Class Friction Values
# ---------------------------------------------------------------------------

DEFAULT_ROAD_CLASS_FRICTION = {
    'S1100': 0.5,   # Primary road (interstate/US highway)
    'S1200': 0.7,   # Secondary road (state highway)
    'S1400': 1.0,   # Local road - paved but slower
    'S1500': 1.3,   # Vehicular trail (4WD)
    'S1630': 1.5,   # Ramp
    'S1640': 1.0,   # Service drive
    'S1740': 1.5,   # Ice road / winter-only
    'S1780': 2.0,   # Parking lot road
    'no_road': 8.0, # No road present
}

# ---------------------------------------------------------------------------
# NLCD Land Cover Friction Values
# ---------------------------------------------------------------------------

NLCD_FRICTION = {
    11: 5.0,   # Open water
    12: 3.0,   # Perennial ice/snow
    21: 1.0,   # Developed, open space
    22: 1.0,   # Developed, low intensity
    23: 1.0,   # Developed, medium intensity
    24: 1.0,   # Developed, high intensity
    31: 1.2,   # Barren land
    41: 1.46,  # Deciduous forest
    42: 1.46,  # Evergreen forest
    43: 1.46,  # Mixed forest
    51: 1.3,   # Dwarf scrub (Alaska only)
    52: 1.3,   # Shrub/scrub
    71: 1.15,  # Grassland/herbaceous
    72: 1.2,   # Sedge/herbaceous (Alaska only)
    73: 1.15,  # Lichens (Alaska only)
    74: 1.15,  # Moss (Alaska only)
    81: 1.1,   # Pasture/hay
    82: 1.1,   # Cultivated crops
    90: 3.0,   # Woody wetlands
    95: 3.0,   # Emergent herbaceous wetlands
}

# ---------------------------------------------------------------------------
# Permafrost Friction Values
# ---------------------------------------------------------------------------

PERMAFROST_FRICTION = {
    'continuous': 1.5,
    'discontinuous': 1.3,
    'sporadic': 1.15,
    'isolated': 1.1,
    'none': 1.0,
}

# ---------------------------------------------------------------------------
# Delivery Method Weights
# ---------------------------------------------------------------------------

DEFAULT_DELIVERY_METHOD_WEIGHTS = {
    'Road': {
        'slope_weight': 1.0,
        'lulc_weight': 0.5,
        'road_network_weight': 1.5,
        'permafrost_weight': 1.2,
        'water_barrier': 10.0,
    },
    'Barge': {
        'slope_weight': 0.0,
        'lulc_weight': 0.3,
        'road_network_weight': 0.0,
        'permafrost_weight': 0.5,
        'water_navigable': 0.5,
        'waterway_proximity_weight': 1.5,
    },
    'Plane': {
        'slope_weight': 0.1,
        'lulc_weight': 0.2,
        'road_network_weight': 0.0,
        'permafrost_weight': 0.3,
        'airport_proximity_weight': 2.0,
        'base_air_friction': 1.2,
    },
}

# ---------------------------------------------------------------------------
# Dataset Definitions
# ---------------------------------------------------------------------------

DATASETS = {
    'dem': {
        'name': 'USGS 3DEP DEM',
        'url': 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/USGS_Seamless_DEM_1.vrt',
        'filename': 'alaska_dem.tif',
        'source': 'USGS 3DEP via AWS Open Data',
        'purpose': 'Slope computation',
    },
    'nlcd': {
        'name': 'NLCD 2021 Alaska',
        'url': 'https://s3-us-west-2.amazonaws.com/mrlc/nlcd_2021_land_cover_l48_20230630.zip',
        'filename': 'nlcd_2021_alaska.tif',
        'source': 'MRLC NLCD 2021',
        'purpose': 'Land cover friction',
    },
    'roads': {
        'name': 'Census TIGER Roads Alaska',
        'url': 'https://www2.census.gov/geo/tiger/TIGER2023/ROADS/tl_2023_02_prisecroads.zip',
        'filename': 'tl_2023_02_prisecroads.shp',
        'source': 'Census TIGER/Line 2023',
        'purpose': 'Road network connectivity',
    },
    'waterways': {
        'name': 'USGS NHD Alaska Flowlines',
        'url': 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/State/Shape/NHD_H_Alaska_State_Shape.zip',
        'filename': 'NHDFlowline.shp',
        'source': 'USGS NHD',
        'purpose': 'Barge navigability',
    },
    'airports': {
        'name': 'FAA Airports',
        'url': 'https://opendata.arcgis.com/api/v3/datasets/e747ab91a11045e8b3f8a3efd093d3b5_0/downloads/data?format=geojson&spatialRefId=4326',
        'filename': 'faa_airports.geojson',
        'source': 'FAA NASR',
        'purpose': 'Plane access points',
    },
    'ports': {
        'name': 'USACE Ports',
        'url': 'https://opendata.arcgis.com/api/v3/datasets/3ed5925b84d94734963d0e6e3f0dbc86_0/downloads/data?format=geojson&spatialRefId=4326',
        'filename': 'usace_ports.geojson',
        'source': 'USACE NDC',
        'purpose': 'Barge access points',
    },
    'permafrost': {
        'name': 'NSIDC Permafrost',
        'url': 'https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Aa3584760-79b8-4b02-ae6e-4fb5e0837f5c',
        'filename': 'permafrost_extent.shp',
        'source': 'NSIDC Brown et al.',
        'purpose': 'Ground stability',
    },
}


# ---------------------------------------------------------------------------
# Grid Utilities
# ---------------------------------------------------------------------------

def make_grid_params():
    """Return grid shape and affine-like parameters for the Alaska raster."""
    xmin = AK_BOUNDS_3338['xmin']
    ymax = AK_BOUNDS_3338['ymax']
    ncols = int((AK_BOUNDS_3338['xmax'] - xmin) / CELL_SIZE_M)
    nrows = int((ymax - AK_BOUNDS_3338['ymin']) / CELL_SIZE_M)
    return {
        'nrows': nrows,
        'ncols': ncols,
        'xmin': xmin,
        'ymax': ymax,
        'cell_size': CELL_SIZE_M,
    }


def lonlat_to_rowcol(lon, lat, grid):
    """Convert WGS84 lon/lat to raster row/col indices."""
    x, y = TRANSFORMER_4326_TO_3338.transform(lon, lat)
    col = int((x - grid['xmin']) / grid['cell_size'])
    row = int((grid['ymax'] - y) / grid['cell_size'])
    return row, col


def rowcol_to_xy(row, col, grid):
    """Convert raster row/col to EPSG:3338 x/y (cell center)."""
    x = grid['xmin'] + (col + 0.5) * grid['cell_size']
    y = grid['ymax'] - (row + 0.5) * grid['cell_size']
    return x, y


# ---------------------------------------------------------------------------
# Data Acquisition Functions
# ---------------------------------------------------------------------------

def download_file(url, dest_path, chunk_size=8192):
    """Download a file from URL to dest_path with progress logging."""
    dest_path = Path(dest_path)
    if dest_path.exists():
        logger.info(f"  Already cached: {dest_path.name}")
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading {dest_path.name}...")

    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()

        total = int(resp.headers.get('content-length', 0))
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (chunk_size * 100) == 0:
                    pct = (downloaded / total) * 100
                    logger.info(f"    {pct:.0f}% ({downloaded // 1024 // 1024}MB)")

        logger.info(f"  Downloaded: {dest_path.name} ({downloaded // 1024 // 1024}MB)")
        return dest_path

    except requests.RequestException as e:
        logger.error(f"  Download failed for {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise


def extract_zip(zip_path, extract_dir):
    """Extract a zip file if not already extracted."""
    extract_dir = Path(extract_dir)
    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info(f"  Already extracted: {extract_dir.name}")
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    logger.info(f"  Extracted to: {extract_dir}")
    return extract_dir


def download_dataset(dataset_key):
    """Download and prepare a single dataset.

    Args:
        dataset_key: Key from DATASETS dict (e.g., 'dem', 'roads')

    Returns:
        str: Path to the downloaded/extracted file ready for use
    """
    ds = DATASETS[dataset_key]
    url = ds['url']
    filename = ds['filename']
    dest = DATA_DIR / filename

    if filename.endswith('.shp'):
        # Shapefiles come in zips - download zip, extract, find .shp
        zip_dest = DATA_DIR / f"{dataset_key}.zip"
        extract_dir = DATA_DIR / dataset_key

        if not extract_dir.exists() or not any(extract_dir.rglob('*.shp')):
            download_file(url, zip_dest)
            extract_zip(zip_dest, extract_dir)

        # Find the .shp file
        shp_files = list(extract_dir.rglob('*.shp'))
        if not shp_files:
            raise FileNotFoundError(f"No .shp file found in {extract_dir}")

        # Prefer the specific filename if it exists
        for shp in shp_files:
            if shp.name == filename:
                return str(shp)
        return str(shp_files[0])

    elif filename.endswith('.geojson'):
        download_file(url, dest)
        return str(dest)

    elif filename.endswith('.tif'):
        # For rasters, may need to download and clip to Alaska
        download_file(url, dest)
        return str(dest)

    else:
        download_file(url, dest)
        return str(dest)


def download_all_datasets():
    """Download all required datasets. Returns dict of {key: filepath}."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    status = {}

    for key in DATASETS:
        try:
            path = download_dataset(key)
            results[key] = path
            status[key] = {'status': 'success', 'path': path}
            logger.info(f"  {key}: OK -> {path}")
        except Exception as e:
            results[key] = None
            status[key] = {'status': 'failed', 'error': str(e)}
            logger.warning(f"  {key}: FAILED - {e}")

    return results, status


# ---------------------------------------------------------------------------
# Base Friction Layer Computation
# ---------------------------------------------------------------------------

def compute_slope_friction(dem_path, grid):
    """Compute slope friction raster from DEM.

    Args:
        dem_path: Path to DEM raster (or None if unavailable)
        grid: Grid parameters dict

    Returns:
        np.ndarray: Slope friction raster (nrows x ncols)
    """
    nrows, ncols = grid['nrows'], grid['ncols']

    if dem_path is None:
        logger.warning("DEM not available, using uniform slope friction = 1.0")
        return np.ones((nrows, ncols), dtype=np.float32)

    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds

        dst_transform = from_bounds(
            AK_BOUNDS_3338['xmin'], AK_BOUNDS_3338['ymin'],
            AK_BOUNDS_3338['xmax'], AK_BOUNDS_3338['ymax'],
            ncols, nrows
        )

        with rasterio.open(dem_path) as src:
            dem_ak = np.zeros((nrows, ncols), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dem_ak,
                dst_transform=dst_transform,
                dst_crs=CRS_ALASKA,
                resampling=Resampling.bilinear,
            )

        # Compute slope from DEM (degrees)
        dy, dx = np.gradient(dem_ak, CELL_SIZE_M)
        slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        # Map slope to friction
        friction = np.ones_like(slope_deg, dtype=np.float32)
        friction[slope_deg > 5] = 1.4
        friction[slope_deg > 15] = 1.75

        logger.info(f"Slope friction: min={friction.min():.2f}, max={friction.max():.2f}, "
                     f"mean={friction.mean():.2f}")
        return friction

    except Exception as e:
        logger.warning(f"Could not compute slope from DEM: {e}. Using uniform friction.")
        return np.ones((nrows, ncols), dtype=np.float32)


def compute_lulc_friction(nlcd_path, grid):
    """Compute land use/land cover friction from NLCD.

    Args:
        nlcd_path: Path to NLCD raster (or None)
        grid: Grid parameters dict

    Returns:
        np.ndarray: LULC friction raster
    """
    nrows, ncols = grid['nrows'], grid['ncols']

    if nlcd_path is None:
        logger.warning("NLCD not available, using uniform LULC friction = 1.2")
        return np.full((nrows, ncols), 1.2, dtype=np.float32)

    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds

        dst_transform = from_bounds(
            AK_BOUNDS_3338['xmin'], AK_BOUNDS_3338['ymin'],
            AK_BOUNDS_3338['xmax'], AK_BOUNDS_3338['ymax'],
            ncols, nrows
        )

        with rasterio.open(nlcd_path) as src:
            nlcd_ak = np.zeros((nrows, ncols), dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=nlcd_ak,
                dst_transform=dst_transform,
                dst_crs=CRS_ALASKA,
                resampling=Resampling.nearest,
            )

        # Map NLCD classes to friction
        friction = np.full((nrows, ncols), 1.2, dtype=np.float32)  # default
        for code, fval in NLCD_FRICTION.items():
            friction[nlcd_ak == code] = fval

        logger.info(f"LULC friction: min={friction.min():.2f}, max={friction.max():.2f}, "
                     f"mean={friction.mean():.2f}")
        return friction

    except Exception as e:
        logger.warning(f"Could not compute LULC friction: {e}. Using uniform.")
        return np.full((nrows, ncols), 1.2, dtype=np.float32)


def compute_road_network_friction(roads_path, grid):
    """Compute road network friction by rasterizing road geometries.

    Cells ON a road get friction based on road class (MTFCC code).
    Cells NOT on any road get high friction (8.0).

    Args:
        roads_path: Path to TIGER roads shapefile (or None)
        grid: Grid parameters dict

    Returns:
        np.ndarray: Road network friction raster
    """
    nrows, ncols = grid['nrows'], grid['ncols']
    no_road_val = DEFAULT_ROAD_CLASS_FRICTION['no_road']

    if roads_path is None:
        logger.warning("Roads not available, using uniform road friction = 1.0")
        return np.ones((nrows, ncols), dtype=np.float32)

    try:
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        roads = gpd.read_file(roads_path)
        if roads.crs != CRS_ALASKA:
            roads = roads.to_crs(CRS_ALASKA)

        transform = from_bounds(
            AK_BOUNDS_3338['xmin'], AK_BOUNDS_3338['ymin'],
            AK_BOUNDS_3338['xmax'], AK_BOUNDS_3338['ymax'],
            ncols, nrows
        )

        # Start with no_road friction everywhere
        friction = np.full((nrows, ncols), no_road_val, dtype=np.float32)

        # Rasterize each road class separately, worst (highest friction)
        # roads first so better roads overwrite them
        mtfcc_col = 'MTFCC' if 'MTFCC' in roads.columns else None
        if mtfcc_col is None:
            # Fall back: treat all roads as local (1.0)
            logger.warning("No MTFCC column in roads data, treating all as local roads")
            shapes = [(geom, 1.0) for geom in roads.geometry if geom is not None]
            if shapes:
                road_raster = rasterize(
                    shapes, out_shape=(nrows, ncols),
                    transform=transform, fill=0, dtype=np.float32
                )
                friction[road_raster > 0] = road_raster[road_raster > 0]
        else:
            # Sort by friction value descending so lower friction (better roads) wins
            road_classes = sorted(
                DEFAULT_ROAD_CLASS_FRICTION.items(),
                key=lambda x: x[1], reverse=True
            )
            for mtfcc, fval in road_classes:
                if mtfcc == 'no_road':
                    continue
                subset = roads[roads[mtfcc_col] == mtfcc]
                if subset.empty:
                    continue
                shapes = [(geom, fval) for geom in subset.geometry if geom is not None]
                if shapes:
                    layer = rasterize(
                        shapes, out_shape=(nrows, ncols),
                        transform=transform, fill=0, dtype=np.float32
                    )
                    mask = layer > 0
                    friction[mask] = layer[mask]

        on_road = np.sum(friction < no_road_val)
        total = nrows * ncols
        logger.info(f"Road friction: {on_road} cells on roads ({100*on_road/total:.1f}%), "
                     f"{total - on_road} off-road cells")
        return friction

    except Exception as e:
        logger.warning(f"Could not compute road friction: {e}. Using uniform.")
        return np.ones((nrows, ncols), dtype=np.float32)


def compute_permafrost_friction(permafrost_path, grid):
    """Compute permafrost friction from NSIDC data.

    Args:
        permafrost_path: Path to permafrost shapefile (or None)
        grid: Grid parameters dict

    Returns:
        np.ndarray: Permafrost friction raster
    """
    nrows, ncols = grid['nrows'], grid['ncols']

    if permafrost_path is None:
        logger.warning("Permafrost data not available, using uniform friction = 1.0")
        return np.ones((nrows, ncols), dtype=np.float32)

    try:
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        pf = gpd.read_file(permafrost_path)
        if pf.crs != CRS_ALASKA:
            pf = pf.to_crs(CRS_ALASKA)

        transform = from_bounds(
            AK_BOUNDS_3338['xmin'], AK_BOUNDS_3338['ymin'],
            AK_BOUNDS_3338['xmax'], AK_BOUNDS_3338['ymax'],
            ncols, nrows
        )

        friction = np.ones((nrows, ncols), dtype=np.float32)

        # Try to find permafrost extent column
        extent_col = None
        for col in ['EXTENT', 'PERMAFROST', 'extent', 'PF_EXTENT', 'NUM_CODE']:
            if col in pf.columns:
                extent_col = col
                break

        if extent_col is None:
            logger.warning("No permafrost extent column found, using uniform friction")
            return friction

        # Map extent categories to friction values
        extent_map = {
            'C': 'continuous', 'c': 'continuous', 'Continuous': 'continuous',
            'D': 'discontinuous', 'd': 'discontinuous', 'Discontinuous': 'discontinuous',
            'S': 'sporadic', 's': 'sporadic', 'Sporadic': 'sporadic',
            'I': 'isolated', 'i': 'isolated', 'Isolated': 'isolated',
        }

        for category, pf_key in set(extent_map.items()):
            fval = PERMAFROST_FRICTION.get(pf_key, 1.0)
            subset = pf[pf[extent_col].astype(str).str.startswith(category[0].upper())]
            if subset.empty:
                continue
            shapes = [(geom, fval) for geom in subset.geometry if geom is not None]
            if shapes:
                layer = rasterize(
                    shapes, out_shape=(nrows, ncols),
                    transform=transform, fill=0, dtype=np.float32
                )
                mask = layer > 0
                friction[mask] = layer[mask]

        logger.info(f"Permafrost friction: min={friction.min():.2f}, max={friction.max():.2f}")
        return friction

    except Exception as e:
        logger.warning(f"Could not compute permafrost friction: {e}. Using uniform.")
        return np.ones((nrows, ncols), dtype=np.float32)


def compute_waterway_proximity(waterways_path, grid):
    """Compute distance-to-waterway raster for Barge friction.

    Args:
        waterways_path: Path to NHD flowlines shapefile (or None)
        grid: Grid parameters dict

    Returns:
        np.ndarray: Waterway proximity raster (distance in cells, clamped)
    """
    nrows, ncols = grid['nrows'], grid['ncols']

    if waterways_path is None:
        logger.warning("Waterways not available, using uniform proximity")
        return np.full((nrows, ncols), 50.0, dtype=np.float32)

    try:
        from scipy.ndimage import distance_transform_edt
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        ww = gpd.read_file(waterways_path)
        if ww.crs != CRS_ALASKA:
            ww = ww.to_crs(CRS_ALASKA)

        transform = from_bounds(
            AK_BOUNDS_3338['xmin'], AK_BOUNDS_3338['ymin'],
            AK_BOUNDS_3338['xmax'], AK_BOUNDS_3338['ymax'],
            ncols, nrows
        )

        # Rasterize waterways as binary mask
        shapes = [(geom, 1) for geom in ww.geometry if geom is not None]
        if not shapes:
            return np.full((nrows, ncols), 50.0, dtype=np.float32)

        water_mask = rasterize(
            shapes, out_shape=(nrows, ncols),
            transform=transform, fill=0, dtype=np.uint8
        )

        # Distance transform: distance from each cell to nearest waterway
        dist = distance_transform_edt(water_mask == 0).astype(np.float32)
        dist = np.clip(dist, 0, 100)  # Cap at 100 cells = 100km

        logger.info(f"Waterway proximity: {np.sum(water_mask > 0)} waterway cells, "
                     f"mean distance={dist.mean():.1f} cells")
        return dist

    except Exception as e:
        logger.warning(f"Could not compute waterway proximity: {e}")
        return np.full((nrows, ncols), 50.0, dtype=np.float32)


def compute_airport_proximity(airports_path, grid):
    """Compute distance-to-airport raster for Plane friction.

    Args:
        airports_path: Path to FAA airports GeoJSON (or None)
        grid: Grid parameters dict

    Returns:
        np.ndarray: Airport proximity raster (distance in cells, clamped)
    """
    nrows, ncols = grid['nrows'], grid['ncols']

    if airports_path is None:
        logger.warning("Airports not available, using uniform proximity")
        return np.full((nrows, ncols), 50.0, dtype=np.float32)

    try:
        from scipy.ndimage import distance_transform_edt

        airports = gpd.read_file(airports_path)
        if airports.crs != CRS_ALASKA:
            airports = airports.to_crs(CRS_ALASKA)

        # Filter to Alaska airports (approximate bounding box)
        ak_airports = airports.cx[
            AK_BOUNDS_3338['xmin']:AK_BOUNDS_3338['xmax'],
            AK_BOUNDS_3338['ymin']:AK_BOUNDS_3338['ymax']
        ]

        if ak_airports.empty:
            logger.warning("No Alaska airports found in dataset")
            return np.full((nrows, ncols), 50.0, dtype=np.float32)

        # Mark airport cells
        airport_mask = np.zeros((nrows, ncols), dtype=np.uint8)
        for _, apt in ak_airports.iterrows():
            if apt.geometry is not None:
                x, y = apt.geometry.x, apt.geometry.y
                col = int((x - grid['xmin']) / grid['cell_size'])
                row = int((grid['ymax'] - y) / grid['cell_size'])
                if 0 <= row < nrows and 0 <= col < ncols:
                    airport_mask[row, col] = 1

        n_airports = np.sum(airport_mask > 0)
        if n_airports == 0:
            return np.full((nrows, ncols), 50.0, dtype=np.float32)

        dist = distance_transform_edt(airport_mask == 0).astype(np.float32)
        dist = np.clip(dist, 0, 100)

        logger.info(f"Airport proximity: {n_airports} airport cells, "
                     f"mean distance={dist.mean():.1f} cells")
        return dist

    except Exception as e:
        logger.warning(f"Could not compute airport proximity: {e}")
        return np.full((nrows, ncols), 50.0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Composite Friction Surfaces
# ---------------------------------------------------------------------------

def compute_composite_friction(base_layers, delivery_method, weights=None):
    """Combine base friction layers into a delivery-method-specific surface.

    Args:
        base_layers: dict with keys 'slope', 'lulc', 'road_network',
                     'permafrost', 'waterway_prox', 'airport_prox'
        delivery_method: 'Road', 'Barge', or 'Plane'
        weights: Optional custom weights dict (uses defaults if None)

    Returns:
        np.ndarray: Composite friction surface
    """
    if weights is None:
        weights = DEFAULT_DELIVERY_METHOD_WEIGHTS.get(delivery_method, {})

    slope = base_layers.get('slope', np.ones_like(base_layers['road_network']))
    lulc = base_layers.get('lulc', np.ones_like(base_layers['road_network']))
    road = base_layers.get('road_network', np.ones_like(slope))
    pf = base_layers.get('permafrost', np.ones_like(slope))
    ww_prox = base_layers.get('waterway_prox', np.full_like(slope, 50.0))
    apt_prox = base_layers.get('airport_prox', np.full_like(slope, 50.0))

    if delivery_method == 'Road':
        # Road friction: driven by road network, modified by slope and permafrost
        composite = (
            road ** weights.get('road_network_weight', 1.5) *
            slope ** weights.get('slope_weight', 1.0) *
            pf ** weights.get('permafrost_weight', 1.2) *
            lulc ** weights.get('lulc_weight', 0.5)
        )
        # Water cells are impassable barriers for trucks
        water_barrier = weights.get('water_barrier', 10.0)
        composite[lulc >= 5.0] = water_barrier

    elif delivery_method == 'Barge':
        # Barge: water is low friction, land is high
        water_friction = weights.get('water_navigable', 0.5)
        ww_weight = weights.get('waterway_proximity_weight', 1.5)

        # Base: high friction everywhere (land)
        composite = np.full_like(slope, 5.0, dtype=np.float32)
        # Near waterways: friction decreases
        # Normalize proximity to 0-1 range (0 = on waterway, 1 = far away)
        ww_norm = ww_prox / 100.0
        composite = water_friction + ww_norm * (5.0 - water_friction) * ww_weight
        # Permafrost affects port infrastructure
        composite *= pf ** weights.get('permafrost_weight', 0.5)

    elif delivery_method == 'Plane':
        # Plane: near-uniform base + airport proximity penalty
        base = weights.get('base_air_friction', 1.2)
        apt_weight = weights.get('airport_proximity_weight', 2.0)

        # Normalize airport proximity
        apt_norm = apt_prox / 100.0
        composite = np.full_like(slope, base, dtype=np.float32)
        composite += apt_norm * apt_weight
        # Slight slope and permafrost effects (for landing strips)
        composite *= slope ** weights.get('slope_weight', 0.1)
        composite *= pf ** weights.get('permafrost_weight', 0.3)

    else:
        # Unknown method: use road friction as fallback
        logger.warning(f"Unknown delivery method '{delivery_method}', using Road friction")
        return compute_composite_friction(base_layers, 'Road', weights)

    # Ensure minimum friction of 0.1 (no free travel)
    composite = np.maximum(composite, 0.1)

    logger.info(f"{delivery_method} composite: min={composite.min():.3f}, "
                 f"max={composite.max():.3f}, mean={composite.mean():.3f}")
    return composite.astype(np.float32)


# ---------------------------------------------------------------------------
# Pairwise Cost Computation
# ---------------------------------------------------------------------------

def haversine_distance(lon1, lat1, lon2, lat2):
    """Distance in miles between two (lon, lat) points."""
    R = 3959
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_pairwise_friction_costs(friction_surface, facilities, grid):
    """Compute pairwise friction costs between facilities using MCP.

    Uses scikit-image's MCP_Geometric for least-cost path computation
    on the friction surface.

    Args:
        friction_surface: 2D numpy array (composite friction for one method)
        facilities: list of (facility_id, lon, lat) tuples
        grid: Grid parameters dict

    Returns:
        list of (src_id, dst_id, haversine_mi, friction_cost, friction_ratio)
    """
    try:
        from skimage.graph import MCP_Geometric
    except ImportError:
        logger.error("scikit-image required for MCP. Install with: pip install scikit-image")
        raise

    nrows, ncols = grid['nrows'], grid['ncols']

    # Convert facility locations to row/col
    facility_pixels = []
    for fid, lon, lat in facilities:
        row, col = lonlat_to_rowcol(lon, lat, grid)
        row = max(0, min(row, nrows - 1))
        col = max(0, min(col, ncols - 1))
        facility_pixels.append((fid, lon, lat, row, col))

    if len(facility_pixels) < 2:
        return []

    results = []
    mcp = MCP_Geometric(friction_surface, fully_connected=True)

    for i, (fid_a, lon_a, lat_a, row_a, col_a) in enumerate(facility_pixels):
        # Compute cumulative cost from this facility to all others
        starts = [(row_a, col_a)]
        cumulative_costs, _ = mcp.find_costs(starts)

        for j in range(i + 1, len(facility_pixels)):
            fid_b, lon_b, lat_b, row_b, col_b = facility_pixels[j]

            friction_cost = float(cumulative_costs[row_b, col_b])
            haver_mi = haversine_distance(lon_a, lat_a, lon_b, lat_b)

            # Friction ratio: how much longer than straight line
            ratio = friction_cost / haver_mi if haver_mi > 0 else 1.0

            results.append((fid_a, fid_b, haver_mi, friction_cost, ratio))

    return results


# ---------------------------------------------------------------------------
# DuckDB Schema and Storage
# ---------------------------------------------------------------------------

def init_friction_tables(con):
    """Create friction-specific tables in DuckDB.

    Args:
        con: DuckDB connection
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS friction_layers (
            layer_id INTEGER PRIMARY KEY,
            layer_name VARCHAR,
            source_dataset VARCHAR,
            resolution_m INTEGER,
            crs VARCHAR,
            file_path VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS friction_weights (
            weight_id INTEGER PRIMARY KEY,
            delivery_method VARCHAR,
            layer_name VARCHAR,
            category VARCHAR,
            friction_value DOUBLE,
            adjusted_by_agent BOOLEAN DEFAULT FALSE,
            adjustment_reason VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS friction_costs (
            src INTEGER,
            dst INTEGER,
            delivery_method VARCHAR,
            haversine_miles DOUBLE,
            friction_cost DOUBLE,
            friction_ratio DOUBLE,
            PRIMARY KEY (src, dst, delivery_method)
        )
    """)

    # Add friction_cost column to connects_to if not already there
    try:
        con.execute("ALTER TABLE connects_to ADD COLUMN friction_cost DOUBLE")
        logger.info("Added friction_cost column to connects_to table")
    except Exception:
        # Column already exists
        pass

    logger.info("Friction tables initialized in DuckDB")


def store_friction_weights(con, delivery_method, weights):
    """Store friction weights for a delivery method in DuckDB.

    Args:
        con: DuckDB connection
        delivery_method: e.g., 'Road', 'Barge', 'Plane'
        weights: dict of weight_name -> value
    """
    # Get next weight_id
    max_id = con.execute(
        "SELECT COALESCE(MAX(weight_id), 0) FROM friction_weights"
    ).fetchone()[0]

    for i, (name, value) in enumerate(weights.items()):
        con.execute(
            "INSERT INTO friction_weights VALUES (?, ?, ?, ?, ?, FALSE, NULL) "
            "ON CONFLICT DO NOTHING",
            [max_id + i + 1, delivery_method, name, 'weight', value]
        )


def store_friction_layer_info(con, layers_info):
    """Store friction layer metadata in DuckDB.

    Args:
        con: DuckDB connection
        layers_info: list of (layer_name, source, resolution, crs, path)
    """
    max_id = con.execute(
        "SELECT COALESCE(MAX(layer_id), 0) FROM friction_layers"
    ).fetchone()[0]

    for i, (name, source, res, crs, path) in enumerate(layers_info):
        con.execute(
            "INSERT INTO friction_layers VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [max_id + i + 1, name, source, res, crs, str(path)]
        )


def store_friction_costs(con, costs, delivery_method):
    """Store pairwise friction costs in DuckDB.

    Args:
        con: DuckDB connection
        costs: list of (src, dst, haversine_mi, friction_cost, ratio)
        delivery_method: Delivery method string
    """
    for src, dst, haver, fcost, ratio in costs:
        # friction_costs table (both directions)
        con.execute(
            "INSERT INTO friction_costs VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (src, dst, delivery_method) DO UPDATE SET "
            "friction_cost = EXCLUDED.friction_cost, friction_ratio = EXCLUDED.friction_ratio",
            [src, dst, delivery_method, haver, fcost, ratio]
        )
        con.execute(
            "INSERT INTO friction_costs VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (src, dst, delivery_method) DO UPDATE SET "
            "friction_cost = EXCLUDED.friction_cost, friction_ratio = EXCLUDED.friction_ratio",
            [dst, src, delivery_method, haver, fcost, ratio]
        )

        # Also update connects_to table
        con.execute(
            "UPDATE connects_to SET friction_cost = ? WHERE src = ? AND dst = ?",
            [fcost, src, dst]
        )
        con.execute(
            "UPDATE connects_to SET friction_cost = ? WHERE src = ? AND dst = ?",
            [fcost, dst, src]
        )

    logger.info(f"Stored {len(costs)} friction cost pairs for {delivery_method}")


def get_facilities_by_group(con):
    """Get facilities grouped by (region, delivery_method) from DuckDB.

    Returns:
        dict: {(region, method): [(facility_id, lon, lat), ...]}
    """
    rows = con.execute("""
        SELECT f.facility_id, f.longitude, f.latitude,
               li.region_name,
               COALESCE(um.method_name, 'Unknown') AS method
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        WHERE li.region_name != 'Unassigned'
        ORDER BY li.region_name, method
    """).fetchall()

    groups = defaultdict(list)
    for fid, lon, lat, region, method in rows:
        groups[(region, method)].append((fid, lon, lat))

    return dict(groups)


# ---------------------------------------------------------------------------
# CrewAI Tools
# ---------------------------------------------------------------------------

@tool("download_dataset")
def download_dataset_tool(dataset_key: str) -> str:
    """Download and validate a geospatial dataset for friction computation.

    Args:
        dataset_key: One of: dem, nlcd, roads, waterways, airports, ports, permafrost
    """
    if dataset_key not in DATASETS:
        return f"Unknown dataset: {dataset_key}. Valid keys: {list(DATASETS.keys())}"
    try:
        path = download_dataset(dataset_key)
        ds = DATASETS[dataset_key]
        return json.dumps({
            'status': 'success',
            'dataset': ds['name'],
            'path': path,
            'source': ds['source'],
            'purpose': ds['purpose'],
        })
    except Exception as e:
        return json.dumps({'status': 'failed', 'error': str(e)})


@tool("get_friction_weights")
def get_friction_weights_tool(delivery_method: str) -> str:
    """Get current friction weights for a delivery method.

    Args:
        delivery_method: One of Road, Barge, Plane
    """
    weights = DEFAULT_DELIVERY_METHOD_WEIGHTS.get(delivery_method, {})

    # Also check DB for agent-adjusted weights
    try:
        con = pipeline.get_duckdb_connection(DB_PATH, read_only=True)
        rows = con.execute(
            "SELECT layer_name, friction_value, adjusted_by_agent, adjustment_reason "
            "FROM friction_weights WHERE delivery_method = ?",
            [delivery_method]
        ).fetchall()
        con.close()

        db_weights = {}
        for name, val, adjusted, reason in rows:
            db_weights[name] = {
                'value': val,
                'adjusted_by_agent': adjusted,
                'reason': reason,
            }

        return json.dumps({
            'delivery_method': delivery_method,
            'default_weights': weights,
            'db_weights': db_weights,
        }, indent=2)
    except Exception:
        return json.dumps({
            'delivery_method': delivery_method,
            'default_weights': weights,
        }, indent=2)


@tool("adjust_friction_weight")
def adjust_friction_weight_tool(delivery_method: str, weight_name: str,
                                new_value: float, reason: str) -> str:
    """Adjust a friction weight value with justification.

    Args:
        delivery_method: Road, Barge, or Plane
        weight_name: Name of the weight to adjust (e.g., 'slope_weight')
        new_value: New friction weight value
        reason: Justification for the adjustment
    """
    try:
        con = pipeline.get_duckdb_connection(DB_PATH)

        # Check if weight exists
        existing = con.execute(
            "SELECT weight_id FROM friction_weights "
            "WHERE delivery_method = ? AND layer_name = ?",
            [delivery_method, weight_name]
        ).fetchone()

        if existing:
            con.execute(
                "UPDATE friction_weights SET friction_value = ?, "
                "adjusted_by_agent = TRUE, adjustment_reason = ? "
                "WHERE delivery_method = ? AND layer_name = ?",
                [new_value, reason, delivery_method, weight_name]
            )
        else:
            max_id = con.execute(
                "SELECT COALESCE(MAX(weight_id), 0) FROM friction_weights"
            ).fetchone()[0]
            con.execute(
                "INSERT INTO friction_weights VALUES (?, ?, ?, 'weight', ?, TRUE, ?)",
                [max_id + 1, delivery_method, weight_name, new_value, reason]
            )

        con.close()
        return json.dumps({
            'status': 'adjusted',
            'delivery_method': delivery_method,
            'weight_name': weight_name,
            'new_value': new_value,
            'reason': reason,
        })
    except Exception as e:
        return json.dumps({'status': 'error', 'error': str(e)})


@tool("compute_friction_surface")
def compute_friction_surface_tool(delivery_method: str) -> str:
    """Compute the composite friction surface for a delivery method.

    Args:
        delivery_method: Road, Barge, or Plane
    """
    try:
        grid = make_grid_params()

        # Load base layers (use cached data paths)
        data_paths = {}
        for key in DATASETS:
            try:
                path = download_dataset(key)
                data_paths[key] = path
            except Exception:
                data_paths[key] = None

        base_layers = {
            'slope': compute_slope_friction(data_paths.get('dem'), grid),
            'lulc': compute_lulc_friction(data_paths.get('nlcd'), grid),
            'road_network': compute_road_network_friction(data_paths.get('roads'), grid),
            'permafrost': compute_permafrost_friction(data_paths.get('permafrost'), grid),
            'waterway_prox': compute_waterway_proximity(data_paths.get('waterways'), grid),
            'airport_prox': compute_airport_proximity(data_paths.get('airports'), grid),
        }

        composite = compute_composite_friction(base_layers, delivery_method)

        return json.dumps({
            'status': 'computed',
            'delivery_method': delivery_method,
            'shape': list(composite.shape),
            'min': float(composite.min()),
            'max': float(composite.max()),
            'mean': float(composite.mean()),
            'std': float(composite.std()),
        })
    except Exception as e:
        return json.dumps({'status': 'error', 'error': str(e)})


@tool("compute_pairwise_costs")
def compute_pairwise_costs_tool(delivery_method: str) -> str:
    """Compute pairwise friction costs between facilities for a delivery method.

    Args:
        delivery_method: Road, Barge, or Plane
    """
    try:
        con = pipeline.get_duckdb_connection(DB_PATH)
        grid = make_grid_params()

        groups = get_facilities_by_group(con)
        method_groups = {k: v for k, v in groups.items() if k[1] == delivery_method}

        if not method_groups:
            con.close()
            return json.dumps({
                'status': 'no_groups',
                'message': f'No facility groups found for {delivery_method}'
            })

        # Load and compute friction surface
        data_paths = {}
        for key in DATASETS:
            try:
                path = download_dataset(key)
                data_paths[key] = path
            except Exception:
                data_paths[key] = None

        base_layers = {
            'slope': compute_slope_friction(data_paths.get('dem'), grid),
            'lulc': compute_lulc_friction(data_paths.get('nlcd'), grid),
            'road_network': compute_road_network_friction(data_paths.get('roads'), grid),
            'permafrost': compute_permafrost_friction(data_paths.get('permafrost'), grid),
            'waterway_prox': compute_waterway_proximity(data_paths.get('waterways'), grid),
            'airport_prox': compute_airport_proximity(data_paths.get('airports'), grid),
        }

        composite = compute_composite_friction(base_layers, delivery_method)
        init_friction_tables(con)

        total_pairs = 0
        for (region, method), facilities in method_groups.items():
            if len(facilities) < 2:
                continue

            costs = compute_pairwise_friction_costs(composite, facilities, grid)
            store_friction_costs(con, costs, delivery_method)
            total_pairs += len(costs)

        con.close()
        return json.dumps({
            'status': 'computed',
            'delivery_method': delivery_method,
            'total_pairs': total_pairs,
            'num_groups': len(method_groups),
        })
    except Exception as e:
        return json.dumps({'status': 'error', 'error': str(e)})


@tool("validate_friction_costs")
def validate_friction_costs_tool(delivery_method: str) -> str:
    """Validate friction costs for a delivery method - check for issues.

    Args:
        delivery_method: Road, Barge, or Plane
    """
    try:
        con = pipeline.get_duckdb_connection(DB_PATH, read_only=True)

        costs = con.execute(
            "SELECT src, dst, haversine_miles, friction_cost, friction_ratio "
            "FROM friction_costs WHERE delivery_method = ?",
            [delivery_method]
        ).fetchall()

        if not costs:
            con.close()
            return json.dumps({
                'status': 'no_data',
                'message': f'No friction costs found for {delivery_method}'
            })

        ratios = [r[4] for r in costs if r[4] is not None]
        fcosts = [r[3] for r in costs if r[3] is not None]
        haversines = [r[2] for r in costs]

        issues = []

        # Check 1: friction cost should be >= haversine
        below_haversine = [(r[0], r[1], r[4]) for r in costs
                           if r[3] is not None and r[3] < r[2] * 0.8]
        if below_haversine:
            issues.append({
                'type': 'below_haversine',
                'count': len(below_haversine),
                'samples': below_haversine[:5],
                'severity': 'warning',
            })

        # Check 2: no NaN/inf
        nan_count = sum(1 for r in costs if r[3] is None or
                        (isinstance(r[3], float) and (math.isnan(r[3]) or math.isinf(r[3]))))
        if nan_count > 0:
            issues.append({
                'type': 'nan_inf_values',
                'count': nan_count,
                'severity': 'critical',
            })

        # Check 3: ratios in plausible bounds
        if ratios:
            high_ratio = [(r[0], r[1], r[4]) for r in costs
                          if r[4] is not None and r[4] > 10.0]
            low_ratio = [(r[0], r[1], r[4]) for r in costs
                         if r[4] is not None and r[4] < 0.5]

            if high_ratio:
                issues.append({
                    'type': 'high_ratio_outliers',
                    'count': len(high_ratio),
                    'samples': high_ratio[:5],
                    'severity': 'warning',
                    'suggestion': 'Consider reducing friction weights for affected areas',
                })
            if low_ratio:
                issues.append({
                    'type': 'low_ratio_outliers',
                    'count': len(low_ratio),
                    'samples': low_ratio[:5],
                    'severity': 'warning',
                    'suggestion': 'Friction cost unusually low - check surface computation',
                })

        # Check 4: coverage
        expected = con.execute(
            "SELECT COUNT(*) FROM connects_to ct "
            "JOIN facilities f1 ON ct.src = f1.facility_id "
            "JOIN uses_method um ON f1.facility_id = um.facility_id "
            "WHERE um.method_name = ?",
            [delivery_method]
        ).fetchone()[0]

        actual = len(costs) // 2  # divide by 2 since we store both directions
        if expected > 0 and actual < expected * 0.9:
            issues.append({
                'type': 'coverage_gap',
                'expected_pairs': expected,
                'actual_pairs': actual,
                'coverage_pct': round(100 * actual / expected, 1),
                'severity': 'warning',
            })

        con.close()

        pass_fail = 'PASS' if not any(i['severity'] == 'critical' for i in issues) else 'FAIL'

        return json.dumps({
            'delivery_method': delivery_method,
            'status': pass_fail,
            'total_pairs': len(costs),
            'ratio_stats': {
                'min': round(min(ratios), 3) if ratios else None,
                'max': round(max(ratios), 3) if ratios else None,
                'mean': round(sum(ratios) / len(ratios), 3) if ratios else None,
                'median': round(sorted(ratios)[len(ratios) // 2], 3) if ratios else None,
            },
            'issues': issues,
            'num_issues': len(issues),
        }, indent=2)
    except Exception as e:
        return json.dumps({'status': 'error', 'error': str(e)})


@tool("get_friction_outliers")
def get_friction_outliers_tool(delivery_method: str, threshold: float = 5.0) -> str:
    """Get facility pairs with friction ratios above threshold.

    Args:
        delivery_method: Road, Barge, or Plane
        threshold: Friction ratio threshold (default 5.0)
    """
    try:
        con = pipeline.get_duckdb_connection(DB_PATH, read_only=True)
        outliers = con.execute(
            "SELECT fc.src, fc.dst, fc.haversine_miles, fc.friction_cost, "
            "fc.friction_ratio, f1.community_name AS src_community, "
            "f2.community_name AS dst_community "
            "FROM friction_costs fc "
            "JOIN facilities f1 ON fc.src = f1.facility_id "
            "JOIN facilities f2 ON fc.dst = f2.facility_id "
            "WHERE fc.delivery_method = ? AND fc.friction_ratio > ? "
            "ORDER BY fc.friction_ratio DESC LIMIT 20",
            [delivery_method, threshold]
        ).fetchall()
        con.close()

        results = []
        for src, dst, haver, fcost, ratio, src_comm, dst_comm in outliers:
            results.append({
                'src': src, 'dst': dst,
                'src_community': src_comm, 'dst_community': dst_comm,
                'haversine_miles': round(haver, 1),
                'friction_cost': round(fcost, 1),
                'friction_ratio': round(ratio, 3),
            })

        return json.dumps({
            'delivery_method': delivery_method,
            'threshold': threshold,
            'num_outliers': len(results),
            'outliers': results,
        }, indent=2)
    except Exception as e:
        return json.dumps({'status': 'error', 'error': str(e)})


@tool("query_friction_summary")
def query_friction_summary_tool() -> str:
    """Get a summary of all friction costs across delivery methods."""
    try:
        con = pipeline.get_duckdb_connection(DB_PATH, read_only=True)
        summary = con.execute("""
            SELECT delivery_method,
                   COUNT(*) AS num_pairs,
                   ROUND(AVG(friction_ratio), 3) AS avg_ratio,
                   ROUND(MIN(friction_ratio), 3) AS min_ratio,
                   ROUND(MAX(friction_ratio), 3) AS max_ratio,
                   ROUND(STDDEV(friction_ratio), 3) AS std_ratio,
                   ROUND(AVG(haversine_miles), 1) AS avg_haversine,
                   ROUND(AVG(friction_cost), 1) AS avg_friction_cost
            FROM friction_costs
            GROUP BY delivery_method
            ORDER BY delivery_method
        """).fetchall()
        con.close()

        results = []
        for row in summary:
            results.append({
                'delivery_method': row[0],
                'num_pairs': row[1],
                'avg_ratio': row[2],
                'min_ratio': row[3],
                'max_ratio': row[4],
                'std_ratio': row[5],
                'avg_haversine_mi': row[6],
                'avg_friction_cost': row[7],
            })

        return json.dumps({'summary': results}, indent=2)
    except Exception as e:
        return json.dumps({'status': 'error', 'error': str(e)})


# ---------------------------------------------------------------------------
# CrewAI Agents and Tasks
# ---------------------------------------------------------------------------

def create_friction_crew():
    """Create the friction layer CrewAI crew with 3 agents and 5 tasks.

    Returns:
        Crew: Configured CrewAI crew
    """
    llm = pipeline.get_llm()

    # --- Agents ---

    data_acquisition_agent = Agent(
        role="Geospatial Data Acquisition Specialist",
        goal="Download and validate all required geospatial datasets for friction "
             "surface computation, ensuring data quality and completeness.",
        backstory="""Expert in geospatial data acquisition with deep knowledge of
        USGS, Census, FAA, USACE, and NSIDC data portals. Ensures all datasets are
        properly downloaded, extracted, and validated before friction computation
        begins. Understands Alaska-specific data challenges including large file
        sizes, projection systems, and data gaps in remote areas.""",
        verbose=True,
        llm=llm,
        tools=[download_dataset_tool],
    )

    friction_modeler_agent = Agent(
        role="Friction Surface Modeler",
        goal="Compute accurate delivery-method-specific friction surfaces and "
             "pairwise facility costs. Adjust friction weights based on validation "
             "feedback to improve cost accuracy.",
        backstory="""Expert in geospatial cost surface analysis with specialization
        in Alaska transportation networks. Understands road class hierarchies,
        waterway navigability, airport accessibility, and permafrost impacts on
        infrastructure. Builds composite friction surfaces that realistically model
        the cost of moving fuel between facilities by different delivery methods.
        When given validation feedback, systematically adjusts weights and recomputes
        to resolve identified issues.""",
        verbose=True,
        llm=llm,
        tools=[
            get_friction_weights_tool,
            adjust_friction_weight_tool,
            compute_friction_surface_tool,
            compute_pairwise_costs_tool,
        ],
    )

    validation_agent = Agent(
        role="Friction Cost Validation Analyst",
        goal="Validate computed friction costs for physical plausibility. Identify "
             "outliers, coverage gaps, and systematic biases. Provide actionable "
             "recommendations for the Friction Modeler to fix issues.",
        backstory="""Expert in transportation cost validation with knowledge of
        Alaska's geography and fuel delivery logistics. Knows expected cost ratios
        for different delivery methods and regions. Identifies when friction costs
        are implausible (too high or too low) and traces issues back to specific
        friction weights or data gaps. Produces structured recommendations that
        the Friction Modeler can act on directly.""",
        verbose=True,
        llm=llm,
        tools=[
            validate_friction_costs_tool,
            get_friction_outliers_tool,
            query_friction_summary_tool,
        ],
    )

    # --- Tasks ---

    task1_data_acquisition = Task(
        description="""Download and validate all 7 geospatial datasets required for
        friction surface computation:
        1. DEM (USGS 3DEP) - for slope computation
        2. Land Cover (NLCD 2021) - for terrain friction
        3. Roads (Census TIGER Alaska) - for road network
        4. Waterways (USGS NHD Alaska) - for barge routes
        5. Airports (FAA NASR) - for plane access
        6. Ports (USACE) - for barge access
        7. Permafrost (NSIDC) - for ground stability

        For each dataset, use the download_dataset tool with the appropriate key.
        Report the status of each download including file path and any issues.""",
        agent=data_acquisition_agent,
        expected_output="""A JSON report listing each dataset with:
        - Download status (success/failed)
        - File path
        - File size
        - Any data quality concerns""",
    )

    task2_compute_initial = Task(
        description="""Using the downloaded datasets, compute friction surfaces and
        pairwise costs for all three delivery methods: Road, Barge, and Plane.

        For each delivery method:
        1. Review the current friction weights using get_friction_weights
        2. Compute the composite friction surface using compute_friction_surface
        3. Compute pairwise costs between facilities using compute_pairwise_costs

        Report statistics for each method including surface statistics and
        number of facility pairs processed.""",
        agent=friction_modeler_agent,
        context=[task1_data_acquisition],
        expected_output="""A report for each delivery method containing:
        - Friction surface statistics (min, max, mean, std)
        - Number of facility pairs computed
        - Current weight values used""",
    )

    task3_validate = Task(
        description="""Validate the computed friction costs for all delivery methods.
        For each method (Road, Barge, Plane):

        1. Use validate_friction_costs to check for:
           - Costs below Haversine distance (physically impossible for roads)
           - NaN/infinity values
           - Friction ratios outside plausible bounds
           - Coverage gaps (missing facility pairs)

        2. Use get_friction_outliers to identify extreme cases

        3. Use query_friction_summary for cross-method comparison

        Produce a structured recommendations report with:
        - Specific outlier pairs (src, dst, method) with explanation
        - Coverage gaps and which regions are affected
        - Weight adjustment suggestions with specific values
        - Layer-specific issues (e.g., missing ice roads in TIGER data)
        - Pass/fail status per delivery method

        Expected bounds:
        - Road: friction ratios typically 1.5-3.0x (up to 5x in remote areas)
        - Barge: friction ratios 0.8-2.0x (water shortcuts possible)
        - Plane: friction ratios 1.1-1.5x (nearly straight-line)""",
        agent=validation_agent,
        context=[task2_compute_initial],
        expected_output="""A structured validation report in JSON format with:
        - Per-method pass/fail status
        - Outlier pairs with specific (src, dst) and ratios
        - Coverage gap details
        - Weight adjustment recommendations (method, weight_name, current_value, suggested_value, reason)
        - Layer-specific issues
        - Overall assessment""",
    )

    task4_fix_friction = Task(
        description="""Review the validation report from the previous task and apply
        recommended fixes to the friction weights and surfaces.

        For each recommendation:
        1. Use adjust_friction_weight to update the weight value, providing the
           validation recommendation as the reason
        2. Use compute_friction_surface to recompute the affected surface
        3. Use compute_pairwise_costs to recompute costs for the affected method

        Only adjust weights that were specifically flagged. Do not make speculative
        changes. Log every adjustment with clear justification.

        If a recommendation suggests data is missing (e.g., ice roads), note it
        as a known limitation rather than trying to fix it through weight adjustment.""",
        agent=friction_modeler_agent,
        context=[task3_validate],
        expected_output="""A report of all adjustments made:
        - Each weight change (method, weight_name, old_value, new_value, reason)
        - Recomputed surface statistics
        - Known limitations noted but not fixable through weights
        - Summary of changes""",
    )

    task5_final_validation = Task(
        description="""Perform a final validation pass after the Friction Modeler
        has applied fixes. For each delivery method:

        1. Use validate_friction_costs to recheck all metrics
        2. Use query_friction_summary for updated cross-method comparison
        3. Compare results against initial validation to confirm improvements

        Report:
        - Which issues were resolved
        - Which issues persist (with explanation)
        - Final pass/fail per method
        - Overall confidence level in the friction costs

        If critical issues persist, report them clearly but do not trigger
        another iteration. The pipeline will proceed with the best available costs.""",
        agent=validation_agent,
        context=[task4_fix_friction],
        expected_output="""A final validation report with:
        - Per-method pass/fail status
        - Comparison with initial validation (improvements/regressions)
        - Remaining issues and their severity
        - Overall confidence level (High/Medium/Low)
        - Recommendation on whether to proceed or flag for manual review""",
    )

    # --- Crew ---

    crew = Crew(
        agents=[data_acquisition_agent, friction_modeler_agent, validation_agent],
        tasks=[
            task1_data_acquisition,
            task2_compute_initial,
            task3_validate,
            task4_fix_friction,
            task5_final_validation,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew


# ---------------------------------------------------------------------------
# Main Entry Points
# ---------------------------------------------------------------------------

def prepare_friction_data():
    """Step 1: Download all geospatial datasets (no DB needed).

    Can be run before regionalization to pre-cache data.
    """
    logger.info("=" * 60)
    logger.info("FRICTION LAYER: Downloading geospatial datasets...")
    logger.info("=" * 60)

    data_paths, status = download_all_datasets()

    successful = sum(1 for s in status.values() if s['status'] == 'success')
    logger.info(f"\nDownloaded {successful}/{len(DATASETS)} datasets successfully")

    return data_paths, status


def compute_and_store_costs(data_paths=None):
    """Step 2: Compute friction surfaces and pairwise costs, store in DB.

    Requires regionalization to have been run first (DB must exist with
    facilities populated).

    Args:
        data_paths: Optional dict of {dataset_key: filepath}. If None,
                    will attempt to download/use cached data.
    """
    logger.info("=" * 60)
    logger.info("FRICTION LAYER: Computing friction surfaces and costs...")
    logger.info("=" * 60)

    # Connect to existing DB
    con = pipeline.get_duckdb_connection(DB_PATH)
    init_friction_tables(con)

    # Get grid parameters
    grid = make_grid_params()
    logger.info(f"Grid: {grid['nrows']} x {grid['ncols']} cells at {CELL_SIZE_M}m")

    # Load or download data
    if data_paths is None:
        data_paths = {}
        for key in DATASETS:
            try:
                data_paths[key] = download_dataset(key)
            except Exception:
                data_paths[key] = None

    # Store layer metadata
    layers_info = []
    for key, path in data_paths.items():
        if path:
            ds = DATASETS[key]
            layers_info.append((
                ds['name'], ds['source'], CELL_SIZE_M, CRS_ALASKA, path
            ))
    store_friction_layer_info(con, layers_info)

    # Compute base friction layers
    logger.info("\nComputing base friction layers...")
    base_layers = {
        'slope': compute_slope_friction(data_paths.get('dem'), grid),
        'lulc': compute_lulc_friction(data_paths.get('nlcd'), grid),
        'road_network': compute_road_network_friction(data_paths.get('roads'), grid),
        'permafrost': compute_permafrost_friction(data_paths.get('permafrost'), grid),
        'waterway_prox': compute_waterway_proximity(data_paths.get('waterways'), grid),
        'airport_prox': compute_airport_proximity(data_paths.get('airports'), grid),
    }

    # Get facility groups
    groups = get_facilities_by_group(con)
    if not groups:
        logger.warning("No facility groups found in DB. Has regionalization been run?")
        con.close()
        return

    # Store default weights
    for method, weights in DEFAULT_DELIVERY_METHOD_WEIGHTS.items():
        store_friction_weights(con, method, weights)

    # Compute composite surfaces and pairwise costs per delivery method
    methods_seen = set()
    for (region, method), facilities in groups.items():
        methods_seen.add(method)

    for method in methods_seen:
        logger.info(f"\n--- {method} ---")

        # Get custom weights if agent has adjusted them
        weights = dict(DEFAULT_DELIVERY_METHOD_WEIGHTS.get(method, {}))
        try:
            adjusted = con.execute(
                "SELECT layer_name, friction_value FROM friction_weights "
                "WHERE delivery_method = ? AND adjusted_by_agent = TRUE",
                [method]
            ).fetchall()
            for name, val in adjusted:
                weights[name] = val
        except Exception:
            pass

        composite = compute_composite_friction(base_layers, method, weights)

        # Compute pairwise costs for all groups of this method
        method_groups = {k: v for k, v in groups.items() if k[1] == method}
        total_pairs = 0

        for (region, _), facilities in method_groups.items():
            if len(facilities) < 2:
                continue

            logger.info(f"  {region}: {len(facilities)} facilities")
            costs = compute_pairwise_friction_costs(composite, facilities, grid)
            store_friction_costs(con, costs, method)
            total_pairs += len(costs)

        logger.info(f"  Total pairs for {method}: {total_pairs}")

    # Print summary
    summary = con.execute("""
        SELECT delivery_method,
               COUNT(*) AS pairs,
               ROUND(AVG(friction_ratio), 3) AS avg_ratio,
               ROUND(MIN(friction_ratio), 3) AS min_ratio,
               ROUND(MAX(friction_ratio), 3) AS max_ratio
        FROM friction_costs
        GROUP BY delivery_method
    """).fetchall()

    logger.info("\n" + "=" * 60)
    logger.info("Friction Cost Summary:")
    for method, pairs, avg, mn, mx in summary:
        logger.info(f"  {method}: {pairs} pairs, avg_ratio={avg}, "
                     f"range=[{mn}, {mx}]")
    logger.info("=" * 60)

    con.close()


def main():
    """Main entry point: run full friction layer pipeline with CrewAI agents.

    This is called as Step 2 in run_graph.py, after regionalization.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Friction Layer Module")
    logger.info("=" * 60)

    # Initialize friction tables in DB
    con = pipeline.get_duckdb_connection(DB_PATH)
    init_friction_tables(con)
    con.close()

    # Run CrewAI crew (handles data download, computation, validation loop)
    crew = create_friction_crew()
    result = crew.kickoff()

    logger.info("\n" + "=" * 60)
    logger.info("FRICTION LAYER CREW COMPLETE")
    logger.info("=" * 60)
    logger.info(str(result))

    # Verify results in DB
    con = pipeline.get_duckdb_connection(DB_PATH, read_only=True)
    try:
        cost_count = con.execute(
            "SELECT COUNT(*) FROM friction_costs"
        ).fetchone()[0]
        logger.info(f"Total friction costs in DB: {cost_count}")

        updated_connects = con.execute(
            "SELECT COUNT(*) FROM connects_to WHERE friction_cost IS NOT NULL"
        ).fetchone()[0]
        logger.info(f"connects_to edges with friction cost: {updated_connects}")
    except Exception as e:
        logger.warning(f"Could not verify results: {e}")
    finally:
        con.close()

    return result


if __name__ == "__main__":
    pipeline.set_cwd('/media/volume/Preliminary_mas_runs')
    main()
