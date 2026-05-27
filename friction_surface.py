"""friction_surface.py

Deterministic, agent-free friction-surface construction.

Builds 36 mode-specific monthly friction rasters from preprocessed inputs
(slope, LULC, permafrost, monthly sea ice, monthly river ice) for three
ground modes: overland, barge, ice_road.

Design principles
-----------------
1. LULC and permafrost are independent factors. LULC enters a year-round
   static base; permafrost is a seasonal modifier (1.0 in winter, scales to
   PERMAFROST_MAX_SHOULDER in May/Oct, PERMAFROST_MAX_SUMMER in Jun-Sep).
2. NoData is the sole impassability mechanism. There is no sentinel value.
   WhiteboxTools CostDistance routes around NoData pixels.
3. 12 monthly surfaces per mode (36 total). Sea ice and river ice are
   applied per-month.

No CrewAI, no LLM calls. Pure numpy / rasterio.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Editable constants
# ---------------------------------------------------------------------------

SLOPE_THRESHOLDS = (2.0, 8.0)             # degrees
SLOPE_FRICTION = (1.0, 1.4, 1.75)         # <2, 2-8, >=8

# Dynamic World modal class codes (integer keys). value=None marks NoData
# in the overland base; mode-specific surfaces handle water themselves.
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

PERMAFROST_WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
PERMAFROST_SHOULDER_MONTHS = {5, 10}
PERMAFROST_SUMMER_MONTHS = {6, 7, 8, 9}
PERMAFROST_MAX_SHOULDER = 1.15            # at 100% permafrost in May / Oct
PERMAFROST_MAX_SUMMER = 1.40              # at 100% permafrost in Jun-Sep

ICE_PROB_THRESHOLD = 0.5                  # combined sea/river ice probability
WATER_FRICTION_BARGE = 0.5                # ice-free water under barge mode
WATER_FRICTION_ICEROAD = 0.8              # high-ice water under ice_road mode

FRICTION_NODATA = -9999.0

MODES = ("overland", "barge", "ice_road")


# ---------------------------------------------------------------------------
# Single-factor reclassifications
# ---------------------------------------------------------------------------

def compute_slope_friction(slope_path: str | Path) -> tuple[np.ndarray, dict]:
    """Reclassify a slope raster (degrees) into a friction array.

    Returns (friction, profile). Profile is from the source raster and is
    the canonical grid for downstream stacking.
    """
    with rasterio.open(slope_path) as src:
        slope = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    lo, hi = SLOPE_THRESHOLDS
    f_flat, f_roll, f_mtn = SLOPE_FRICTION

    friction = np.full(slope.shape, f_flat, dtype=np.float32)
    friction[(slope >= lo) & (slope < hi)] = f_roll
    friction[slope >= hi] = f_mtn
    return friction, profile


def compute_lulc_friction(
    lulc_path: str | Path,
    lulc_lookup: dict[int, float | None] = LULC_FRICTION,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reclassify a LULC raster into friction values.

    Returns (friction, water_mask, profile). Water pixels (class
    LULC_WATER_CLASS or any class mapped to None) get FRICTION_NODATA in
    the friction array; water_mask is True at those pixels.
    """
    with rasterio.open(lulc_path) as src:
        lulc = src.read(1)
        profile = src.profile.copy()

    friction = np.full(lulc.shape, FRICTION_NODATA, dtype=np.float32)
    water_mask = np.zeros(lulc.shape, dtype=bool)

    for cls, value in lulc_lookup.items():
        cls_pixels = lulc == cls
        if value is None:
            water_mask |= cls_pixels
        else:
            friction[cls_pixels] = value

    # Treat the canonical water class as water even if its lookup entry was
    # changed: keeps the barge / ice_road logic well-defined.
    water_mask |= lulc == LULC_WATER_CLASS
    friction[water_mask] = FRICTION_NODATA
    return friction, water_mask, profile


def build_static_base(
    slope_path: str | Path,
    lulc_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Year-round terrain base = max(slope_friction, lulc_friction).

    Water pixels are FRICTION_NODATA; mode-specific surfaces handle them.
    Permafrost is intentionally NOT included here.

    Returns (static_base, water_mask, profile).
    """
    slope_fric, profile = compute_slope_friction(slope_path)
    lulc_fric, water_mask, _ = compute_lulc_friction(lulc_path)

    if slope_fric.shape != lulc_fric.shape:
        raise ValueError(
            f"Slope shape {slope_fric.shape} != LULC shape {lulc_fric.shape}; "
            "inputs must be on a common grid."
        )

    base = np.maximum(slope_fric, lulc_fric).astype(np.float32)
    base[water_mask] = FRICTION_NODATA
    return base, water_mask, profile


def compute_permafrost_modifier(
    permafrost_path: str | Path,
    month: int,
) -> np.ndarray:
    """Per-pixel multiplicative modifier (>=1.0) based on permafrost extent.

    Winter (Nov-Apr): 1.0 everywhere (frozen ground is trafficable).
    Shoulder (May, Oct): 1.0 + p * (PERMAFROST_MAX_SHOULDER - 1.0).
    Summer (Jun-Sep): 1.0 + p * (PERMAFROST_MAX_SUMMER - 1.0).

    The raster may be encoded 0-1 or 0-100; this function normalizes by
    inspecting the input range.
    """
    if month < 1 or month > 12:
        raise ValueError(f"month must be 1..12, got {month}")

    with rasterio.open(permafrost_path) as src:
        permafrost = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        permafrost = np.where(permafrost == nodata, 0.0, permafrost)

    finite_max = float(np.nanmax(permafrost)) if permafrost.size else 0.0
    if finite_max > 1.0:
        permafrost = permafrost / 100.0
    permafrost = np.clip(permafrost, 0.0, 1.0)

    if month in PERMAFROST_WINTER_MONTHS:
        return np.ones(permafrost.shape, dtype=np.float32)
    if month in PERMAFROST_SHOULDER_MONTHS:
        peak = PERMAFROST_MAX_SHOULDER
    elif month in PERMAFROST_SUMMER_MONTHS:
        peak = PERMAFROST_MAX_SUMMER
    else:
        raise ValueError(f"month {month} fell through season classification")

    return (1.0 + permafrost * (peak - 1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Mode-specific friction
# ---------------------------------------------------------------------------

def _load_ice(path: str | Path) -> np.ndarray:
    """Load an ice-probability raster and normalize to 0-1."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, 0.0, arr)
    finite_max = float(np.nanmax(arr)) if arr.size else 0.0
    if finite_max > 1.0:
        arr = arr / 100.0
    return np.clip(arr, 0.0, 1.0)


def build_mode_friction(
    static_base: np.ndarray,
    water_mask: np.ndarray,
    permafrost_mod: np.ndarray,
    sea_ice_arr: np.ndarray,
    river_ice_arr: np.ndarray,
    mode: str,
    month: int,
) -> np.ndarray:
    """Produce a mode-specific monthly friction surface.

    Impassable pixels are FRICTION_NODATA. The signature is uniform across
    modes so the driver can loop cleanly.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if static_base.shape != water_mask.shape:
        raise ValueError("static_base and water_mask shape mismatch")

    combined_ice = np.maximum(sea_ice_arr, river_ice_arr)
    ice_present = combined_ice > ICE_PROB_THRESHOLD
    out = np.full(static_base.shape, FRICTION_NODATA, dtype=np.float32)
    land_mask = ~water_mask
    valid_base = land_mask & (static_base != FRICTION_NODATA)

    if mode == "overland":
        out[valid_base] = static_base[valid_base] * permafrost_mod[valid_base]
        return out

    if mode == "barge":
        navigable = water_mask & ~ice_present
        out[navigable] = WATER_FRICTION_BARGE
        return out

    if mode == "ice_road":
        ice_corridor = water_mask & ice_present
        out[ice_corridor] = WATER_FRICTION_ICEROAD
        # Ice-road season is winter -> permafrost modifier is ones, so the
        # static base is correct on land without further multiplication.
        out[valid_base] = static_base[valid_base]
        return out

    raise AssertionError(f"unhandled mode {mode!r}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _resolve_path(input_dir: Path, name: str) -> Path:
    p = input_dir / name
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p}")
    return p


def write_friction_stack(
    input_dir: str | Path,
    output_dir: str | Path,
    modes: Iterable[str] = MODES,
    months: Iterable[int] = range(1, 13),
) -> dict[tuple[str, int], Path]:
    """Build and write all mode-month friction surfaces.

    Inputs expected under input_dir:
      slope.tif, lulc.tif, permafrost.tif,
      sea_ice/sea_ice_{01..12}.tif,
      river_ice/river_ice_{01..12}.tif

    Outputs under output_dir:
      {mode}_{MM}.tif for each (mode, month).

    Returns a dict mapping (mode, month) -> output Path.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slope_path = _resolve_path(input_dir, "slope.tif")
    lulc_path = _resolve_path(input_dir, "lulc.tif")
    permafrost_path = _resolve_path(input_dir, "permafrost.tif")

    static_base, water_mask, profile = build_static_base(slope_path, lulc_path)

    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=FRICTION_NODATA,
        compress="lzw",
    )

    written: dict[tuple[str, int], Path] = {}
    modes = tuple(modes)
    months = tuple(months)

    permafrost_cache: dict[int, np.ndarray] = {}

    for month in months:
        sea_ice_path = _resolve_path(input_dir, f"sea_ice/sea_ice_{month:02d}.tif")
        river_ice_path = _resolve_path(input_dir, f"river_ice/river_ice_{month:02d}.tif")
        sea_ice = _load_ice(sea_ice_path)
        river_ice = _load_ice(river_ice_path)

        if month not in permafrost_cache:
            permafrost_cache[month] = compute_permafrost_modifier(permafrost_path, month)
        permafrost_mod = permafrost_cache[month]

        for mode in modes:
            arr = build_mode_friction(
                static_base=static_base,
                water_mask=water_mask,
                permafrost_mod=permafrost_mod,
                sea_ice_arr=sea_ice,
                river_ice_arr=river_ice,
                mode=mode,
                month=month,
            )
            out_path = output_dir / f"{mode}_{month:02d}.tif"
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(arr, 1)
            written[(mode, month)] = out_path
            logger.info("wrote %s", out_path)

    return written
