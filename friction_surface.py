# -*- coding: utf-8 -*-
"""friction_surface.py

Core friction surface computation module.

Builds composite friction rasters from GEE-exported layers and computes
least-cost paths between facility pairs using WhiteboxTools. Results are
written to the DuckDB graph database.

This module is pure computation -- no LLM or CrewAI agents.
"""

import os
import tempfile
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy

import friction_config as fc
import pipeline


# =========================================================================
# 1. Load rasters
# =========================================================================

def load_rasters(raster_dir=None):
    """Load all GEE rasters listed in friction_config.RASTER_FILES.

    Args:
        raster_dir: Optional override for the raster directory.  When
            provided the base directory in each RASTER_FILES path is
            replaced.

    Returns:
        dict  -- keys match RASTER_FILES keys, values are
                 (numpy_array, rasterio_profile) tuples.

    Raises:
        FileNotFoundError: if any expected raster is missing.
        ValueError: if rasters do not share the same CRS / transform / shape.
    """
    rasters = {}
    ref_crs = None
    ref_transform = None
    ref_shape = None

    for key, default_path in fc.RASTER_FILES.items():
        if raster_dir is not None:
            path = os.path.join(raster_dir, os.path.basename(default_path))
        else:
            path = default_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Raster not found: {path}")

        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            profile = dict(src.profile)

        # Validate consistency across all rasters
        if ref_crs is None:
            ref_crs = profile["crs"]
            ref_transform = profile["transform"]
            ref_shape = (profile["height"], profile["width"])
        else:
            if str(profile["crs"]) != str(ref_crs):
                raise ValueError(
                    f"CRS mismatch for '{key}': expected {ref_crs}, "
                    f"got {profile['crs']}"
                )
            if profile["transform"] != ref_transform:
                raise ValueError(
                    f"Transform mismatch for '{key}'"
                )
            cur_shape = (profile["height"], profile["width"])
            if cur_shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch for '{key}': expected {ref_shape}, "
                    f"got {cur_shape}"
                )

        rasters[key] = (arr, profile)

    print(f"Loaded {len(rasters)} rasters  CRS={ref_crs}  "
          f"shape={ref_shape}  pixel_size={ref_transform.a:.1f}m")
    for k in rasters:
        print(f"  - {k}")

    return rasters


# =========================================================================
# 2. Classify slope
# =========================================================================

def classify_slope(slope_array, nodata=-9999):
    """Reclassify slope degrees into friction multipliers.

    Uses friction_config.SLOPE_THRESHOLDS and SLOPE_FRICTION.

    Args:
        slope_array: 2-D numpy array of slope in degrees.
        nodata: Value that marks missing data in the array.

    Returns:
        np.ndarray (float32) -- friction values.
    """
    lo, hi = fc.SLOPE_THRESHOLDS
    out = np.full_like(slope_array, nodata, dtype=np.float32)

    valid = slope_array != nodata
    flat = valid & (slope_array < lo)
    rolling = valid & (slope_array >= lo) & (slope_array <= hi)
    mountain = valid & (slope_array > hi)

    out[flat] = fc.SLOPE_FRICTION["flat"]
    out[rolling] = fc.SLOPE_FRICTION["rolling"]
    out[mountain] = fc.SLOPE_FRICTION["mountain"]

    return out


# =========================================================================
# 3. Build combined LULC + permafrost friction
# =========================================================================

def build_lulc_permafrost_friction(lulc, permafrost):
    """Combine LULC and permafrost layers into a single friction array.

    For LULC classes that appear in PERMAFROST_LULC_MATRIX the combined
    value is taken from the matrix.  For all other classes the base
    LULC_FRICTION_ROAD value is used.

    Args:
        lulc: 2-D int/float array of LULC class codes.
        permafrost: 2-D int/float array of permafrost zone codes.

    Returns:
        np.ndarray (float32) -- combined friction.
    """
    out = np.full(lulc.shape, fc.IMPASSABLE, dtype=np.float32)

    # Vectorised: iterate unique LULC codes (small set 0-8)
    for lulc_code, base_friction in fc.LULC_FRICTION_ROAD.items():
        mask = (lulc.astype(int) == lulc_code)
        if not np.any(mask):
            continue

        if lulc_code in fc.PERMAFROST_LULC_MATRIX:
            # Apply permafrost-specific friction for each zone
            pf_map = fc.PERMAFROST_LULC_MATRIX[lulc_code]
            for pf_zone, pf_friction in pf_map.items():
                zone_mask = mask & (permafrost.astype(int) == pf_zone)
                out[zone_mask] = pf_friction
        else:
            out[mask] = base_friction

    return out


# =========================================================================
# 3b. Classify water type per pixel
# =========================================================================

def classify_water_type(rasters):
    """Classify each pixel into a water type code for seasonal multiplier lookup.

    Returns:
        np.ndarray (int8) -- water type codes from friction_config.WATER_TYPE_*.
    """
    lulc = rasters["lulc"][0]
    rivers = rasters["rivers"][0]
    sea_ice = rasters["sea_ice"][0]
    sea_ice_profile = rasters["sea_ice"][1]
    sea_ice_nodata = sea_ice_profile.get("nodata", -9999)

    out = np.zeros(lulc.shape, dtype=np.int8)

    water = (lulc.astype(int) == 0)
    out[water] = fc.WATER_TYPE_OPEN

    out[rivers.astype(int) == 1] = fc.WATER_TYPE_RIVER

    # Pixels with <20% ice stay WATER_TYPE_OPEN (navigable year-round by default)
    non_river_water = water & (out != fc.WATER_TYPE_RIVER)
    ice_valid = non_river_water & (sea_ice >= 0)
    if sea_ice_nodata is not None:
        ice_valid = ice_valid & (sea_ice != sea_ice_nodata)

    frozen = ice_valid & (sea_ice > fc.SEA_ICE_THRESHOLDS["sea_ice"])
    marginal = (ice_valid
                & (sea_ice > fc.SEA_ICE_THRESHOLDS["marginal_sea_ice"])
                & (sea_ice <= fc.SEA_ICE_THRESHOLDS["sea_ice"]))

    out[frozen] = fc.WATER_TYPE_SEASONALLY_FROZEN
    out[marginal] = fc.WATER_TYPE_SEASONALLY_MARGINAL

    return out


# =========================================================================
# 3c. Build season-specific barge friction surface
# =========================================================================

def build_friction_barge_seasonal(rasters, season, water_type=None):
    """Build a season-specific barge friction surface.

    Each water pixel gets: base_friction * seasonal_mult(water_type, region, season).
    Pixels where the multiplier is IMPASSABLE become np.nan (true barrier).

    Args:
        rasters: dict from load_rasters().
        season: one of "summer", "shoulder", "winter".
        water_type: optional precomputed water type array (avoids recomputation).

    Returns:
        np.ndarray (float32) -- seasonal barge friction, np.nan = barrier.
    """
    base = build_friction_barge(rasters)

    if water_type is None:
        water_type = classify_water_type(rasters)

    region_arr = rasters["regions"][0].astype(int)
    multiplier = np.ones_like(base, dtype=np.float32)

    for wt_code, feature_name in fc.WATER_TYPE_FEATURES.items():
        wt_mask = (water_type == wt_code)

        global_mult = fc.SEASONAL_MULTIPLIERS.get((feature_name, season), 1.0)
        multiplier[wt_mask] = global_mult

        for region_id, region_name in fc.REGION_IDS.items():
            override = fc.REGIONAL_SEASONAL_OVERRIDES.get(
                (region_name, feature_name, season))
            if override is not None:
                multiplier[wt_mask & (region_arr == region_id)] = override

    out = base * multiplier

    impassable = (multiplier >= fc.IMPASSABLE) | (base >= fc.IMPASSABLE)
    out[impassable] = np.nan

    return out


# =========================================================================
# 4. Build road friction surface
# =========================================================================

def _dilate_mask(mask, pixels):
    """Binary-dilate a boolean mask by *pixels* rounds (3x3 structuring element).

    Uses pure numpy (no scipy dependency).  Edge pixels are padded with
    False so dilation cannot wrap around.
    """
    out = mask
    for _ in range(pixels):
        p = np.pad(out, 1, constant_values=False)
        out = (
            p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:]
            | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
            | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:]
        )
    return out


def build_friction_road(rasters):
    """Build composite road-delivery friction surface.

    Logic:
        a. Start with max(slope_friction, lulc_permafrost_friction) per pixel.
        b. Where roads exist (GRIP4 binary presence, optionally dilated by
           ROAD_BUFFER_PIXELS), override with ROAD_PRESENT_FRICTION.
        c. Where rivers exist, set to IMPASSABLE.
        d. Where LULC is water (class 0), set to IMPASSABLE.

    Args:
        rasters: dict from load_rasters().

    Returns:
        np.ndarray (float32) -- road friction surface.
    """
    slope_arr = rasters["slope"][0]
    lulc_arr = rasters["lulc"][0]
    pf_arr = rasters["permafrost"][0]
    roads_pres = rasters["roads_presence"][0]
    rivers = rasters["rivers"][0]

    slope_friction = classify_slope(slope_arr)
    lulc_pf_friction = build_lulc_permafrost_friction(lulc_arr, pf_arr)

    # (a) baseline: max of slope and lulc/permafrost friction
    # Treat nodata slope pixels as 0 friction contribution
    slope_clean = np.where(slope_friction == -9999, 0.0, slope_friction)
    base = np.maximum(slope_clean, lulc_pf_friction)

    # (b) override where roads are present (uniform friction value)
    road_mask = (roads_pres.astype(int) == fc.GRIP4_ROAD_PRESENT)
    if fc.ROAD_BUFFER_PIXELS > 0:
        road_mask = _dilate_mask(road_mask, fc.ROAD_BUFFER_PIXELS)
    base[road_mask] = fc.ROAD_PRESENT_FRICTION

    # (c) rivers -> impassable for road
    for rclass in fc.RIVER_FRICTION_ROAD:
        river_mask = (rivers.astype(int) == rclass)
        base[river_mask] = fc.IMPASSABLE

    # (d) water LULC -> impassable
    water_mask = (lulc_arr.astype(int) == 0)
    base[water_mask] = fc.IMPASSABLE

    return base.astype(np.float32)


# =========================================================================
# 5. Build barge friction surface
# =========================================================================

def build_friction_barge(rasters):
    """Build barge-delivery friction surface.

    Logic:
        a. Land cells = IMPASSABLE.
        b. Water LULC (class 0) = 1.0 (navigable).
        c. River cells: major = 1.0, minor = 1.5.
        d. Port locations from ports_alaska.geojson get PORT_FRICTION.

    Args:
        rasters: dict from load_rasters().

    Returns:
        np.ndarray (float32) -- barge friction surface.
    """
    lulc_arr = rasters["lulc"][0]
    rivers = rasters["rivers"][0]
    profile = rasters["lulc"][1]

    shape = lulc_arr.shape
    out = np.full(shape, fc.IMPASSABLE, dtype=np.float32)

    # (b) Water is navigable
    water_mask = (lulc_arr.astype(int) == 0)
    out[water_mask] = 1.0

    # (c) Rivers
    for rclass, rfric in fc.RIVER_FRICTION_BARGE.items():
        rmask = (rivers.astype(int) == rclass)
        out[rmask] = rfric

    # (d) Port locations
    ports_path = fc.VECTOR_FILES["ports"]
    if os.path.exists(ports_path):
        import geopandas as gpd
        from pyproj import Transformer

        ports = gpd.read_file(ports_path)
        transform = profile["transform"]
        target_crs = str(profile["crs"])

        # Reproject port coordinates to raster CRS if needed
        if ports.crs is not None and str(ports.crs) != target_crs:
            ports = ports.to_crs(target_crs)

        for _, port in ports.iterrows():
            geom = port.geometry
            px, py = geom.x, geom.y
            try:
                r, c = rowcol(transform, px, py)
            except Exception:
                continue

            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                port_class = port.get("port_class", "port")
                base_fric = fc.PORT_FRICTION.get(port_class,
                                                  fc.PORT_FRICTION["port"])
                buf = fc.PORT_BUFFER.get(port_class, 1)
                decay = fc.PORT_DECAY_PER_PIXEL.get(port_class, 0.0)

                for dr in range(-buf, buf + 1):
                    for dc in range(-buf, buf + 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                            d = max(abs(dr), abs(dc))
                            fric = base_fric + d * decay
                            if fric < out[rr, cc]:
                                out[rr, cc] = fric

    return out


# =========================================================================
# 6. Save friction rasters
# =========================================================================
#
# Note: Plane delivery uses direct Haversine distance (airport-to-airport)
# rather than a cell-by-cell friction surface.  Plane edges are populated
# with avg_friction=1.0 and path_length_miles=distance_miles in main().

def save_friction_rasters(road, barge_surfaces, profile, output_dir=None):
    """Write friction surfaces to GeoTIFF files.

    Args:
        road: 2-D array -- road friction surface.
        barge_surfaces: dict {season: ndarray} for summer/shoulder/winter,
            OR a single 2-D array (legacy single-surface mode).
        profile: rasterio profile dict (CRS, transform, etc.).
        output_dir: Directory for output files.  Defaults to raster_dir.
    """
    if output_dir is None:
        output_dir = pipeline.get_raster_dir()
    os.makedirs(output_dir, exist_ok=True)

    write_profile = profile.copy()
    write_profile.update(dtype="float32", count=1, compress="lzw",
                         nodata=fc.FRICTION_NODATA)

    names = {"friction_road.tif": road}
    if isinstance(barge_surfaces, dict):
        for season, arr in barge_surfaces.items():
            names[f"friction_barge_{season}.tif"] = arr
    else:
        names["friction_barge.tif"] = barge_surfaces

    for fname, arr in names.items():
        path = os.path.join(output_dir, fname)
        arr_out = np.where(np.isnan(arr), fc.FRICTION_NODATA,
                           arr).astype(np.float32)
        with rasterio.open(path, "w", **write_profile) as dst:
            dst.write(arr_out, 1)
        print(f"Saved {path}  ({arr.shape})")


# =========================================================================
# 8. Compute least-cost paths for a delivery method
# =========================================================================

def _make_source_raster(profile, src_x, src_y, tmp_dir):
    """Create a single-pixel source raster for WhiteboxTools cost_distance.

    The output raster has 0 at the source cell and nodata (-9999) elsewhere.

    Args:
        profile: rasterio profile dict.
        src_x: projected X coordinate of the source facility.
        src_y: projected Y coordinate of the source facility.
        tmp_dir: directory for temporary files.

    Returns:
        str -- path to the source raster TIF.
    """
    height = profile["height"]
    width = profile["width"]
    transform = profile["transform"]

    arr = np.full((height, width), -9999, dtype=np.float32)
    r, c = rowcol(transform, src_x, src_y)
    if 0 <= r < height and 0 <= c < width:
        arr[r, c] = 0.0
    else:
        raise ValueError(
            f"Source ({src_x}, {src_y}) falls outside raster extent"
        )

    src_profile = profile.copy()
    src_profile.update(dtype="float32", count=1, nodata=-9999)

    path = os.path.join(tmp_dir, f"source_{r}_{c}.tif")
    with rasterio.open(path, "w", **src_profile) as dst:
        dst.write(arr, 1)

    return path


def _extract_cost_at(cost_surface_path, x, y):
    """Read the accumulated cost value at a projected coordinate.

    Returns:
        float or None if the coordinate falls outside or on nodata.
    """
    with rasterio.open(cost_surface_path) as src:
        transform = src.transform
        nodata = src.nodata
        r, c = rowcol(transform, x, y)
        if 0 <= r < src.height and 0 <= c < src.width:
            val = src.read(1)[r, c]
            if nodata is not None and val == nodata:
                return None
            return float(val)
    return None


def _sample_path_friction(backlink_path, friction_tif_path, dst_x, dst_y,
                          water_type_arr=None):
    """Trace the cost pathway from destination back to source and sample
    friction values along it.

    Uses WhiteboxTools cost_pathway to produce a binary path raster, then
    samples the friction surface along that path.

    Args:
        water_type_arr: optional water type array for diagnostic fractions.

    Returns:
        (waf, max_friction, path_length_miles, diagnostics) where diagnostics
        is a dict of water type fractions or None.
    """
    wbt = pipeline.get_whitebox_wbt()
    work_dir = wbt.get_working_dir() or tempfile.mkdtemp()

    # Create destination point raster
    with rasterio.open(backlink_path) as src:
        profile = dict(src.profile)
        transform = src.transform
        height, width = src.height, src.width

    dst_arr = np.full((height, width), -9999, dtype=np.float32)
    r, c = rowcol(transform, dst_x, dst_y)
    if not (0 <= r < height and 0 <= c < width):
        return None, None, None, None
    dst_arr[r, c] = 0.0

    dst_profile = profile.copy()
    dst_profile.update(dtype="float32", count=1, nodata=-9999)
    dst_raster = os.path.join(work_dir, f"dst_{r}_{c}.tif")
    with rasterio.open(dst_raster, "w", **dst_profile) as dst:
        dst.write(dst_arr, 1)

    pathway_out = os.path.join(work_dir, f"pathway_{r}_{c}.tif")

    try:
        wbt.cost_pathway(
            destination=dst_raster,
            backlink=backlink_path,
            output=pathway_out,
        )
    except Exception as e:
        print(f"  cost_pathway failed: {e}")
        return None, None, None, None

    if not os.path.exists(pathway_out):
        return None, None, None, None

    # Read path cells and sample friction
    with rasterio.open(pathway_out) as src:
        path_arr = src.read(1)
        pw_nodata = src.nodata
        pw_transform = src.transform
        pixel_size = abs(pw_transform.a)  # metres

    with rasterio.open(friction_tif_path) as src:
        friction_arr = src.read(1)

    # Path cells are non-zero / non-nodata
    if pw_nodata is not None:
        path_mask = (path_arr != pw_nodata) & (path_arr != 0)
    else:
        path_mask = path_arr != 0

    n_cells = int(np.sum(path_mask))
    if n_cells == 0:
        return None, None, None, None

    sampled = friction_arr[path_mask]
    waf = float(np.mean(sampled))
    max_friction = float(np.max(sampled))
    path_length_m = n_cells * pixel_size
    path_length_miles = path_length_m / 1609.344

    diagnostics = None
    if water_type_arr is not None:
        wt_along_path = water_type_arr[path_mask]
        n = len(wt_along_path)
        if n > 0:
            diagnostics = {
                "river_frac": float(
                    np.sum(wt_along_path == fc.WATER_TYPE_RIVER) / n),
                "sea_ice_frac": float(
                    np.sum(wt_along_path == fc.WATER_TYPE_SEASONALLY_FROZEN) / n),
                "marginal_frac": float(
                    np.sum(wt_along_path == fc.WATER_TYPE_SEASONALLY_MARGINAL) / n),
            }

    # Cleanup temp files
    for p in [dst_raster, pathway_out]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    return waf, max_friction, path_length_miles, diagnostics


def compute_paths_for_method(con, friction_tif_path, method_name,
                             season=None, water_type_arr=None):
    """Compute least-cost paths for all edges of a given delivery method.

    For each unique source facility:
        1. Create a single-pixel source raster.
        2. Run WhiteboxTools cost_distance -> cost surface + backlink.
        3. For each destination, extract accumulated cost and trace path.

    Args:
        con: DuckDB connection.
        friction_tif_path: Path to the friction raster for this method.
        method_name: Delivery method name (e.g. "Road", "Barge", "Plane").
        season: Optional season name (e.g. "summer"). Included in results.
        water_type_arr: Optional water type array for diagnostic fractions.

    Returns:
        list of dicts: [{src, dst, avg_friction, max_friction,
                         path_length_miles, season, diagnostics}, ...]
    """
    # Fetch edges for this method via uses_method join
    edges = con.execute("""
        SELECT ct.src, ct.dst
        FROM connects_to ct
        JOIN uses_method um ON ct.src = um.facility_id
        WHERE um.method_name = ?
    """, [method_name]).fetchall()

    if not edges:
        print(f"No edges for method '{method_name}'")
        return []

    # Group destinations by source
    src_to_dsts = {}
    for src, dst in edges:
        src_to_dsts.setdefault(src, []).append(dst)

    # Load friction raster profile
    with rasterio.open(friction_tif_path) as src_ds:
        profile = dict(src_ds.profile)

    # Facility coordinates (projected)
    fac_rows = con.execute(
        "SELECT facility_id, x_3413, y_3413 FROM facilities"
    ).fetchall()
    fac_coords = {fid: (x, y) for fid, x, y in fac_rows}

    wbt = pipeline.get_whitebox_wbt()
    work_dir = wbt.get_working_dir() or tempfile.mkdtemp()

    results = []
    total_sources = len(src_to_dsts)

    for i, (src_id, dst_ids) in enumerate(src_to_dsts.items(), 1):
        if src_id not in fac_coords:
            print(f"  Skipping source {src_id}: no coordinates")
            continue

        sx, sy = fac_coords[src_id]
        print(f"  [{i}/{total_sources}] Source {src_id}  "
              f"({len(dst_ids)} destinations)")

        # Create source raster
        try:
            src_raster = _make_source_raster(profile, sx, sy, work_dir)
        except ValueError as e:
            print(f"    {e}")
            continue

        # Run cost_distance once per source
        cost_out = os.path.join(work_dir, f"cost_{src_id}.tif")
        backlink_out = os.path.join(work_dir, f"backlink_{src_id}.tif")

        try:
            wbt.cost_distance(
                source=src_raster,
                cost=friction_tif_path,
                out_accum=cost_out,
                out_backlink=backlink_out,
            )
        except Exception as e:
            print(f"    cost_distance failed: {e}")
            continue

        if not os.path.exists(cost_out):
            print(f"    cost_distance produced no output for source {src_id}")
            continue

        # Read off accumulated costs for each destination
        for dst_id in dst_ids:
            if dst_id not in fac_coords:
                continue
            dx, dy = fac_coords[dst_id]

            acc_cost = _extract_cost_at(cost_out, dx, dy)
            if acc_cost is None:
                print(f"    {src_id} -> {dst_id}: unreachable")
                results.append({
                    "src": src_id,
                    "dst": dst_id,
                    "avg_friction": None,
                    "max_friction": None,
                    "path_length_miles": None,
                    "season": season,
                    "diagnostics": None,
                })
                continue

            waf, max_f, length_mi, diag = _sample_path_friction(
                backlink_out, friction_tif_path, dx, dy,
                water_type_arr=water_type_arr,
            )

            results.append({
                "src": src_id,
                "dst": dst_id,
                "avg_friction": waf,
                "max_friction": max_f,
                "path_length_miles": length_mi,
                "season": season,
                "diagnostics": diag,
            })

        # Cleanup source-specific temp files
        for p in [src_raster, cost_out, backlink_out]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    print(f"  {method_name}: computed {len(results)} paths")
    return results


# =========================================================================
# 9. Update graph with friction values
# =========================================================================

def update_graph_friction(con, results, season=None):
    """Write friction values back to the connects_to edge table in DuckDB.

    When season is provided, writes to friction_{season} column.
    Also writes diagnostic fractions (river_frac, sea_ice_frac, marginal_frac)
    suffixed by season when present.

    Args:
        con: DuckDB connection (read-write).
        results: list of dicts from compute_paths_for_method().
        season: optional season name for seasonal column writes.
    """
    if not results:
        return

    # Ensure columns exist
    existing_cols = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'connects_to'"
        ).fetchall()
    }

    base_cols = ["avg_friction", "max_friction", "path_length_miles"]
    if season:
        base_cols.append(f"friction_{season}")
    diag_cols = []
    if season in ("summer", "winter"):
        diag_cols = [
            f"river_frac_{season}",
            f"sea_ice_frac_{season}",
            f"marginal_frac_{season}",
        ]

    for col in base_cols + diag_cols:
        if col not in existing_cols:
            con.execute(
                f"ALTER TABLE connects_to ADD COLUMN {col} DOUBLE"
            )
            existing_cols.add(col)

    # Batch update
    for r in results:
        if r["avg_friction"] is None:
            if season:
                con.execute(
                    f"UPDATE connects_to "
                    f"SET friction_{season} = NULL "
                    f"WHERE src = ? AND dst = ?",
                    [r["src"], r["dst"]],
                )
            continue

        if season:
            con.execute(
                f"UPDATE connects_to "
                f"SET avg_friction = COALESCE(avg_friction, ?), "
                f"    max_friction = COALESCE(max_friction, ?), "
                f"    path_length_miles = COALESCE(path_length_miles, ?), "
                f"    friction_{season} = ? "
                f"WHERE src = ? AND dst = ?",
                [r["avg_friction"], r["max_friction"], r["path_length_miles"],
                 r["avg_friction"], r["src"], r["dst"]],
            )
        else:
            con.execute(
                "UPDATE connects_to "
                "SET avg_friction = ?, max_friction = ?, path_length_miles = ? "
                "WHERE src = ? AND dst = ?",
                [r["avg_friction"], r["max_friction"], r["path_length_miles"],
                 r["src"], r["dst"]],
            )

        diag = r.get("diagnostics")
        if diag and season in ("summer", "winter"):
            con.execute(
                f"UPDATE connects_to "
                f"SET river_frac_{season} = ?, "
                f"    sea_ice_frac_{season} = ?, "
                f"    marginal_frac_{season} = ? "
                f"WHERE src = ? AND dst = ?",
                [diag["river_frac"], diag["sea_ice_frac"],
                 diag["marginal_frac"], r["src"], r["dst"]],
            )

    n_updated = sum(1 for r in results if r["avg_friction"] is not None)
    label = f" ({season})" if season else ""
    print(f"Updated {n_updated}/{len(results)} edges in connects_to{label}")


# =========================================================================
# 10. Main pipeline entry point
# =========================================================================

def main(con=None):
    """Run the full friction surface pipeline.

    Steps:
        a. Get or create DuckDB connection.
        b. Reproject facilities to EPSG:3413.
        c. Load all rasters.
        d. Build friction_road, friction_barge (no plane friction — see below).
        e. Save friction rasters.
        f. For Road and Barge, compute least-cost paths via WhiteboxTools.
        g. For Plane, set avg_friction=1.0 and path_length_miles=distance_miles
           (direct Haversine — planes fly airport-to-airport, not cell-by-cell).
        h. Update graph with friction values.
        i. Print summary statistics.

    Compound delivery methods ("Plane or Barge", "Plane or Road") are resolved
    to a single method during regionalization. By the time this module runs,
    every facility has exactly one method in uses_method.

    Args:
        con: Optional existing DuckDB connection.  If None, one is created.
    """
    print("=" * 60)
    print("FRICTION SURFACE PIPELINE")
    print("=" * 60)

    # (a) DuckDB connection
    own_con = False
    if con is None:
        con = pipeline.get_duckdb_connection()
        own_con = True

    try:
        # (b) Reproject facilities
        print("\n--- Reprojecting facilities to EPSG:3413 ---")
        pipeline.reproject_facilities(con)

        # (c) Load rasters
        print("\n--- Loading rasters ---")
        rasters = load_rasters()

        # Grab a reference profile for output
        ref_profile = rasters["lulc"][1]

        # (d) Build friction surfaces
        print("\n--- Building road friction surface ---")
        friction_road = build_friction_road(rasters)
        print(f"  Range: [{friction_road.min():.2f}, {friction_road.max():.2f}]")

        print("\n--- Building seasonal barge friction surfaces ---")
        water_type = classify_water_type(rasters)
        barge_surfaces = {}
        for season in fc.SEASONS:
            barge_surfaces[season] = build_friction_barge_seasonal(
                rasters, season, water_type=water_type)
            arr = barge_surfaces[season]
            valid = arr[~np.isnan(arr)]
            n_barrier = int(np.sum(np.isnan(arr)))
            print(f"  {season}: valid=[{valid.min():.2f}, {valid.max():.2f}]  "
                  f"barriers={n_barrier:,}")

        # (e) Save friction rasters
        print("\n--- Saving friction rasters ---")
        save_friction_rasters(friction_road, barge_surfaces, ref_profile)

        # (f) Compute least-cost paths
        raster_dir = pipeline.get_raster_dir()

        # Road paths (unchanged)
        all_results = []
        road_tif = os.path.join(raster_dir, "friction_road.tif")
        print("\n--- Computing Road paths ---")
        road_results = compute_paths_for_method(con, road_tif, "Road")
        all_results.extend(road_results)
        update_graph_friction(con, road_results)

        # Barge paths per season
        for season in fc.SEASONS:
            barge_tif = os.path.join(raster_dir,
                                     f"friction_barge_{season}.tif")
            sample_diag = season in ("summer", "winter")
            print(f"\n--- Computing Barge paths ({season}) ---")
            results = compute_paths_for_method(
                con, barge_tif, "Barge",
                season=season,
                water_type_arr=water_type if sample_diag else None,
            )
            update_graph_friction(con, results, season=season)
            all_results.extend(results)

        # (g) Plane: direct Haversine distance (no friction surface)
        print("\n--- Setting Plane edges to direct Haversine distance ---")
        con.execute("""
            UPDATE connects_to
            SET avg_friction = 1.0,
                max_friction = 1.0,
                path_length_miles = distance_miles
            WHERE src IN (SELECT facility_id FROM uses_method WHERE method_name = 'Plane')
              AND distance_miles IS NOT NULL
        """)
        plane_count = con.execute("""
            SELECT COUNT(*) FROM connects_to ct
            JOIN uses_method um ON ct.src = um.facility_id
            WHERE um.method_name = 'Plane' AND ct.avg_friction IS NOT NULL
        """).fetchone()[0]
        print(f"  {plane_count} Plane edges set to Haversine distance")

        # (h) Graph already updated inline above

        # (i) Summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        valid = [r for r in all_results if r["avg_friction"] is not None]
        unreachable = [r for r in all_results if r["avg_friction"] is None]
        print(f"Total edges processed: {len(all_results)}")
        print(f"  Reachable:   {len(valid)}")
        print(f"  Unreachable: {len(unreachable)}")

        if valid:
            frictions = [r["avg_friction"] for r in valid]
            lengths = [r["path_length_miles"] for r in valid
                       if r["path_length_miles"] is not None]
            print(f"  Avg friction: {np.mean(frictions):.3f} "
                  f"(min={np.min(frictions):.3f}, max={np.max(frictions):.3f})")
            if lengths:
                print(f"  Avg path length: {np.mean(lengths):.1f} mi "
                      f"(min={np.min(lengths):.1f}, "
                      f"max={np.max(lengths):.1f})")

    finally:
        if own_con:
            con.close()

    print("\nFriction surface pipeline complete.")


if __name__ == "__main__":
    main()
