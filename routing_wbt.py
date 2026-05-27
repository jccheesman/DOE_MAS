"""routing_wbt.py

WhiteboxTools cost-distance wrapper and edge computation.

Integration with the DuckDB graph:
  - Writes the full 12-month x 3-mode detail to a new table
    `mode_specific_edges (src, dst, mode, month, path_length_m,
                          weighted_avg_friction, total_cost, unreachable)`.
  - Also backfills `connects_to.avg_friction / path_length_miles /
    delivery_cost` with the representative month per mode
    (overland=Jun, barge=Jul, ice_road=Feb). This keeps
    tsp_model_graph.py working unchanged: it reads only those existing
    columns from `connects_to`.

No CrewAI, no LLM calls.
"""

from __future__ import annotations

import logging
import math
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point

from friction_config import BASELINE_RATES
from pipeline import get_whitebox_wbt

logger = logging.getLogger(__name__)


METERS_PER_MILE = 1609.344
DEFAULT_REPR_MONTHS = {"overland": 6, "barge": 7, "ice_road": 2}
METHOD_TO_MODE = {"Road": "overland", "Barge": "barge", "Plane": "plane"}
RATE_KEY_BY_MODE = {"overland": "Road", "barge": "Barge", "ice_road": "Road", "plane": "Plane"}


# ---------------------------------------------------------------------------
# WhiteboxTools setup
# ---------------------------------------------------------------------------

_wbt = get_whitebox_wbt()
_wbt.verbose = False
try:
    _wbt.set_compress_rasters(True)
except Exception:  # older whitebox builds may lack the setter; not fatal
    logger.debug("set_compress_rasters not available on this WBT build")


def _wbt_work_dir() -> Path:
    """Return WBT's working directory as an absolute Path.

    WBT joins relative output paths against its own working directory, so
    feeding it relative paths produces duplicated path components. We
    resolve once and reuse.
    """
    return Path(_wbt.get_working_dir()).resolve()


# Cache once so every wrapper sees the same absolute path.
_WBT_WORK = _wbt_work_dir()


# ---------------------------------------------------------------------------
# WBT wrappers
# ---------------------------------------------------------------------------

def compute_cost_distance(
    friction_path: str | Path,
    source_points_path: str | Path,
    wbt=_wbt,
) -> tuple[Path, Path]:
    """Run WBT CostDistance. Returns (cost_accum_path, backlink_path).

    WBT routes around NoData pixels in the friction raster automatically
    when the raster's nodata value is set in its profile.
    """
    work = _WBT_WORK
    tag = uuid.uuid4().hex[:8]
    accum_path = work / f"cost_accum_{tag}.tif"
    backlink_path = work / f"backlink_{tag}.tif"
    wbt.cost_distance(
        source=str(source_points_path),
        cost=str(friction_path),
        out_accum=str(accum_path),
        out_backlink=str(backlink_path),
    )
    return accum_path, backlink_path


def compute_least_cost_path(
    cost_accum_path: str | Path,
    backlink_path: str | Path,
    dest_points_path: str | Path,
    wbt=_wbt,
) -> Path:
    """Run WBT CostPathway. Returns path raster path."""
    work = _WBT_WORK
    tag = uuid.uuid4().hex[:8]
    out_path = work / f"path_{tag}.tif"
    wbt.cost_pathway(
        destination=str(dest_points_path),
        backlink=str(backlink_path),
        output=str(out_path),
    )
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rasterize_point(
    lon: float,
    lat: float,
    profile: dict,
    out_path: Path,
    src_crs: str = "EPSG:4326",
) -> Path:
    """Rasterize a single (lon, lat) point onto the friction grid.

    The output raster has value 1 at the burned pixel and NoData elsewhere,
    matching WBT's expectation for source/destination rasters.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, profile["crs"], always_xy=True)
    x, y = transformer.transform(lon, lat)
    geom = Point(x, y)
    shapes = [(geom, 1)]
    arr = rasterize(
        shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    point_profile = profile.copy()
    point_profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    with rasterio.open(out_path, "w", **point_profile) as dst:
        dst.write(arr, 1)
    return out_path


def _sample_path(
    path_raster: Path,
    friction_raster: Path,
) -> tuple[float | None, float | None]:
    """Sample the traced least-cost path.

    Walks the in-path neighbor graph: cardinal neighbors contribute
    pixel_size, diagonal neighbors contribute pixel_size*sqrt(2). Each
    undirected segment is counted once (via the (nr,nc) > (r,c) ordering)
    and the friction along it is the mean of the two endpoint frictions.

    Returns (weighted_avg_friction, total_length_m), or (None, None) if
    the path raster contains no traced pixels.
    """
    with rasterio.open(path_raster) as p_src:
        path = p_src.read(1)
        path_nodata = p_src.nodata
        pixel_size = abs(p_src.transform.a)

    mask = path > 0
    if path_nodata is not None:
        mask &= path != path_nodata
    if not mask.any():
        return None, None

    with rasterio.open(friction_raster) as f_src:
        friction = f_src.read(1)
        f_nodata = f_src.nodata

    rows, cols = np.where(mask)
    H, W = mask.shape
    sqrt2 = math.sqrt(2.0)

    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    total_len_m = 0.0
    weighted = 0.0

    def _eff_friction(r: int, c: int, fallback: float) -> float:
        v = friction[r, c]
        if f_nodata is not None and v == f_nodata:
            return fallback
        return float(v)

    for r, c in zip(rows.tolist(), cols.tolist()):
        f_p = _eff_friction(r, c, fallback=1.0)
        for dr, dc in cardinal:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and (nr, nc) > (r, c):
                seg = pixel_size
                f_n = _eff_friction(nr, nc, fallback=f_p)
                total_len_m += seg
                weighted += seg * (f_p + f_n) / 2.0
        for dr, dc in diagonal:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and (nr, nc) > (r, c):
                seg = pixel_size * sqrt2
                f_n = _eff_friction(nr, nc, fallback=f_p)
                total_len_m += seg
                weighted += seg * (f_p + f_n) / 2.0

    if total_len_m <= 0:
        return None, None
    return weighted / total_len_m, total_len_m


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in meters."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Edge computation
# ---------------------------------------------------------------------------

def _load_facility_pairs(graph_con) -> pd.DataFrame:
    """Pull facility pairs from connects_to with method codes for both ends."""
    return graph_con.execute(
        """
        SELECT c.src, c.dst,
               fs.longitude AS src_lon, fs.latitude AS src_lat,
               fd.longitude AS dst_lon, fd.latitude AS dst_lat,
               COALESCE(ums.method_name, fs.delivery_method_1) AS src_method,
               COALESCE(umd.method_name, fd.delivery_method_1) AS dst_method
        FROM connects_to c
        JOIN facilities fs ON c.src = fs.facility_id
        JOIN facilities fd ON c.dst = fd.facility_id
        LEFT JOIN uses_method ums ON c.src = ums.facility_id
        LEFT JOIN uses_method umd ON c.dst = umd.facility_id
        WHERE c.src < c.dst
        """
    ).fetchdf()


def _pair_mode(src_method: str | None, dst_method: str | None) -> str | None:
    """Resolve the common method for a pair. Returns None if mismatch / unknown."""
    if src_method is None or dst_method is None:
        return None
    if src_method != dst_method:
        return None
    return METHOD_TO_MODE.get(src_method)


def _compute_one_edge(
    src_id: int,
    dst_id: int,
    src_lon: float,
    src_lat: float,
    dst_lon: float,
    dst_lat: float,
    friction_path: Path,
    profile: dict,
) -> tuple[float | None, float | None, bool]:
    """Run WBT cost-distance for a single pair. Returns (length_m, friction, unreachable)."""
    work = _WBT_WORK
    tag = uuid.uuid4().hex[:8]
    src_raster = work / f"src_{src_id}_{tag}.tif"
    dst_raster = work / f"dst_{dst_id}_{tag}.tif"
    _rasterize_point(src_lon, src_lat, profile, src_raster)
    _rasterize_point(dst_lon, dst_lat, profile, dst_raster)

    try:
        accum, backlink = compute_cost_distance(friction_path, src_raster)
        path_raster = compute_least_cost_path(accum, backlink, dst_raster)
        avg_friction, length_m = _sample_path(path_raster, friction_path)
    except Exception as e:
        logger.warning("WBT failed for pair (%s,%s) on %s: %s",
                       src_id, dst_id, friction_path.name, e)
        return None, None, True
    finally:
        for f in (src_raster, dst_raster):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    if avg_friction is None or length_m is None or length_m <= 0:
        return None, None, True
    return length_m, avg_friction, False


def compute_edge_costs(
    graph_con,
    friction_stack: dict[tuple[str, int], Path],
    output_dir: str | Path,
    repr_months: dict[str, int] = DEFAULT_REPR_MONTHS,
    months: Iterable[int] = range(1, 13),
) -> pd.DataFrame:
    """Compute cost-distance edges for all facility pairs across modes/months.

    For each ordered pair (src < dst) from connects_to:
      - Road -> compute overland and ice_road across all months.
      - Barge -> compute barge across all months.
      - Plane -> Haversine length, friction=1.0, mode='plane', month=0.

    Returns DataFrame with columns: src_facility_id, dst_facility_id, mode,
    month, path_length_m, weighted_avg_friction, total_cost, unreachable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    months = tuple(months)

    pairs = _load_facility_pairs(graph_con)
    if pairs.empty:
        logger.warning("connects_to has no pairs to evaluate")
        return pd.DataFrame(
            columns=[
                "src_facility_id", "dst_facility_id", "mode", "month",
                "path_length_m", "weighted_avg_friction", "total_cost",
                "unreachable",
            ]
        )

    # Cache the canonical profile from any friction raster.
    any_friction = next(iter(friction_stack.values()))
    with rasterio.open(any_friction) as src:
        profile = src.profile.copy()

    records: list[dict] = []
    plane_rate = BASELINE_RATES["Plane"]

    for row in pairs.itertuples(index=False):
        mode = _pair_mode(row.src_method, row.dst_method)
        if mode is None:
            logger.debug("skipping pair (%s,%s): method mismatch (%s,%s)",
                         row.src, row.dst, row.src_method, row.dst_method)
            continue

        if mode == "plane":
            length_m = _haversine_m(row.src_lon, row.src_lat, row.dst_lon, row.dst_lat)
            miles = length_m / METERS_PER_MILE
            records.append(dict(
                src_facility_id=int(row.src),
                dst_facility_id=int(row.dst),
                mode="plane",
                month=0,
                path_length_m=length_m,
                weighted_avg_friction=1.0,
                total_cost=miles * plane_rate,
                unreachable=False,
            ))
            continue

        # Ground modes: compute every month. For Road pairs also compute
        # ice_road across every month; Barge pairs only compute barge.
        mode_set = ("overland", "ice_road") if mode == "overland" else ("barge",)
        for m in mode_set:
            rate_m = BASELINE_RATES[RATE_KEY_BY_MODE[m]]
            for month in months:
                key = (m, month)
                if key not in friction_stack:
                    logger.warning("friction stack missing %s; skipping", key)
                    continue
                length_m, avg_fric, unreachable = _compute_one_edge(
                    int(row.src), int(row.dst),
                    row.src_lon, row.src_lat, row.dst_lon, row.dst_lat,
                    friction_stack[key], profile,
                )
                if unreachable:
                    records.append(dict(
                        src_facility_id=int(row.src),
                        dst_facility_id=int(row.dst),
                        mode=m,
                        month=month,
                        path_length_m=None,
                        weighted_avg_friction=None,
                        total_cost=None,
                        unreachable=True,
                    ))
                else:
                    miles = length_m / METERS_PER_MILE
                    records.append(dict(
                        src_facility_id=int(row.src),
                        dst_facility_id=int(row.dst),
                        mode=m,
                        month=month,
                        path_length_m=length_m,
                        weighted_avg_friction=avg_fric,
                        total_cost=miles * avg_fric * rate_m,
                        unreachable=False,
                    ))

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Graph writer
# ---------------------------------------------------------------------------

def _ensure_mode_specific_edges_table(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS mode_specific_edges (
            src INTEGER,
            dst INTEGER,
            mode VARCHAR,
            month INTEGER,
            path_length_m DOUBLE,
            weighted_avg_friction DOUBLE,
            total_cost DOUBLE,
            unreachable BOOLEAN,
            PRIMARY KEY (src, dst, mode, month)
        )
        """
    )


def write_edges_to_graph(
    edge_df: pd.DataFrame,
    graph_con,
    repr_months: dict[str, int] = DEFAULT_REPR_MONTHS,
) -> None:
    """Persist edges to DuckDB.

    Writes all rows (reachable + unreachable, for auditability) to
    mode_specific_edges. Then backfills connects_to.avg_friction /
    path_length_miles / delivery_cost from the representative month per
    mode, in both directions (src->dst and dst->src), so tsp_model_graph.py
    consumes the new costs unchanged.
    """
    _ensure_mode_specific_edges_table(graph_con)

    if edge_df.empty:
        logger.info("no edges to write")
        return

    # Replace any existing rows for these (src,dst,mode,month) keys to keep
    # the write idempotent across reruns.
    graph_con.execute("BEGIN")
    try:
        graph_con.execute("DELETE FROM mode_specific_edges WHERE 1=1")
        graph_con.register("edge_df_tmp", edge_df.rename(columns={
            "src_facility_id": "src",
            "dst_facility_id": "dst",
        }))
        graph_con.execute(
            """
            INSERT INTO mode_specific_edges
            SELECT src, dst, mode, month, path_length_m,
                   weighted_avg_friction, total_cost, unreachable
            FROM edge_df_tmp
            """
        )
        graph_con.unregister("edge_df_tmp")

        unreachable_count = int(edge_df["unreachable"].fillna(True).sum())
        logger.info(
            "inserted %d rows into mode_specific_edges (%d unreachable)",
            len(edge_df), unreachable_count,
        )

        # Backfill connects_to for the representative month of each mode.
        # Skip 'plane' (handled separately below) and 'ice_road' (not a
        # delivery_method in current data; ice_road rows live only in
        # mode_specific_edges).
        for mode, month in repr_months.items():
            if mode == "ice_road":
                continue
            mask = (edge_df["mode"] == mode) & (edge_df["month"] == month) & (~edge_df["unreachable"])
            subset = edge_df[mask]
            for r in subset.itertuples(index=False):
                miles = r.path_length_m / METERS_PER_MILE
                _update_connects_to_row(
                    graph_con, int(r.src_facility_id), int(r.dst_facility_id),
                    miles, float(r.weighted_avg_friction), float(r.total_cost),
                )

        # Plane edges: month=0 sentinel in records; populate connects_to.
        plane_subset = edge_df[(edge_df["mode"] == "plane") & (~edge_df["unreachable"])]
        for r in plane_subset.itertuples(index=False):
            miles = r.path_length_m / METERS_PER_MILE
            _update_connects_to_row(
                graph_con, int(r.src_facility_id), int(r.dst_facility_id),
                miles, float(r.weighted_avg_friction), float(r.total_cost),
            )

        graph_con.execute("COMMIT")
    except Exception:
        graph_con.execute("ROLLBACK")
        raise


def _update_connects_to_row(con, src: int, dst: int, miles: float, friction: float, cost: float) -> None:
    """Update connects_to in both directions (edges are symmetric)."""
    con.execute(
        """
        UPDATE connects_to
        SET avg_friction = ?, path_length_miles = ?, delivery_cost = ?
        WHERE (src = ? AND dst = ?) OR (src = ? AND dst = ?)
        """,
        [friction, miles, cost, src, dst, dst, src],
    )
