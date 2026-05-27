# Friction-surface pipeline (agent-free)

Deterministic, reproducible construction of mode-specific monthly friction
surfaces and cost-distance edges for the Alaska bulk-fuel logistics graph.

This stage is **agent-free**: no CrewAI, no LLM calls. The three modules
that make up the pipeline (`friction_surface.py`, `routing_wbt.py`,
`run_friction_pipeline.py`) import nothing from CrewAI. Agents remain in
scope downstream (`market_cost_analysis.py`).

## Inputs

All inputs are assumed preprocessed to a common CRS (EPSG:3413), resolution
(150 m), and extent. Paths are resolved under `$RASTER_DIR` (default
`./rasters`).

| Path | Description | Units |
|---|---|---|
| `slope.tif` | Per-pixel slope from FabDEM | degrees, float32 |
| `lulc.tif`  | Dynamic World modal class | int (0–8) |
| `permafrost.tif` | Near-surface permafrost extent (Pastick et al.) | 0–1 or 0–100; auto-normalized |
| `sea_ice/sea_ice_{01..12}.tif` | AOOS Historical Sea Ice Atlas monthly climatology | 0–1 or 0–100; auto-normalized |
| `river_ice/river_ice_{01..12}.tif` | Brown et al. river ice phenology monthly probability | 0–1 or 0–100; auto-normalized |

Dynamic World class codes used:
`0 water, 1 trees, 2 grass, 3 flooded_vegetation, 4 crops, 5 shrub_scrub, 6 built_area, 7 bare_ground, 8 snow_ice`.

## Outputs

- **Friction rasters** written to `$RASTER_DIR/friction/{mode}_{MM}.tif`
  (36 files total). float32, NoData = -9999, LZW-compressed.
- **`mode_specific_edges`** DuckDB table with one row per
  `(src, dst, mode, month)`, including `path_length_m`,
  `weighted_avg_friction`, `total_cost`, and an `unreachable` boolean.
- **`connects_to`** backfill: `avg_friction`, `path_length_miles`,
  `delivery_cost` populated for the representative month per mode
  (overland=Jun, barge=Jul) and for Plane edges (Haversine).
  `tsp_model_graph.py` consumes these columns unchanged.

## Design decisions

### LULC and permafrost are independent factors

The current (CrewAI-driven) pipeline combines LULC and permafrost into a
single `PERMAFROST_LULC_MATRIX` baked into a year-round friction layer.
That confounds two physically distinct effects: LULC affects trafficability
year-round, while permafrost only affects trafficability during the thaw
window. Combining them means the same matrix multiplier penalizes a built
area equally in February (frozen, trafficable) and August (thawed, soft) —
which is wrong.

The new pipeline keeps them separate:

- **LULC** enters the year-round **static base**.
- **Permafrost** is a **seasonal modifier**: 1.0 in winter (Nov–Apr),
  linearly scales to `PERMAFROST_MAX_SHOULDER = 1.15` at 100% extent in
  May/Oct, and to `PERMAFROST_MAX_SUMMER = 1.40` at 100% extent in Jun–Sep.

This also reflects mode-specific reality: permafrost is irrelevant to barge
and ice-road travel, so those modes ignore the modifier.

### NoData is the sole impassability mechanism

Impassable pixels are stored as `-9999` (the raster's nodata value), not as
a sentinel high number (`999`/`9999`). WhiteboxTools `CostDistance` treats
NoData as non-traversible and routes around it.

Applied uniformly:

- Overland over water → NoData.
- Barge over land → NoData.
- Barge over ice-covered water (combined sea/river ice probability above
  `ICE_PROB_THRESHOLD = 0.5`) → NoData.
- Ice-road over open water (probability below the threshold) → NoData.

A raster represents only what the mode can actually traverse. Numeric
friction values are reserved for traversible pixels.

### 12 monthly surfaces per mode (36 total)

Sea ice and river ice climatologies are seasonal, not binary winter/summer
events. The pipeline writes one friction raster per (mode, month), so any
representative month can be selected downstream without rebuilding. The
representative months wired into the TSP-facing backfill default to
overland=June, barge=July, ice_road=February.

## How to run

```bash
# Build all 36 surfaces and compute edges in one pass:
python run_friction_pipeline.py

# Reuse existing surfaces and only recompute edges (useful after tweaking
# constants in routing_wbt.py):
python run_friction_pipeline.py --skip-surfaces
```

The DuckDB graph database (`regionalization.duckdb`) must already exist
with `facilities`, `connects_to`, and `uses_method` populated by
`regionalization_graph.py`.

## Runtime expectations

The dominant cost is the per-pair WhiteboxTools `CostDistance` +
`CostPathway` round-trip. Cost-distance is single-threaded in the WBT
build shipped with this repo. Approximate wall-clock for typical Alaska
inputs (~700 facilities, ~5–10 k same-region same-method pairs):

- Surface stack (36 rasters @ 150 m): a few minutes on commodity hardware.
- Edge computation: scales linearly with pair count × month count. Most
  pairs are Plane (Haversine, near-instant) or Barge (small-region cohorts).
  Plan for an hour or more on a full statewide run.

If memory becomes a bottleneck, `build_mode_friction` accepts a `window`
argument for tile-based processing; this is off by default.

## Module map

| File | Purpose |
|---|---|
| `friction_surface.py` | Raster reclassifications and 36-surface writer. Pure numpy / rasterio. |
| `routing_wbt.py` | WBT cost-distance wrapper, edge computation, DuckDB writer (`mode_specific_edges` + `connects_to` backfill). |
| `run_friction_pipeline.py` | Orchestration. Input validation, surface build, edge compute, DuckDB write, summary printout. |
