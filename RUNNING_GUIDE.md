# DOE MAS — Running Guide

End-to-end walkthrough for running the Alaska fuel delivery multi-agent framework, from raw data acquisition through final TSP route reports.

## Overview

The framework runs in **two environments**:

1. **Google Colab** — runs `notebooks/gee_preprocessing.ipynb` to pull geospatial data from Google Earth Engine and external sources, then exports aligned rasters to Google Drive.
2. **Jetstream2 VM** — runs the main pipeline (`run_graph.py`) with the Ollama LLM server and DuckDB graph database. The friction surface computation and all CrewAI agents live here.

```
┌─────────────────────┐      ┌──────────────────────┐
│  Colab (GEE)        │      │  Jetstream2 VM       │
│                     │      │                      │
│  gee_preprocessing  │─────▶│  friction_surface.py │
│   → rasters/        │ rast │  friction_agents.py  │
│   → vectors/        │ vect │  tsp_model_graph.py  │
│                     │      │                      │
└─────────────────────┘      └──────────────────────┘
```

---

## Prerequisites

Before starting, make sure you have:

- [ ] A **Google account** with access to Colab and Google Drive (~2 GB free)
- [ ] A **Google Earth Engine project** (free, sign up at https://earthengine.google.com/)
- [ ] A **Jetstream2 allocation** with access to a GPU-capable VM (A100 recommended)
- [ ] This repository cloned or the code files ready to upload to the VM

### Data files you'll need to download manually

Several datasets can't be automated — download them to your computer first:

| File | Source | Notes |
|------|--------|-------|
| `airports.csv` | https://davidmegginson.github.io/ourairports-data/airports.csv | ~13 MB, all global airports |
| `NWN_Waterway_Network_Lines.zip` | https://geospatial-usace.opendata.arcgis.com/datasets/ace7645d305647448a84492a3b909d48_1 | Click "Download" → Shapefile |
| `AK_Ports_and_Harbors.zip` | AK DOT&PF | Shapefile of Alaska ports/harbors |
| `Utilities_Bulk_Fuel_Inventory.csv` | Alaska Energy Authority | Facility site locations (already in repo) |
| `Alaska_Energy_Authority_Library.shp` | Alaska Energy Authority | Regional boundaries (already in repo) |

---

## Phase 1: GEE Preprocessing (Google Colab)

**Goal:** Produce 6 aligned rasters + 3 vector files on Google Drive.

### 1.1 Upload data files to Google Drive

Create the following folder structure on your Drive:

```
My Drive/
├── friction_surface_exports/       ← GEE will write rasters here
├── Masters/DOE-MAS/Version of Code/GDB_FL_1.0/codes/
│   ├── airports.csv
│   ├── NWN_Waterway_Network_Lines.zip
│   ├── AK_Ports_and_Harbors.zip
│   ├── Utilities_Bulk_Fuel_Inventory.csv
│   └── notebooks/
│       └── gee_preprocessing.ipynb
```

(If you use a different folder layout, you'll need to update the file paths inside Cells 3, 17, 19, and 21 of the notebook.)

### 1.2 Open the notebook in Colab

1. Upload `notebooks/gee_preprocessing.ipynb` to Colab (File → Upload notebook)
2. Or open directly from GitHub via Colab's File → Open notebook → GitHub tab

### 1.3 Run the setup cells (1, 2, 3, 4)

**Cell 2 — Imports & GEE authentication:**
- First run will prompt for GEE authentication (follow the link, sign in, paste token)
- Update `ee.Initialize(project='...')` with **your own GEE project ID**
- If any imports fail, uncomment the `!pip install` line and run it once

**Cell 3 — Drive mount:**
- Authorizes Colab to read/write your Google Drive
- Sets `RASTER_DIR` to `/content/drive/My Drive/friction_surface_exports`
- If your Drive folder is different, edit this cell before running

**Cell 4 — `export_aligned_raster()` helper:**
- Just run it (no user input needed)

**Cell 5 — Helper functions for raster processing:**
- Defines `get_alaska_boundary_3413`, `reproject_raster_to_3413`, `clip_raster_to_alaska`, `resample_to_reference_grid`, `rasterize_vector_to_reference`
- Just run it

### 1.4 Run the GEE export cells (6, 8, 14)

These cells start **asynchronous** GEE export tasks. They return immediately but the exports run on Google's servers in the background:

- **Cell 6 — LULC** (Dynamic World modal 2023)
- **Cell 8 — DEM + Slope** (FabDEM)
- **Cell 14 — Roads presence** (GRIP4)

After running each, you'll see `Export started: ...`. **Watch the GEE Tasks tab** (https://code.earthengine.google.com/tasks) to monitor progress — each task takes 5–15 minutes.

### 1.5 Run the local processing cells (10, 16, 17, 19, 21)

These process data locally on the Colab VM (not async):

**Cell 10 — Permafrost (Pangaea):**
- Downloads Obu et al. 2018 PERPROB (~100 MB)
- Reprojects, clips, reclassifies to 4 zones
- Saves at native 1 km resolution — needs Cell 12 to resample later

**Cell 16 — NWN navigable waterways:**
- Extracts zip from Drive
- Reprojects to EPSG:3413
- Clips to Alaska boundary
- Rasterizes to match the LULC reference grid
- **Requires `lulc_alaska_modal.tif` to already be downloaded** — run this after Cell 6's GEE task completes

**Cell 17/18 — Airports (OurAirports CSV):**
- Loads `airports.csv` from Drive
- Filters to `iso_region == 'US-AK'`
- Exports as `airports_alaska.geojson`

**Cell 19/20 — Ports (AK DOT&PF):**
- Extracts port shapefile from Drive zip
- Classifies as full port vs beach landing
- Exports as `ports_alaska.geojson`

**Cell 21/22 — Bulk fuel facilities:**
- Loads `Utilities_Bulk_Fuel_Inventory.csv`
- Reprojects to EPSG:3413
- Exports as `facilities_alaska.geojson`

### 1.6 Wait for GEE exports to finish

After starting Cells 6, 8, and 14, go to https://code.earthengine.google.com/tasks and wait until all 4 tasks (LULC, DEM, Slope, Roads_Presence) show **COMPLETED**.

You can also uncomment the `monitor_tasks(export_tasks)` line in Cell 24 and run it to block until all tasks finish.

Once complete, check your Drive folder `/content/drive/My Drive/friction_surface_exports/` — you should see:
- `lulc_alaska_modal.tif`
- `slope_alaska.tif`
- `dem_alaska.tif`
- `roads_presence_alaska.tif`

### 1.7 Run Cell 12 (permafrost resample to 150m)

Now that `lulc_alaska_modal.tif` exists, run Cell 12 to resample the 1 km permafrost raster to match the 150 m reference grid. It will **overwrite** `permafrost_alaska.tif` in place.

### 1.8 Run Cell 25 (alignment verification)

This reads every raster in `RASTER_DIR` and prints a table showing shape, CRS, resolution, and value range. **All 6 rasters should have identical shape (24369, 13768), CRS EPSG:3413, and resolution 150 m.** The cell ends with `"ALL RASTERS ALIGNED"` if everything lines up.

### 1.9 Download artifacts to local machine

From your Drive folder `/content/drive/My Drive/friction_surface_exports/`, download all `.tif` files to your local machine. You'll transfer them to Jetstream2 in Phase 3.

Also download the 3 GeoJSONs from `./vectors/` (created in Colab's local filesystem — check the left sidebar Files panel):
- `airports_alaska.geojson`
- `ports_alaska.geojson`
- `facilities_alaska.geojson`

---

## Phase 2: Jetstream2 VM Setup

**Goal:** A running Jetstream2 VM with Python environment, Ollama LLM server, and the project code.

### 2.1 Create or resume the VM

**First-time setup:** Follow `JETSTREAM_SETUP_GUIDE.md` sections A.1 through A.9 to:
- Create an Ubuntu VM with NVIDIA A100
- Attach a persistent volume
- Set up Python venv
- Install dependencies from `requirements.txt`
- Install Ollama + pull `llama3.1:70b`

**Resuming existing instance:** Follow `JETSTREAM_SETUP_GUIDE.md` section B:
```bash
module load miniforge
cd /media/volume/<your-volume-name>
source venv/bin/activate
sudo systemctl start ollama  # if not auto-started
```

### 2.2 Verify Ollama is running

```bash
curl http://localhost:11434/api/tags
```

Should return a JSON list of installed models with `llama3.1:70b` in it.

### 2.3 Ensure project code is up to date

If you've pulled new changes from GitHub:

```bash
cd /media/volume/<your-volume-name>
git pull origin main
```

Or upload individual modified files via Guacamole drag-and-drop, then move them from `/home/exouser/` to the volume directory.

---

## Phase 3: Transfer Data to Jetstream2

**Goal:** All GEE-exported rasters and vectors end up in `./rasters/` and `./vectors/` on the VM.

### 3.1 Create data directories

```bash
cd /media/volume/<your-volume-name>
mkdir -p rasters vectors
```

### 3.2 Upload files via Guacamole

Drag and drop these from your local machine into the Guacamole desktop:

**Rasters (into `rasters/`):**
- `lulc_alaska_modal.tif`
- `slope_alaska.tif`
- `dem_alaska.tif`
- `permafrost_alaska.tif`
- `roads_presence_alaska.tif`
- `rivers_alaska.tif`

**Vectors (into `vectors/`):**
- `airports_alaska.geojson`
- `ports_alaska.geojson`
- `facilities_alaska.geojson`

Then move them to the correct locations:

```bash
mv /home/exouser/*.tif /media/volume/<your-volume-name>/rasters/
mv /home/exouser/*.geojson /media/volume/<your-volume-name>/vectors/
```

### 3.3 Verify file presence

```bash
ls -lh rasters/
ls -lh vectors/
```

All 6 rasters and 3 vectors should be listed.

---

## Phase 4: Run the Main Pipeline

**Goal:** Execute `run_graph.py` to completion, producing graph database + JSON reports.

### 4.1 Start a tmux session

Always use tmux so the pipeline survives Guacamole disconnects:

```bash
tmux new -s pipeline
```

### 4.2 Run the pipeline

```bash
cd /media/volume/<your-volume-name>
python run_graph.py
```

The pipeline runs 6 steps sequentially (see `run_graph.py`):

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `regionalization_graph.py` | Builds DuckDB graph from facilities + regions; assigns delivery methods |
| 2 | `market_cost_analysis.py` | Market/cost analysis via CrewAI agents |
| 3 | `friction_surface.py` | Builds 3 friction rasters; runs WhiteboxTools cost_distance for every `connects_to` edge |
| 4 | `friction_agents.py` | Seasonal Enhancer + Cost Estimator + Validation agents update the graph |
| 5 | `tsp_model_graph.py` | TSP optimization using friction-weighted edge costs |
| 6 | Save outputs | Copies JSON reports to `outputs/` |

**Expected runtime:** 2–6 hours for a full run, mostly dominated by the LLM agent calls in Steps 2, 4, and 5 and the WhiteboxTools cost_distance runs in Step 3.

### 4.3 Detach safely

Press `Ctrl+B` then `D` to detach from tmux. The pipeline keeps running in the background.

### 4.4 Check progress

Reattach any time:

```bash
tmux attach -t pipeline
```

Or tail the log file directly:

```bash
tail -f outputs/pipeline_*.log
```

### 4.5 Handle crashes / resume

If the pipeline crashes or you need to restart, just re-run:

```bash
python run_graph.py
```

It reads `outputs/.checkpoint` and skips any steps that already finished. To force a complete fresh run, delete the checkpoint first:

```bash
rm outputs/.checkpoint
```

---

## Phase 5: Inspect Outputs

**Goal:** Confirm the pipeline produced meaningful results.

### 5.1 Check the output files

```bash
ls -lh outputs/
```

You should see:
- `pipeline_YYYYMMDD_HHMMSS.log` — full run log
- `market_cost_analysis_report.json` — Step 2 output
- `friction_analysis_report.json` — Step 4 output
- `tsp_final_report.json` — Step 5 output (final route analysis)
- `regionalization.duckdb` — graph database (in the project root, not `outputs/`)

### 5.2 Quick graph database inspection

```bash
python3 -c "
import duckdb
con = duckdb.connect('regionalization.duckdb', read_only=True)
print('Facilities:', con.execute('SELECT COUNT(*) FROM facilities').fetchone()[0])
print('Regions:   ', con.execute('SELECT COUNT(*) FROM regions').fetchone()[0])
print('Edges:     ', con.execute('SELECT COUNT(*) FROM connects_to').fetchone()[0])
print('Routes:    ', con.execute('SELECT COUNT(*) FROM part_of_route').fetchone()[0])
print()
print('Friction stats (mean per delivery method):')
rows = con.execute('''
    SELECT COALESCE(um.method_name, \"Unknown\") AS method,
           COUNT(*) AS edges,
           ROUND(AVG(ct.avg_friction), 3) AS mean_friction,
           ROUND(AVG(ct.delivery_cost), 0) AS mean_cost
    FROM connects_to ct
    LEFT JOIN uses_method um ON ct.src = um.facility_id
    WHERE ct.avg_friction IS NOT NULL
    GROUP BY um.method_name
''').fetchall()
for r in rows:
    print(' ', r)
con.close()
"
```

### 5.3 Run the visualization notebook (optional)

For interactive inspection of the friction surfaces and routes, open `notebooks/friction_surface.ipynb` on the Jetstream VM (via Jupyter, not Colab — it needs access to the `rasters/` directory and the DuckDB database):

```bash
cd /media/volume/<your-volume-name>
jupyter notebook notebooks/friction_surface.ipynb
```

The notebook produces 8 visualization types — see the [Friction Surface Notebook section](#friction-surface-notebook-cell-reference) below.

---

## Quick-start cheat sheet

For subsequent runs after everything is set up:

```bash
# 1. Resume VM, activate environment
module load miniforge
cd /media/volume/<your-volume-name>
source venv/bin/activate
sudo systemctl start ollama

# 2. Run pipeline in tmux
tmux new -s pipeline
python run_graph.py

# 3. Detach with Ctrl+B D, reattach with:
tmux attach -t pipeline

# 4. When finished, inspect outputs
cat outputs/tsp_final_report.json | python3 -m json.tool | head -50
```

---

## Friction Surface Notebook cell reference

`notebooks/friction_surface.ipynb` (runs on Jetstream2, not Colab):

| Cell | Purpose |
|------|---------|
| 1–2 | Imports + load GEE rasters |
| 3 | **Viz 1** — Input layer maps (LULC, slope, permafrost, roads, rivers, DEM) |
| 4 | Build composite friction rasters (road/barge/sky) |
| 5 | **Viz 2** — Composite friction surface maps |
| 6 | **Viz 3** — Facilities overlaid on road friction surface |
| 7 | Run least-cost path computation (calls `friction_surface.compute_paths_for_method`) |
| 8 | **Viz 4** — Least-cost path traces for sample edges |
| 9 | **Viz 5** — Haversine vs friction path comparison |
| 10 | **Viz 6** — Seasonal friction heatmaps |
| 11 | **Viz 7** — Regional friction summary bar charts |
| 12 | **Viz 8** — Delivery cost distributions |
| 13 | Summary statistics from the graph DB |

All plots save to `notebooks/*.png` alongside inline display.

---

## Troubleshooting

### GEE export tasks stuck at "RUNNING"

GEE sometimes takes 30+ minutes for large exports. Check task status at https://code.earthengine.google.com/tasks. If it's been over an hour, cancel and retry — sometimes GEE queues drain slowly.

### `FileNotFoundError: Reference raster not found`

Cell 12 (permafrost resample) or Cell 16 (NWN waterways) was run before `lulc_alaska_modal.tif` was downloaded from Drive to `RASTER_DIR`. Make sure the GEE task finished, then verify the file exists:

```python
import os
print(os.listdir(os.environ.get('RASTER_DIR', './rasters')))
```

### Ollama timeouts in agent steps

The pipeline has a 30-minute LLM timeout baked in. If you still hit timeouts:

```bash
# Check Ollama is healthy
curl http://localhost:11434/api/tags
sudo systemctl status ollama

# Increase timeout in pipeline.py get_llm() if needed
```

### `KeyError: 'roads_type'`

You're running an old version of `friction_surface.py`. Pull the latest:

```bash
git pull origin main
```

### WhiteboxTools cost_distance errors

Check that WhiteboxTools is installed and all 6 rasters in `rasters/` have identical shape and CRS. Run the alignment verification cell in `gee_preprocessing.ipynb` or the top of `friction_surface.ipynb` to confirm.

### Disk full on root filesystem

Ollama models should be on the volume, not root. Check with `df -h`. If models ended up on root, move them:

```bash
sudo systemctl stop ollama
sudo mv /usr/share/ollama/.ollama /mnt/ollama_volume/ollama/
sudo ln -s /mnt/ollama_volume/ollama /usr/share/ollama/.ollama
sudo systemctl start ollama
```

### Pipeline checkpoint issues

If you want to re-run a specific step only, edit `outputs/.checkpoint` and remove that step from the `completed_steps` list. Or delete the file entirely for a fresh run.

---

## File locations reference

| Item | Path on Jetstream2 |
|------|---------------------|
| Project root | `/media/volume/<name>/` |
| Python venv | `/media/volume/<name>/venv/` |
| Ollama models | `/mnt/ollama_volume/ollama/models/` |
| Input rasters | `/media/volume/<name>/rasters/*.tif` |
| Input vectors | `/media/volume/<name>/vectors/*.geojson` |
| Graph database | `/media/volume/<name>/regionalization.duckdb` |
| Pipeline logs | `/media/volume/<name>/outputs/pipeline_*.log` |
| Output reports | `/media/volume/<name>/outputs/*.json` |
| Checkpoint file | `/media/volume/<name>/outputs/.checkpoint` |

---

## Related docs

- `JETSTREAM_SETUP_GUIDE.md` — First-time VM setup (create instance, install Python, Ollama, etc.)
- `README.md` — High-level project overview and file listing
- `requirements.txt` — Python dependencies
- `installations.txt` — Supplementary install commands
