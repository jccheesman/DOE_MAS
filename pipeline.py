'''Pipeline for MAS-DOE

This module accesses files that are created by notebooks and
need to be access by successive runs'''


import pandas as pd
import os
import geopandas as gpd
import json
import shutil
import subprocess
import sys
import time
import logging
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

def set_cwd(path):
	print(os.getcwd())
	desired_path = path

	if os.getcwd() != desired_path:
  		os.chdir(desired_path)
  		print(os.getcwd())

def get_json(path):
	with open(path, 'r') as file:
		json_file = json.load(file)
	return json_file


def get_csv(path):
	csv_file = pd.read_csv(path)
	return csv_file

def get_bulk_fuel_lat_long_csv(path):
	csv_file = pd.read_csv(path, usecols =['ASTFacilityID','ASTFacilityLongitude', 'ASTFacilityLatitude', 'Delivery_method'])
	return csv_file

def get_shapefile(path):
	shapefile = gpd.read_file(path)
	return shapefile


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
#
# Agents call `pipeline.get_llm(tier)` where tier is "haiku" or "sonnet".
# The provider is chosen by the LLM_PROVIDER env var:
#   - "openrouter" (default): route Claude Haiku 4.5 / Sonnet 4.6 via OpenRouter
#   - "ollama":               legacy local path, uses OLLAMA_MODEL for both tiers

OPENROUTER_MODELS = {
    "haiku":  "openrouter/anthropic/claude-haiku-4.5",
    "sonnet": "openrouter/anthropic/claude-sonnet-4.6",
}


def get_llm(tier="haiku"):
	"""Return a configured CrewAI LLM instance.

	Args:
		tier: "haiku" for fast/cheap agents, "sonnet" for reasoning/writing
			agents. Ignored when LLM_PROVIDER=ollama.
	"""
	from crewai import LLM

	provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

	if provider == "openrouter":
		if tier not in OPENROUTER_MODELS:
			raise ValueError(
				f"Unknown LLM tier {tier!r}; expected one of "
				f"{sorted(OPENROUTER_MODELS)}"
			)
		api_key = os.getenv("OPENROUTER_API_KEY")
		if not api_key:
			raise RuntimeError(
				"OPENROUTER_API_KEY is not set. Export it before running the "
				"pipeline, or set LLM_PROVIDER=ollama to use the legacy local "
				"Ollama backend."
			)
		model = OPENROUTER_MODELS[tier]
		print(f"LLM configured: {model} (tier={tier}) via OpenRouter")
		return LLM(
			model=model,
			api_key=api_key,
			timeout=3600,
			num_retries=3,
		)

	if provider == "ollama":
		api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
		model_name = os.getenv("OLLAMA_MODEL", "llama3.1:70b")
		os.environ["OLLAMA_API_BASE"] = api_base
		model = f"ollama/{model_name}" if not model_name.startswith("ollama/") else model_name
		print(f"LLM configured: {model} at {api_base} (tier={tier} ignored for Ollama)")
		return LLM(
			model=model,
			base_url=api_base,
			timeout=3600,       # 60 min — 70B model can be slow on complex prompts
			num_retries=3,      # retry on transient connection errors
		)

	raise ValueError(
		f"Unknown LLM_PROVIDER={provider!r}; expected 'openrouter' or 'ollama'."
	)


def check_ollama(max_retries=3, wait_seconds=10):
    """Verify Ollama is responsive; attempt restart if not."""
    api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    url = f"{api_base}/api/tags"

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.urlopen(url, timeout=15)
            req.close()
            print(f"Ollama health check passed (attempt {attempt})")
            return True
        except (urllib.error.URLError, OSError) as e:
            print(f"Ollama health check failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print("Attempting to restart Ollama...")
                subprocess.run(["systemctl", "restart", "ollama"], capture_output=True)
                print(f"Waiting {wait_seconds}s for Ollama to start...")
                time.sleep(wait_seconds)

    raise ConnectionError(
        f"Ollama is not responding at {api_base} after {max_retries} attempts. "
        "Check that the Ollama service is running."
    )


def check_openrouter():
    """Verify OpenRouter is reachable and the API key is set."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ConnectionError(
            "OPENROUTER_API_KEY is not set. Export your OpenRouter key before "
            "running the pipeline, or set LLM_PROVIDER=ollama to use the "
            "legacy local backend."
        )

    url = "https://openrouter.ai/api/v1/models"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        resp = urllib.request.urlopen(req, timeout=15)
        resp.close()
        print("OpenRouter health check passed")
        return True
    except (urllib.error.URLError, OSError) as e:
        raise ConnectionError(
            f"OpenRouter is not reachable at {url}: {e}. "
            "Check network connectivity and API key validity."
        )


def check_llm():
    """Provider-aware LLM health check; delegates based on LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    if provider == "openrouter":
        return check_openrouter()
    if provider == "ollama":
        return check_ollama()
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}; expected 'openrouter' or 'ollama'."
    )


def setup_logging(log_dir="outputs"):
    """Configure logging to both console and a timestamped log file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    print(f"Logging to: {log_file}")
    return log_file


def save_json(source_folder=".", output_folder_name="outputs", pattern="*.json"):
    """
    Copy all JSON files from source folder to output folder.
    
    Args:
        source_folder: Folder to search for JSON files (default: current directory)
        output_folder_name: Destination folder name
        pattern: File pattern to match (default: "*.json")
    """
    source_path = Path(source_folder)
    output_path = Path(output_folder_name)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(source_path.glob(pattern))
    
    # Copy files
    for json_file in json_files:
        if json_file.is_file():
            dest_file = output_path / json_file.name
            counter = 1
            while dest_file.exists():
                dest_file = output_path / f"{json_file.stem}_{counter}{json_file.suffix}"
                counter += 1
            shutil.copy2(json_file, dest_file)
            print(f"Copied: {dest_file.name}")
    
    print(f"Total files copied: {len(json_files)}")


def get_duckdb_connection(db_path='regionalization.duckdb', read_only=False):
    """Connect to the shared DuckDB graph database.

    Args:
        db_path: Path to the DuckDB database file
        read_only: If True, open in read-only mode (safe for concurrent reads)

    Returns:
        duckdb.DuckDBPyConnection
    """
    import duckdb
    return duckdb.connect(db_path, read_only=read_only)


# ---------------------------------------------------------------------------
# Friction surface helpers
# ---------------------------------------------------------------------------

def get_raster_dir():
    """Return the path to the GEE raster directory."""
    return os.getenv("RASTER_DIR", "./rasters")


def get_vector_dir():
    """Return the path to the vector data directory."""
    return os.getenv("VECTOR_DIR", "./vectors")


def get_whitebox_wbt():
    """Return a configured WhiteboxTools instance."""
    from whitebox import WhiteboxTools
    wbt = WhiteboxTools()
    wbt.set_verbose_mode(False)
    work_dir = os.path.join(get_raster_dir(), "wbt_work")
    os.makedirs(work_dir, exist_ok=True)
    wbt.set_working_dir(work_dir)
    return wbt


def reproject_facilities(con):
    """Reproject facility lon/lat to EPSG:3413 and update the facilities table.

    Reads facilities from DuckDB, projects WGS84 coordinates to polar
    stereographic, and writes x_3413 / y_3413 back to the table.

    Args:
        con: DuckDB connection (read-write)
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)

    rows = con.execute(
        "SELECT facility_id, longitude, latitude FROM facilities"
    ).fetchall()

    for fid, lon, lat in rows:
        x, y = transformer.transform(lon, lat)
        con.execute(
            "UPDATE facilities SET x_3413 = ?, y_3413 = ? WHERE facility_id = ?",
            [x, y, fid],
        )

