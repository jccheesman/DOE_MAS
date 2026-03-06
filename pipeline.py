'''Pipeline for MAS-DOE

This module accesses files that are created by notebooks and
need to be access by successive runs'''


import pandas as pd
import os
import geopandas as gpd 
import json
import shutil
from pathlib import Path 

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


def get_llm():
	"""Return a configured LLM instance using Ollama."""
	from crewai import LLM

	api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
	model_name = os.getenv("OLLAMA_MODEL", "llama3.1:70b")

	os.environ["OLLAMA_API_BASE"] = api_base

	model = f"ollama/{model_name}" if not model_name.startswith("ollama/") else model_name

	print(f"LLM configured: {model} at {api_base}")
	return LLM(model=model, base_url=api_base)


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

