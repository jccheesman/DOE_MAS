# -*- coding: utf-8 -*-
"""regionalization_graph.py

DuckDB Graph Database version of the regionalization module.

Uses DuckDB with the DuckPGQ extension to model bulk fuel facility
regionalization as a property graph, enabling graph-based queries
for adjacency, connectivity, and regional analysis.

# Regional Logistics Coordinator (Graph DB)

Goal: To create a graph database of bulk fuel facility sites grouped into
pre-determined regions, with adjacency relationships between regions.

Input data:
- Bulk Fuel Facility Sites (csv)
- Alaska Energy Development Regions (shapefile)

Graph Schema:
- Nodes: regions, facilities
- Edges: located_in (facility -> region), adjacent_to (region <-> region)

Agents:
- Delivery Method Coordinator: Assigns delivery methods to sites
- Logistics Coordinator: Assesses route risks

Outputs:
- A grouped dictionary with lat, long, delivery method and region
- A formatted report on preliminary risk assessments
- A DuckDB graph database with full regional connectivity
"""

# Imports
import os
import json
import duckdb
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import crewai
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool

import pipeline

# Setting Working Directory
pipeline.set_cwd('/media/volume/Preliminary_mas_runs')

# Retrieving Data
bulk_fuel_data = pd.read_csv(
    'Utilities_Bulk_Fuel_Inventory.csv',
    usecols=['ASTFacilityID', 'ASTFacilityLongitude', 'ASTFacilityLatitude']
)
regional_data = gpd.read_file(
    'Alaska_Energy_Authority_Library/Alaska_Energy_Authority_Library.shp'
)
bulk_fuel_csv_path = 'Utilities_Bulk_Fuel_Inventory.csv'
shapefile_path = 'Alaska_Energy_Authority_Library/Alaska_Energy_Authority_Library.shp'

# Globals
global final_regionalized_dict
global bulk_fuel_dict_with_regions
global unassigned_dict
final_regionalized_dict = {}
bulk_fuel_dict_with_regions = {}
unassigned_dict = {}


# ---------------------------------------------------------------------------
# DuckDB Graph Database Setup
# ---------------------------------------------------------------------------

DB_PATH = 'regionalization.duckdb'


def init_duckdb_graph():
    """Initialize DuckDB with graph schema using DuckPGQ extension.

    Creates a persistent DuckDB database file (regionalization.duckdb) with:
    - Node tables: regions, facilities, delivery_methods
    - Edge tables: located_in (facility -> region),
                   uses_method (facility -> delivery_method),
                   adjacent_to (region <-> region)

    Returns:
        duckdb.DuckDBPyConnection: Configured connection with graph schema
    """
    print("Initializing DuckDB graph database...")
    db_version = duckdb.__version__

    # Remove stale database file from a previous run so we start fresh.
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # Enable allow_unsigned_extensions up front so we can fall back to
    # the DuckPGQ S3 repository without recreating the connection.
    con = duckdb.connect(
        database=DB_PATH,
        config={'allow_unsigned_extensions': 'true'}
    )

    # Load DuckPGQ extension for graph support.
    # Try community repository first; fall back to DuckPGQ S3 repository
    # if the extension hasn't been published for this DuckDB version yet.
    installed = False

    # 1. Community repository (signed, preferred)
    try:
        con.execute("INSTALL duckpgq FROM community;")
        con.execute("LOAD duckpgq;")
        installed = True
    except Exception as e:
        print(f"Community install failed ({e}), trying DuckPGQ S3 repository...")

    # 2. DuckPGQ S3 repository (unsigned, latest builds)
    if not installed:
        try:
            con.execute(
                "SET custom_extension_repository = "
                "'http://duckpgq.s3.eu-north-1.amazonaws.com';"
            )
            con.execute("FORCE INSTALL 'duckpgq';")
            con.execute("LOAD 'duckpgq';")
            installed = True
        except Exception as e:
            print(f"DuckPGQ S3 install failed ({e}).")

    if not installed:
        raise RuntimeError(
            f"Failed to install DuckPGQ extension for DuckDB v{db_version}. "
            f"The extension may not yet be available for this DuckDB version. "
            f"Try pinning duckdb to an earlier version, e.g.: "
            f"pip install 'duckdb>=1.1.3,<{db_version}'"
        )
    print("DuckPGQ extension loaded successfully.")

    # --- Node tables ---

    # Regions table
    con.execute("""
        CREATE TABLE regions (
            region_name VARCHAR PRIMARY KEY,
            num_facilities INTEGER DEFAULT 0
        )
    """)

    # Facilities table
    con.execute("""
        CREATE TABLE facilities (
            facility_id INTEGER PRIMARY KEY,
            longitude DOUBLE,
            latitude DOUBLE,
            delivery_method VARCHAR,
            community_name VARCHAR
        )
    """)

    # Delivery methods node table
    con.execute("""
        CREATE TABLE delivery_methods (
            method_name VARCHAR PRIMARY KEY
        )
    """)

    # --- Edge tables ---

    # located_in: facility -> region
    con.execute("""
        CREATE TABLE located_in (
            facility_id INTEGER REFERENCES facilities(facility_id),
            region_name VARCHAR REFERENCES regions(region_name),
            PRIMARY KEY (facility_id, region_name)
        )
    """)

    # uses_method: facility -> delivery_method
    con.execute("""
        CREATE TABLE uses_method (
            facility_id INTEGER REFERENCES facilities(facility_id),
            method_name VARCHAR REFERENCES delivery_methods(method_name),
            PRIMARY KEY (facility_id)
        )
    """)

    # adjacent_to: region <-> region (shared borders)
    con.execute("""
        CREATE TABLE adjacent_to (
            region_a VARCHAR REFERENCES regions(region_name),
            region_b VARCHAR REFERENCES regions(region_name),
            shared_border_length DOUBLE DEFAULT 0.0,
            PRIMARY KEY (region_a, region_b)
        )
    """)

    # --- Create property graph ---
    con.execute("""
        CREATE PROPERTY GRAPH fuel_network
        VERTEX TABLES (
            regions LABEL region,
            facilities LABEL facility,
            delivery_methods LABEL delivery_method
        )
        EDGE TABLES (
            located_in
                SOURCE KEY (facility_id) REFERENCES facilities (facility_id)
                DESTINATION KEY (region_name) REFERENCES regions (region_name)
                LABEL located_in,
            uses_method
                SOURCE KEY (facility_id) REFERENCES facilities (facility_id)
                DESTINATION KEY (method_name) REFERENCES delivery_methods (method_name)
                LABEL uses_method,
            adjacent_to
                SOURCE KEY (region_a) REFERENCES regions (region_name)
                DESTINATION KEY (region_b) REFERENCES regions (region_name)
                LABEL adjacent_to
        )
    """)

    print("Graph schema created successfully.")
    return con


def load_regions_into_db(shapefile_path, region_column, con):
    """Load all region names from the shapefile into the regions table.

    This MUST be called before group_sites_by_region to ensure foreign key
    constraints on the located_in table are satisfied.

    Parameters:
        shapefile_path: Path to Alaska Energy Development Regions shapefile
        region_column: Column name containing region names
        con: DuckDB connection
    """
    regional_shapefile = gpd.read_file(shapefile_path)
    unique_regions = regional_shapefile[region_column].unique()

    for region_name in unique_regions:
        con.execute(
            "INSERT INTO regions (region_name) VALUES (?) ON CONFLICT DO NOTHING",
            [region_name]
        )

    count = con.execute("SELECT COUNT(*) FROM regions").fetchone()[0]
    print(f"Loaded {count} regions into graph database.")


# ---------------------------------------------------------------------------
# Spatial Join + Graph Loading
# ---------------------------------------------------------------------------

def group_sites_by_region(bulk_fuel_csv_path, shapefile_path, region_column, con):
    """Perform spatial join and load facility-region relationships into DuckDB.

    Takes bulk fuel facility data and Alaska regional shapefile, performs
    spatial join, populates the graph database, and builds in-memory
    dictionaries for downstream use.

    Parameters:
        bulk_fuel_csv_path: Path to CSV with bulk fuel facility data
        shapefile_path: Path to Alaska Energy Development Regions shapefile
        region_column: Name of the region column in the shapefile
        con: DuckDB connection with graph schema
    """
    global bulk_fuel_dict_with_regions, unassigned_dict

    # Process input data
    bulk_fuel_data = pd.read_csv(
        bulk_fuel_csv_path,
        usecols=[
            'ASTFacilityID', 'ASTFacilityLongitude',
            'ASTFacilityLatitude', 'Delivery_method',
            'CommunityName'
        ]
    )
    regional_shapefile = gpd.read_file(shapefile_path)
    bulk_fuel_data = bulk_fuel_data.dropna(
        axis=0, how='all',
        subset=['ASTFacilityLongitude', 'ASTFacilityLatitude']
    )

    # Generate synthetic IDs for rows missing ASTFacilityID so no data is lost.
    # Start synthetic IDs above the max existing ID to avoid collisions.
    max_existing_id = int(bulk_fuel_data['ASTFacilityID'].max(skipna=True))
    synthetic_id = max_existing_id + 1
    for idx in bulk_fuel_data.index:
        if pd.isna(bulk_fuel_data.at[idx, 'ASTFacilityID']):
            bulk_fuel_data.at[idx, 'ASTFacilityID'] = synthetic_id
            synthetic_id += 1
    bulk_fuel_data['ASTFacilityID'] = bulk_fuel_data['ASTFacilityID'].astype(int)

    # Convert site coordinates to Point geometries
    geometry = [
        Point(xy) for xy in zip(
            bulk_fuel_data['ASTFacilityLongitude'],
            bulk_fuel_data['ASTFacilityLatitude']
        )
    ]

    # Create a GeoDataFrame from sites
    bulk_fuel_sites_gdf = gpd.GeoDataFrame(bulk_fuel_data, geometry=geometry)
    bulk_fuel_sites_gdf.set_crs(epsg=4326, inplace=True)

    # Check and align CRS
    if bulk_fuel_sites_gdf.crs != regional_shapefile.crs:
        bulk_fuel_sites_gdf = bulk_fuel_sites_gdf.to_crs(regional_shapefile.crs)

    # Perform spatial join
    sites_with_regions = gpd.sjoin(
        bulk_fuel_sites_gdf, regional_shapefile,
        how='left', predicate='within'
    )

    # Load facilities and relationships into DuckDB
    for index, row in sites_with_regions.iterrows():
        facility_id = int(row['ASTFacilityID'])
        region_value = row.get(region_column, None)
        longitude = row['ASTFacilityLongitude']
        latitude = row['ASTFacilityLatitude']
        delivery_method = row.get('Delivery_method', None)
        community_name = row.get('CommunityName', None)

        # Handle NaN values
        if pd.isna(region_value):
            region_value = None
        if pd.isna(delivery_method):
            delivery_method = None
        if pd.isna(community_name):
            community_name = None

        # Insert facility into DuckDB (skip duplicates from spatial join)
        existing_facility = con.execute(
            "SELECT 1 FROM facilities WHERE facility_id = ?",
            [facility_id]
        ).fetchone()
        if not existing_facility:
            con.execute(
                "INSERT INTO facilities VALUES (?, ?, ?, ?, ?)",
                [facility_id, longitude, latitude, delivery_method, community_name]
            )

            # Insert delivery_method node and uses_method edge
            if delivery_method is not None:
                existing_method = con.execute(
                    "SELECT 1 FROM delivery_methods WHERE method_name = ?",
                    [delivery_method]
                ).fetchone()
                if not existing_method:
                    con.execute(
                        "INSERT INTO delivery_methods VALUES (?)",
                        [delivery_method]
                    )
                con.execute(
                    "INSERT INTO uses_method VALUES (?, ?)",
                    [facility_id, delivery_method]
                )

        # Insert located_in edge
        if region_value is not None:
            existing_edge = con.execute(
                "SELECT 1 FROM located_in WHERE facility_id = ? AND region_name = ?",
                [facility_id, region_value]
            ).fetchone()
            if not existing_edge:
                con.execute(
                    "INSERT INTO located_in VALUES (?, ?)",
                    [facility_id, region_value]
                )

            # Build in-memory dictionary
            bulk_fuel_dict_with_regions[facility_id] = {
                'FacilityID': facility_id,
                'Longitude': longitude,
                'Latitude': latitude,
                'Delivery_method': delivery_method,
                'Region': region_value
            }
        else:
            unassigned_dict[facility_id] = {
                'FacilityID': facility_id,
                'Longitude': longitude,
                'Latitude': latitude,
                'Delivery_method': delivery_method,
                'Region': 'Unassigned'
            }

    # Update facility counts per region
    con.execute("""
        UPDATE regions SET num_facilities = (
            SELECT COUNT(*) FROM located_in
            WHERE located_in.region_name = regions.region_name
        )
    """)

    # Save to CSV
    result_df = pd.DataFrame(sites_with_regions.drop(columns='geometry'))
    result_df.to_csv('sites_with_regions.csv', index=False)

    # Statistics
    total_sites = len(bulk_fuel_dict_with_regions)
    sites_with_region = sum(
        1 for v in bulk_fuel_dict_with_regions.values()
        if v['Region'] is not None
    )
    sites_without_region = total_sites - sites_with_region

    summary = f"""
    Regionalization Complete!

    Total Facilities: {total_sites}
    Facilities Assigned to Regions: {sites_with_region}
    Facilities Without Region Assignment: {sites_without_region}

    Results saved to 'sites_with_regions.csv'
    """
    print(summary)

    return bulk_fuel_dict_with_regions, unassigned_dict


# ---------------------------------------------------------------------------
# Adjacency Edge Construction
# ---------------------------------------------------------------------------

def build_adjacency_edges(shapefile_path, region_column, con):
    """Build adjacent_to edges between regions that share borders.

    Uses the shapefile geometries to detect which regions are neighbors
    (share a boundary or touch).

    Parameters:
        shapefile_path: Path to the shapefile
        region_column: Column name for region names
        con: DuckDB connection
    """
    regional_shapefile = gpd.read_file(shapefile_path)
    regions = regional_shapefile[[region_column, 'geometry']].copy()
    regions = regions.rename(columns={region_column: 'region_name'})

    adjacency_count = 0
    for i, region_a in regions.iterrows():
        for j, region_b in regions.iterrows():
            if i >= j:
                continue
            if region_a['geometry'].touches(region_b['geometry']) or \
               region_a['geometry'].intersects(region_b['geometry']):
                # Skip if they only share a point (not a real border)
                intersection = region_a['geometry'].intersection(region_b['geometry'])
                if intersection.is_empty:
                    continue

                shared_length = intersection.length

                name_a = region_a['region_name']
                name_b = region_b['region_name']

                # Insert both directions for undirected adjacency
                con.execute(
                    """INSERT INTO adjacent_to VALUES (?, ?, ?)
                       ON CONFLICT DO NOTHING""",
                    [name_a, name_b, shared_length]
                )
                con.execute(
                    """INSERT INTO adjacent_to VALUES (?, ?, ?)
                       ON CONFLICT DO NOTHING""",
                    [name_b, name_a, shared_length]
                )
                adjacency_count += 1

    print(f"Built {adjacency_count} adjacency edges between regions.")


# ---------------------------------------------------------------------------
# Graph Queries
# ---------------------------------------------------------------------------

def query_facilities_in_region(con, region_name):
    """Query all facilities in a given region using the graph.

    Parameters:
        con: DuckDB connection
        region_name: Name of the region to query

    Returns:
        pd.DataFrame: Facilities in the region
    """
    result = con.execute("""
        SELECT f.facility_id, f.longitude, f.latitude,
               f.delivery_method, f.community_name
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        WHERE li.region_name = ?
        ORDER BY f.facility_id
    """, [region_name]).fetchdf()
    return result


def query_adjacent_regions(con, region_name):
    """Query regions adjacent to a given region.

    Parameters:
        con: DuckDB connection
        region_name: Name of the region

    Returns:
        list: Names of adjacent regions
    """
    result = con.execute("""
        SELECT region_b AS adjacent_region
        FROM adjacent_to
        WHERE region_a = ?
        ORDER BY region_b
    """, [region_name]).fetchdf()
    return result['adjacent_region'].tolist()


def query_graph_summary(con):
    """Print a summary of the graph database contents.

    Parameters:
        con: DuckDB connection
    """
    regions_count = con.execute("SELECT COUNT(*) FROM regions").fetchone()[0]
    facilities_count = con.execute("SELECT COUNT(*) FROM facilities").fetchone()[0]
    edges_count = con.execute("SELECT COUNT(*) FROM located_in").fetchone()[0]
    adj_count = con.execute(
        "SELECT COUNT(*) FROM adjacent_to WHERE region_a < region_b"
    ).fetchone()[0]

    print(f"\n{'='*50}")
    print("Graph Database Summary")
    print(f"{'='*50}")
    print(f"  Region nodes:     {regions_count}")
    print(f"  Facility nodes:   {facilities_count}")
    print(f"  Located_in edges: {edges_count}")
    print(f"  Adjacency edges:  {adj_count}")
    print(f"{'='*50}\n")

    # Per-region breakdown
    region_stats = con.execute("""
        SELECT r.region_name, r.num_facilities,
               COUNT(DISTINCT a.region_b) AS num_neighbors
        FROM regions r
        LEFT JOIN adjacent_to a ON r.region_name = a.region_a
        GROUP BY r.region_name, r.num_facilities
        ORDER BY r.num_facilities DESC
    """).fetchdf()
    print("Per-region statistics:")
    print(region_stats.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Dictionary Structuring (same as original)
# ---------------------------------------------------------------------------

def get_region_statistics(csv_path='sites_with_regions.csv'):
    """Analyze regionalization results and provide statistics by region.

    Parameters:
        csv_path: Path to CSV file with region assignments

    Returns:
        str: Regional statistics
    """
    df = pd.read_csv(csv_path)
    region_counts = df.groupby('NAME').size().sort_values(ascending=False)

    stats = "Facilities per Region:\n"
    for region, count in region_counts.items():
        stats += f"{region}: {count} facilities\n"

    return stats


def structure_regional_dictionaries(bulk_fuel_dict_with_regions):
    """Create a regional nested dictionary for each region in Alaska.

    Parameters:
        bulk_fuel_dict_with_regions: Flat dictionary of facility data

    Returns:
        dict: Nested dictionary keyed by region
    """
    temp_regional_dict = {}
    for facility_id, data in bulk_fuel_dict_with_regions.items():
        region = data.get('Region')
        if region is None:
            continue
        if region not in temp_regional_dict:
            temp_regional_dict[region] = {}
        temp_regional_dict[region][facility_id] = {
            'FacilityID': facility_id,
            'Longitude': data['Longitude'],
            'Latitude': data['Latitude'],
            'Delivery_method': data['Delivery_method']
        }

    return temp_regional_dict


def plot_by_regions(reg_dict, unassigned_dict):
    """Plot facilities by region using separate colors.

    Parameters:
        reg_dict: Regional dictionary of facilities
        unassigned_dict: Dictionary of unassigned facilities
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9A6324', '#808000'
    ]

    for region, sites in reg_dict.items():
        x = [site['Longitude'] for site in sites.values()]
        y = [site['Latitude'] for site in sites.values()]
        ax.scatter(x, y, color=colors.pop(), label=region)

    if unassigned_dict:
        x = [site['Longitude'] for site in unassigned_dict.values()]
        y = [site['Latitude'] for site in unassigned_dict.values()]
        ax.scatter(x, y, color='black', label='Unassigned', s=30, marker='x')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Bulk Fuel Facilities by Region')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


def group_by_delivery_method(regional_dict):
    """Group sites by delivery method within each region.

    Parameters:
        regional_dict: Nested dictionary keyed by region

    Returns:
        dict: Nested dictionary keyed by region, then delivery method
    """
    global final_regionalized_dict
    temp_dict = {}
    for region, sites in regional_dict.items():
        for facility_id, data in sites.items():
            delivery_method = data.get('Delivery_method')
            if delivery_method is None:
                continue
            if region not in temp_dict:
                temp_dict[region] = {}
            if delivery_method not in temp_dict[region]:
                temp_dict[region][delivery_method] = {}
            if facility_id not in temp_dict[region][delivery_method]:
                temp_dict[region][delivery_method][facility_id] = {
                    'Longitude': data['Longitude'],
                    'Latitude': data['Latitude']
                }

    final_regionalized_dict = temp_dict
    return final_regionalized_dict


# ---------------------------------------------------------------------------
# CrewAI Tools
# ---------------------------------------------------------------------------

@tool("add_new_delivery_methods")
def add_new_delivery_methods():
    """Function to add delivery methods to the regionalized dictionary"""
    global final_regionalized_dict
    for region in final_regionalized_dict.items():
        for methods in region.items():
            for facilities in methods.items():
                if facility_id in facilities:
                    if delivery_method == methods:
                        print("Already has correct delivery method.")
                        continue
                    else:
                        facilities.remove(facility_id)
                        print(f"  Removed {facility_id} from {methods} in {region}")
                        if delivery_method not in final_regionalized_dict[region]:
                            final_regionalized_dict[region][delivery_method] = []
                        final_regionalized_dict[region][delivery_method].append(facility_id)
                        print(f"Added {facility_id} to {delivery_method} in {region}")
    return final_regionalized_dict


@tool("get_facility_dictionary")
def get_facility_dictionary() -> str:
    """Retrieves the complete facility dictionary with all site information"""
    return json.dumps(final_regionalized_dict, indent=2)


@tool("update_facility_dictionary")
def update_facility_dictionary(new_dict: str) -> str:
    """Updates the facility dictionary with new data"""
    global final_regionalized_dict
    final_regionalized_dict = json.loads(new_dict)
    return "Dictionary updated successfully"


@tool("save_json")
def save_json(json_report: str):
    """Function to save the logistics assessment report as a JSON file.

    Args:
        json_report (str): A JSON string containing the assessment report
    """
    with open('logistics_report.json', 'w') as f:
        json.dump(json_report, f, indent=4)


def save_as_csv():
    """Save the 'final_regionalized_dict' as a csv file."""
    global final_regionalized_dict
    rows = []
    for region, delivery_methods in final_regionalized_dict.items():
        for delivery_method, facilities in delivery_methods.items():
            for facility_id, data in facilities.items():
                rows.append({
                    'Region': region,
                    'Delivery_method': delivery_method,
                    'Facility_id': facility_id,
                    'Longitude': data['Longitude'],
                    'Latitude': data['Latitude']
                })
    final_df = pd.DataFrame(rows)
    final_df.to_csv('regionalized_df.csv', index=False)
    return final_df


# ---------------------------------------------------------------------------
# CrewAI Agents
# ---------------------------------------------------------------------------

# Set LLM and API Key
with open('.env', 'w', encoding='utf-8') as f:
    f.write(f"GEMINI_API_KEY={pipeline.get_api_key()}\n")
    f.write("MODEL=gemini/gemini-2.5-flash-preview-04-17\n")

load_dotenv()
os.environ["GEMINI_API_KEY"] = pipeline.get_api_key()
llm = LLM(model='gemini/gemini-2.5-flash')

# Agent - Delivery method analyst
delivery_method_agent = Agent(
    role="Delivery Method Coordinator",
    goal="Analyze each bulk fuel facility site's delivery method to ensure each has a set delivery method.",
    backstory="""Expert in logistics and market analysis, with a focus on regional fuel delivery in Alaska, specifcally delivey methods.""",
    verbose=True,
    llm=llm,
    tools=[add_new_delivery_methods, get_facility_dictionary, update_facility_dictionary]
)

delivery_task = Task(
    description=f"""
    Use the get_facility_dictionary tool to retrieve the current facility data.

    Complete the following tasks:
    1. Analyze the dictionary to ensure each site has a delivery method specified.
        a. If sites do not have a specified delivery method:
          - Examine delivery methods used by other facilities in the same region
          - Assign the most common delivery method from that region
          - If the region has mixed methods, assign based on geographic proximity patterns
          - With the facility ID and delivery method, add to the dictionary using add_new_delivery_methods tool.
        b. If sites have multiple delivery methods (e.g., 'Plane or Road'):
          - Keep these sites and groupings as is.
  2. Once complete, return the dictionary in the same format as the input dictionary using the update_facility_dictionary tool""",
    agent=delivery_method_agent,
    expected_output=
    '''In JSON format:
    - The complete modified dictionary with all delivery methods assigned
    - Summary of each newly added delivery method (if no delivery method was present).
    - Summary of sights that have more than 1 delivery method.''',
    verbose=True
)

# Agent - Logistics Coordinator
logistics_agent = Agent(
    role="Logistics Coordinator",
    goal="Provide a high-level assessment of whether each route grouping is realistic and practical for Alaska fuel delivery operations.",
    backstory='''Expert in Alaska's geography and logistics with practical experience in fuel delivery
    operations. Knows the general operational limits of road, plane, and barge delivery methods in Alaska's
    unique environment. Provides straightforward assessments of whether route groupings make practical sense
    based on distance, geography, and delivery method capabilities.''',
    verbose=True,
    llm=llm,
    tools=[save_json]
)

logistics_task = Task(
    description=f"""
    Review the updated dictionary provided in the previous task:{final_regionalized_dict}

    For each grouping (e.g., 'Road - Railbelt', 'Plane - North Slope'),
    provide a general assessment of whether the grouping is realistic and practical.

    Consider:
    - Is the geographic area too large for a single route with this delivery method?
    - Does the delivery method make sense for the distances and terrain involved?
    - Are there obvious geographic or logistical issues that would make this grouping impractical?

    Keep the assessment general and straightforward - focus on obvious concerns rather than
    detailed logistics planning. Flag groupings that seem problematic and suggest simple
    improvements where needed.

    Structure your response as a valid JSON object, then use the 'save_json' tool
    to save it. Pass your complete JSON object (as a string) to the save_json tool.""",
    agent=logistics_agent,
    context=[delivery_task],
    expected_output="""
    A concise assessment report containing:

    1. Overall Assessment:
       - Brief evaluation of each delivery_method-region grouping
       - Simple status for each: "Looks Good", "May Need Review", or "Likely Too Large"
       - High-level reasoning for any concerns

    2. Key Recommendations:
       - List groupings that appear unrealistic or too broad
       - General suggestions for improvement (e.g., "Consider splitting this region into 2-3 smaller areas")
       - Any obvious mismatches between delivery method and geography

    3. Summary:
       - How many groupings seem practical?
       - How many may need adjustments?
       - General confidence level in the current grouping structure

    Format: In a JSON format create a clear, readable summary focusing on practical applicability rather than detailed logistics.

    A JSON object with this exact structure:
    {{
        "overall_assessment": {{
            "Road - Railbelt": {{"status": "Looks Good", "reasoning": "..."}},
            "Plane - North Slope": {{"status": "May Need Review", "reasoning": "..."}}
        }},
        "key_recommendations": [
            "Recommendation 1",
            "Recommendation 2"
        ],
        "summary": {{
            "practical_groupings": 5,
            "needs_adjustments": 2,
            "confidence_level": "High"
        }}
    }}""",
    verbose=True
)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run_regionalization(bulk_fuel_csv_path, shapefile_path, region_column=None):
    """Execute the regionalization workflow with DuckDB graph database.

    Parameters:
        bulk_fuel_csv_path: Path to bulk fuel facility CSV
        shapefile_path: Path to Alaska regions shapefile
        region_column: Column name for region in shapefile

    Returns:
        dict: Dictionary with facility assignments
    """
    # Step 1: Initialize DuckDB graph database
    duckdb_con = init_duckdb_graph()

    # Step 2: Load all regions into the regions table FIRST
    # This prevents foreign key violations when inserting located_in edges
    load_regions_into_db(shapefile_path, region_column, duckdb_con)

    # Step 3: Spatial join + load into DuckDB
    group_sites_by_region(bulk_fuel_csv_path, shapefile_path, region_column, duckdb_con)

    # Step 4: Build adjacency edges
    build_adjacency_edges(shapefile_path, region_column, duckdb_con)

    # Step 5: Print graph summary
    query_graph_summary(duckdb_con)

    # Step 6: Structure dictionaries and plot
    regional_dict = structure_regional_dictionaries(bulk_fuel_dict_with_regions)
    plot_by_regions(regional_dict, unassigned_dict)
    final_regionalized_dict = group_by_delivery_method(regional_dict)

    # Step 7: Run CrewAI
    print("Running CrewAI approach...")
    print("=" * 50)

    crew = Crew(
        agents=[delivery_method_agent, logistics_agent],
        tasks=[delivery_task, logistics_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff(
        inputs={'final_regionalized_dict': final_regionalized_dict}
    )

    print("\n" + "=" * 50)
    print("CREW EXECUTION COMPLETE")
    print("=" * 50)
    print(result)

    return final_regionalized_dict


def main():
    """Main entry point for the regionalization graph module."""
    region_column = 'NAME'
    output_dict = run_regionalization(
        bulk_fuel_csv_path, shapefile_path, region_column
    )
    # Write output dictionary to a json formatted file
    with open("output_regionalization_dictionary.json", "w") as outfile:
        outfile.write(json.dumps(final_regionalized_dict, indent=4))
    save_as_csv()


if __name__ == "__main__":
    main()
