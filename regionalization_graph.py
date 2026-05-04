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
import math
from collections import defaultdict
import duckdb
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import crewai
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

import pipeline

# Setting Working Directory
pipeline.set_cwd('/media/volume/GraphDB_Runs')

# Retrieving Data
bulk_fuel_data = pd.read_csv(
    'Utilities_Bulk_Fuel_Inventory.csv',
    usecols=['ASTFacilityID', 'ASTFacilityLongitude', 'ASTFacilityLatitude']
)
regional_data = gpd.read_file(
    'Alaska_Energy_Authority_Library.shp'
)
bulk_fuel_csv_path = 'Utilities_Bulk_Fuel_Inventory.csv'
shapefile_path = 'Alaska_Energy_Authority_Library.shp'

# Globals
global final_regionalized_dict
final_regionalized_dict = {}


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
            delivery_method_1 VARCHAR,
            delivery_method_2 VARCHAR,
            community_name VARCHAR,
            x_3413 DOUBLE,
            y_3413 DOUBLE
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

    # connects_to: facility <-> facility (all feasible links within
    # the same region/delivery-method group)
    con.execute("""
        CREATE TABLE connects_to (
            src INTEGER REFERENCES facilities(facility_id),
            dst INTEGER REFERENCES facilities(facility_id),
            distance_miles DOUBLE,
            avg_friction DOUBLE,
            max_friction DOUBLE,
            path_length_miles DOUBLE,
            friction_summer DOUBLE,
            friction_shoulder DOUBLE,
            friction_winter DOUBLE,
            delivery_cost DOUBLE,
            cost_summer DOUBLE,
            cost_shoulder DOUBLE,
            cost_winter DOUBLE,
            cost_fwd DOUBLE,
            cost_rev DOUBLE,
            PRIMARY KEY (src, dst)
        )
    """)

    # part_of_route: facility -> facility (optimized TSP tour edges)
    con.execute("""
        CREATE TABLE part_of_route (
            src INTEGER REFERENCES facilities(facility_id),
            dst INTEGER REFERENCES facilities(facility_id),
            route_id INTEGER,
            sequence INTEGER,
            distance_miles DOUBLE,
            PRIMARY KEY (src, dst, route_id, sequence)
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
                LABEL adjacent_to,
            connects_to
                SOURCE KEY (src) REFERENCES facilities (facility_id)
                DESTINATION KEY (dst) REFERENCES facilities (facility_id)
                LABEL connects_to,
            part_of_route
                SOURCE KEY (src) REFERENCES facilities (facility_id)
                DESTINATION KEY (dst) REFERENCES facilities (facility_id)
                LABEL part_of_route
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
# Delivery Method Resolution
# ---------------------------------------------------------------------------

def _resolve_delivery_method(raw_method):
    """Parse compound delivery method, resolve to cheapest option.

    Returns (dm1, dm2) where dm1 is the resolved (cheaper) method and dm2 is
    the alternative (NULL if single method).
    """
    if raw_method is None:
        return None, None
    if ' or ' in raw_method:
        parts = [p.strip() for p in raw_method.split(' or ')]
        # Rank by baseline cost per mile: Barge($2) < Road($3.5) < Plane($11.5)
        COST_RANK = {'Barge': 2.0, 'Road': 3.5, 'Plane': 11.5}
        parts.sort(key=lambda p: COST_RANK.get(p, 999))
        return parts[0], parts[1]
    return raw_method, None


# ---------------------------------------------------------------------------
# Spatial Join + Graph Loading
# ---------------------------------------------------------------------------

def group_sites_by_region(bulk_fuel_csv_path, shapefile_path, region_column, con):
    """Perform spatial join and load facility-region relationships into DuckDB.

    Takes bulk fuel facility data and Alaska regional shapefile, performs
    spatial join, and populates the graph database.

    Parameters:
        bulk_fuel_csv_path: Path to CSV with bulk fuel facility data
        shapefile_path: Path to Alaska Energy Development Regions shapefile
        region_column: Name of the region column in the shapefile
        con: DuckDB connection with graph schema
    """

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
        raw_delivery_method = row.get('Delivery_method', None)
        community_name = row.get('CommunityName', None)

        # Handle NaN values
        if pd.isna(region_value):
            region_value = None
        if pd.isna(raw_delivery_method):
            raw_delivery_method = None
        if pd.isna(community_name):
            community_name = None

        # Resolve compound delivery methods (e.g. "Plane or Barge" → "Barge")
        dm1, dm2 = _resolve_delivery_method(raw_delivery_method)

        # Insert facility into DuckDB (skip duplicates from spatial join)
        existing_facility = con.execute(
            "SELECT 1 FROM facilities WHERE facility_id = ?",
            [facility_id]
        ).fetchone()
        if not existing_facility:
            con.execute(
                "INSERT INTO facilities VALUES (?, ?, ?, ?, ?, ?)",
                [facility_id, longitude, latitude, dm1, dm2, community_name]
            )

            # Insert resolved delivery_method node and uses_method edge
            if dm1 is not None:
                existing_method = con.execute(
                    "SELECT 1 FROM delivery_methods WHERE method_name = ?",
                    [dm1]
                ).fetchone()
                if not existing_method:
                    con.execute(
                        "INSERT INTO delivery_methods VALUES (?)",
                        [dm1]
                    )
                con.execute(
                    "INSERT INTO uses_method VALUES (?, ?)",
                    [facility_id, dm1]
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

        else:
            pass  # Facility without region — already in facilities table

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

    # Statistics from graph
    total_sites = con.execute("SELECT COUNT(*) FROM facilities").fetchone()[0]
    sites_with_region = con.execute("SELECT COUNT(*) FROM located_in").fetchone()[0]
    sites_without_region = total_sites - sites_with_region

    summary = f"""
    Regionalization Complete!

    Total Facilities: {total_sites}
    Facilities Assigned to Regions: {sites_with_region}
    Facilities Without Region Assignment: {sites_without_region}

    Results saved to 'sites_with_regions.csv'
    """
    print(summary)


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


def haversine_distance(lon1, lat1, lon2, lat2):
    """Distance in miles between two (lon, lat) points using Haversine.

    Source: https://community.esri.com/t5/coordinate-reference-systems-blog/
    distance-on-a-sphere-the-haversine-formula/ba-p/902128
    """
    R = 3959  # Earth's radius in miles

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def build_facility_connections(con):
    """Build connects_to edges: all pairwise facility links within each
    (region, delivery_method) group.

    For every pair of facilities that share the same region AND delivery
    method, compute the Haversine distance and insert an edge.  These
    represent the full set of candidate links that the TSP solver may
    choose from.

    Parameters:
        con: DuckDB connection with populated facilities, located_in,
             and uses_method tables.
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

    # Group by (region, method)
    groups = defaultdict(list)
    for fid, lon, lat, region, method in rows:
        groups[(region, method)].append((fid, lon, lat))

    total_edges = 0
    for (region, method), facilities in groups.items():
        n = len(facilities)
        for i in range(n):
            fid_a, lon_a, lat_a = facilities[i]
            for j in range(i + 1, n):
                fid_b, lon_b, lat_b = facilities[j]
                dist = haversine_distance(lon_a, lat_a, lon_b, lat_b)
                # Insert both directions for undirected connectivity
                con.execute(
                    "INSERT INTO connects_to VALUES (?, ?, ?) "
                    "ON CONFLICT DO NOTHING",
                    [fid_a, fid_b, dist]
                )
                con.execute(
                    "INSERT INTO connects_to VALUES (?, ?, ?) "
                    "ON CONFLICT DO NOTHING",
                    [fid_b, fid_a, dist]
                )
                total_edges += 1

    print(f"Built {total_edges} facility connection edges "
          f"across {len(groups)} (region, method) groups.")


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
               f.delivery_method_1, f.delivery_method_2, f.community_name
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


def plot_by_regions(con):
    """Plot facilities by region using separate colors, querying the graph DB.

    Parameters:
        con: DuckDB connection
    """
    rows = con.execute("""
        SELECT f.facility_id, f.longitude, f.latitude,
               COALESCE(li.region_name, 'Unassigned') AS region_name
        FROM facilities f
        LEFT JOIN located_in li ON f.facility_id = li.facility_id
        ORDER BY region_name
    """).fetchall()

    regions = defaultdict(list)
    for fid, lon, lat, region in rows:
        regions[region].append((lon, lat))

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9A6324', '#808000'
    ]

    for region, coords in regions.items():
        if region == 'Unassigned':
            continue
        x = [c[0] for c in coords]
        y = [c[1] for c in coords]
        color = colors.pop() if colors else '#333333'
        ax.scatter(x, y, color=color, label=region)

    if 'Unassigned' in regions:
        x = [c[0] for c in regions['Unassigned']]
        y = [c[1] for c in regions['Unassigned']]
        ax.scatter(x, y, color='black', label='Unassigned', s=30, marker='x')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Bulk Fuel Facilities by Region')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig("outputs/regions.png", dpi=150, bbox_inches="tight")
    plt.close()


def _build_dict_from_graph(con):
    """Build the nested regionalized dictionary from the graph database.

    Queries the graph for all facilities with their region and delivery method
    assignments.  Facilities without a delivery method are grouped under
    'Unassigned' so the CrewAI agents can see and fix them.

    Parameters:
        con: DuckDB connection

    Returns:
        dict: Nested dict keyed by region → delivery_method → facility_id →
              {Longitude, Latitude}
    """
    global final_regionalized_dict
    rows = con.execute("""
        SELECT f.facility_id, f.longitude, f.latitude,
               li.region_name,
               COALESCE(um.method_name, 'Unassigned') AS delivery_method
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        ORDER BY li.region_name, delivery_method
    """).fetchall()

    result = {}
    for fid, lon, lat, region, method in rows:
        result.setdefault(region, {}).setdefault(method, {})[str(fid)] = {
            'Longitude': lon, 'Latitude': lat
        }

    final_regionalized_dict = result
    return final_regionalized_dict


# ---------------------------------------------------------------------------
# CrewAI Tools
# ---------------------------------------------------------------------------

@tool("add_new_delivery_methods")
def add_new_delivery_methods(facility_id: int, delivery_method: str, region: str) -> str:
    """Assign or update a facility's delivery method in the graph database.

    Args:
        facility_id: The facility ID to update.
        delivery_method: The delivery method to assign (Road, Barge, or Plane).
        region: The region the facility belongs to (for confirmation logging).
    """
    con = duckdb.connect(database=DB_PATH)
    # Ensure delivery method node exists
    con.execute(
        "INSERT INTO delivery_methods VALUES (?) ON CONFLICT DO NOTHING",
        [delivery_method]
    )
    # Remove old assignment if any
    con.execute("DELETE FROM uses_method WHERE facility_id = ?", [facility_id])
    # Insert new assignment
    con.execute("INSERT INTO uses_method VALUES (?, ?)", [facility_id, delivery_method])
    # Also update the delivery_method_1 column on the facility node
    con.execute(
        "UPDATE facilities SET delivery_method_1 = ? WHERE facility_id = ?",
        [delivery_method, facility_id]
    )
    con.close()
    return f"Updated facility {facility_id} in {region} to delivery method: {delivery_method}"


@tool("get_facility_dictionary")
def get_facility_dictionary() -> str:
    """Retrieves facility data from the graph database, grouped by region and delivery method."""
    con = duckdb.connect(database=DB_PATH)
    rows = con.execute("""
        SELECT f.facility_id, f.longitude, f.latitude,
               li.region_name,
               COALESCE(um.method_name, 'Unassigned') AS delivery_method
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        ORDER BY li.region_name, delivery_method
    """).fetchall()
    con.close()
    result = {}
    for fid, lon, lat, region, method in rows:
        result.setdefault(region, {}).setdefault(method, {})[str(fid)] = {
            'Longitude': lon, 'Latitude': lat
        }
    return json.dumps(result, indent=2)


@tool("update_facility_dictionary")
def update_facility_dictionary(new_dict: str) -> str:
    """Updates the facility dictionary with new data"""
    global final_regionalized_dict
    final_regionalized_dict = json.loads(new_dict)
    return "Dictionary updated successfully"


def save_as_csv():
    """Save regionalized facility data as a CSV file, querying the graph DB."""
    con = duckdb.connect(database=DB_PATH)
    df = con.execute("""
        SELECT li.region_name AS Region,
               COALESCE(um.method_name, 'Unassigned') AS Delivery_method,
               f.facility_id AS Facility_id,
               f.longitude AS Longitude,
               f.latitude AS Latitude
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        ORDER BY Region, Delivery_method, Facility_id
    """).fetchdf()
    con.close()
    df.to_csv('regionalized_df.csv', index=False)
    return df


# ---------------------------------------------------------------------------
# CrewAI Agents
# ---------------------------------------------------------------------------

# Set LLMs (both regionalization agents use Haiku 4.5 via OpenRouter)
llm_haiku = pipeline.get_llm("haiku")

# Agent - Delivery method analyst
delivery_method_agent = Agent(
    role="Delivery Method Coordinator",
    goal="Analyze each bulk fuel facility site's delivery method to ensure each has a set delivery method.",
    backstory="""Expert in logistics and market analysis, with a focus on regional fuel delivery in Alaska, specifically delivery methods. Bases all decisions strictly on facility data and the graph database rather than assumptions.""",
    verbose=True,
    llm=llm_haiku,
    tools=[add_new_delivery_methods, get_facility_dictionary, update_facility_dictionary]
)

delivery_task = Task(
    description=f"""
    Use the get_facility_dictionary tool to retrieve the current facility data from the graph database.
    The dictionary is nested: region → delivery_method → facility_id → {{Longitude, Latitude}}.
    Facilities without a delivery method appear under the 'Unassigned' key within their region.

    IMPORTANT: Base all decisions on the actual facility data retrieved from the dictionary.
    Do not infer or assume delivery methods beyond what is present in the data.
    Only assign delivery methods that already exist in the dataset (Road, Barge, Plane).

    NOTE: Compound delivery methods (e.g., 'Plane or Barge', 'Plane or Road') have already been
    resolved to a single method during data loading. The facilities table retains the alternative
    method in delivery_method_2 for reference. You do NOT need to resolve compound methods.

    Complete the following tasks:
    1. Analyze the dictionary to ensure each site has a delivery method specified.
        a. If sites are listed under 'Unassigned' (no delivery method):
          - Examine delivery methods used by other facilities in the same region
          - Assign the most common delivery method from that region
          - If the region has mixed methods, assign based on geographic proximity patterns
          - Call add_new_delivery_methods(facility_id=<id>, delivery_method=<method>, region=<region>)
            for each facility that needs assignment.
    2. Once complete, use get_facility_dictionary again to retrieve the updated data and return it
       using the update_facility_dictionary tool.""",
    agent=delivery_method_agent,
    expected_output=
    '''In JSON format:
    - The complete modified dictionary with all delivery methods assigned
    - Summary of each newly added delivery method (if no delivery method was present).''',
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

    # Step 6: Plot facilities by region and build initial dict from graph
    plot_by_regions(duckdb_con)
    _build_dict_from_graph(duckdb_con)

    # Step 7: Run CrewAI (assigns missing delivery methods)
    # Agents now read/write directly to the graph DB via tools.
    print("Running CrewAI approach...")
    print("=" * 50)

    crew = Crew(
        agents=[delivery_method_agent],
        tasks=[delivery_task],
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

    # Step 8: Rebuild dict from graph after agent updates
    _build_dict_from_graph(duckdb_con)

    # Step 9: Build facility connection edges AFTER agents have finalized delivery methods
    build_facility_connections(duckdb_con)

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
