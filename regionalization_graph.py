# -*- coding: utf-8 -*-
"""regionalization_graph.py

DuckDB Graph Database version of the regionalization module.

This module replaces the dictionary-centric regionalization.py with a
DuckDB-backed property graph. Facilities, regions, and delivery methods
are modeled as distinct node types with typed edges.

Graph Schema:
    Node tables: facilities, regions, delivery_methods
    Edge tables: located_in, uses_method, adjacent_to, connects_to, part_of_route

Input data:
    - Bulk Fuel Facility Sites (CSV)
    - Alaska Energy Development Regions (shapefile)

Agents:
    - Delivery Method Coordinator: Uses graph queries to assign delivery methods
      to unassigned facilities based on same-region neighbor patterns

Outputs:
    - regionalization.duckdb: Persistent graph database for cross-module use
    - output_regionalization_dictionary.json: Backward-compatible nested dict
    - regionalized_df.csv: Flat CSV of all facility assignments
    - logistics_report.json: Placeholder for downstream compatibility
    - Two matplotlib visualizations (facility map + graph visualization)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import json
import duckdb
import crewai
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from dotenv import load_dotenv
import pipeline

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
pipeline.set_cwd('/media/volume/Preliminary_mas_runs')

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
bulk_fuel_csv_path = 'Utilities_Bulk_Fuel_Inventory.csv'
shapefile_path = 'Alaska_Energy_Authority_Library/Alaska_Energy_Authority_Library.shp'

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
duckdb_con = None
pgq_available = False

# ===========================================================================
# DuckDB Initialization and Schema
# ===========================================================================

def init_duckdb(db_path='regionalization.duckdb'):
    """Initialize a file-backed DuckDB connection and attempt to load DuckPGQ.

    Args:
        db_path: Path to the DuckDB database file. Persists across modules.

    Returns:
        Tuple of (connection, pgq_available_flag)
    """
    con = duckdb.connect(db_path)
    pgq_available = False
    try:
        con.execute("INSTALL duckpgq FROM community;")
        con.execute("LOAD duckpgq;")
        pgq_available = True
        print("DuckPGQ extension loaded successfully.")
    except Exception as e:
        print(f"DuckPGQ extension not available: {e}")
        print("Falling back to standard SQL for graph queries.")
    return con, pgq_available


def create_graph_schema(con):
    """Create all node and edge table schemas in DuckDB.

    Drops existing tables first to ensure a clean slate.
    Creates three node tables (facilities, regions, delivery_methods)
    and five edge tables (located_in, uses_method, adjacent_to,
    connects_to, part_of_route).
    """
    # Drop in reverse dependency order (edges before nodes)
    for table in ['part_of_route', 'connects_to', 'adjacent_to',
                  'uses_method', 'located_in',
                  'facilities', 'regions', 'delivery_methods']:
        con.execute(f"DROP TABLE IF EXISTS {table}")

    # -- Node tables --
    con.execute("""
        CREATE TABLE facilities (
            facility_id INTEGER PRIMARY KEY,
            longitude DOUBLE,
            latitude DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE regions (
            region_name VARCHAR PRIMARY KEY
        )
    """)
    con.execute("""
        CREATE TABLE delivery_methods (
            method_name VARCHAR PRIMARY KEY
        )
    """)

    # -- Edge tables --
    con.execute("""
        CREATE TABLE located_in (
            facility_id INTEGER REFERENCES facilities(facility_id),
            region_name VARCHAR REFERENCES regions(region_name)
        )
    """)
    con.execute("""
        CREATE TABLE uses_method (
            facility_id INTEGER REFERENCES facilities(facility_id),
            method_name VARCHAR REFERENCES delivery_methods(method_name)
        )
    """)
    con.execute("""
        CREATE TABLE adjacent_to (
            src INTEGER REFERENCES facilities(facility_id),
            dst INTEGER REFERENCES facilities(facility_id),
            distance_miles DOUBLE
        )
    """)
    # Future edge tables (empty schemas for TSP module)
    con.execute("""
        CREATE TABLE connects_to (
            src INTEGER REFERENCES facilities(facility_id),
            dst INTEGER REFERENCES facilities(facility_id),
            distance_miles DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE part_of_route (
            src INTEGER REFERENCES facilities(facility_id),
            dst INTEGER REFERENCES facilities(facility_id),
            route_id INTEGER,
            sequence INTEGER,
            distance_miles DOUBLE
        )
    """)
    print("Graph schema created successfully.")


# ===========================================================================
# Data Loading (replaces dict-based group_sites_by_region)
# ===========================================================================

def group_sites_by_region(bulk_fuel_csv_path, shapefile_path, region_column, con):
    """Perform spatial join and load results directly into DuckDB graph tables.

    This replaces the original dict-based approach. Instead of building nested
    Python dictionaries, facility data flows directly into the graph database.

    Args:
        bulk_fuel_csv_path: Path to the CSV file with bulk fuel facility data
        shapefile_path: Path to the Alaska Energy Development Regions shapefile
        region_column: Name of the region column in the shapefile
        con: DuckDB connection
    """
    # Read and clean input data
    bulk_fuel_data = pd.read_csv(
        bulk_fuel_csv_path,
        usecols=['ASTFacilityID', 'ASTFacilityLongitude',
                 'ASTFacilityLatitude', 'Delivery_method']
    )
    regional_shapefile = gpd.read_file(shapefile_path)
    bulk_fuel_data = bulk_fuel_data.dropna(
        axis=0, how='all',
        subset=['ASTFacilityLongitude', 'ASTFacilityLatitude']
    )

    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(
        bulk_fuel_data['ASTFacilityLongitude'],
        bulk_fuel_data['ASTFacilityLatitude']
    )]
    bulk_fuel_sites_gdf = gpd.GeoDataFrame(bulk_fuel_data, geometry=geometry)
    bulk_fuel_sites_gdf.set_crs(epsg=4326, inplace=True)

    # Align CRS
    if bulk_fuel_sites_gdf.crs != regional_shapefile.crs:
        bulk_fuel_sites_gdf = bulk_fuel_sites_gdf.to_crs(regional_shapefile.crs)

    # Spatial join
    sites_with_regions = gpd.sjoin(
        bulk_fuel_sites_gdf, regional_shapefile,
        how='left', predicate='within'
    )

    # Save CSV for reference
    result_df = pd.DataFrame(sites_with_regions.drop(columns='geometry'))
    result_df.to_csv('sites_with_regions.csv', index=False)

    # Collect unique regions and delivery methods
    all_regions = set()
    all_methods = set()

    assigned_count = 0
    unassigned_count = 0

    for _, row in sites_with_regions.iterrows():
        facility_id = int(row['ASTFacilityID'])
        longitude = float(row['ASTFacilityLongitude'])
        latitude = float(row['ASTFacilityLatitude'])
        region_value = row.get(region_column, None)
        delivery_method = row.get('Delivery_method', None)

        # Handle NaN values
        if pd.isna(region_value):
            region_value = 'Unassigned'
            unassigned_count += 1
        else:
            assigned_count += 1

        if pd.isna(delivery_method) or str(delivery_method).strip() == '':
            delivery_method = None

        # Insert facility node (use OR IGNORE to handle duplicates from sjoin)
        con.execute(
            "INSERT OR IGNORE INTO facilities VALUES (?, ?, ?)",
            [facility_id, longitude, latitude]
        )

        # Insert region node
        all_regions.add(region_value)

        # Insert located_in edge
        # Check for duplicate first (sjoin can produce multiple matches)
        existing = con.execute(
            "SELECT 1 FROM located_in WHERE facility_id = ? AND region_name = ?",
            [facility_id, region_value]
        ).fetchone()
        if not existing:
            con.execute(
                "INSERT INTO located_in VALUES (?, ?)",
                [facility_id, region_value]
            )

        # Insert delivery method node and uses_method edge
        if delivery_method is not None:
            all_methods.add(delivery_method)
            existing = con.execute(
                "SELECT 1 FROM uses_method WHERE facility_id = ?",
                [facility_id]
            ).fetchone()
            if not existing:
                con.execute(
                    "INSERT INTO uses_method VALUES (?, ?)",
                    [facility_id, delivery_method]
                )

    # Batch insert region nodes
    for region in all_regions:
        con.execute(
            "INSERT OR IGNORE INTO regions VALUES (?)",
            [region]
        )

    # Batch insert delivery method nodes
    for method in all_methods:
        con.execute(
            "INSERT OR IGNORE INTO delivery_methods VALUES (?)",
            [method]
        )

    # Print summary
    total = assigned_count + unassigned_count
    print(f"""
    Regionalization Complete!

    Total Facilities: {total}
    Facilities Assigned to Regions: {assigned_count}
    Facilities Without Region Assignment: {unassigned_count}

    Results saved to 'sites_with_regions.csv'
    Data loaded into DuckDB graph database.
    """)


# ===========================================================================
# Adjacency Edge Construction
# ===========================================================================

def build_adjacency_edges(con):
    """Build adjacent_to edges between same-region facilities with haversine distance.

    Self-joins facilities via located_in to find all pairs in the same region.
    Computes haversine distance in SQL. Excludes Unassigned region and self-loops.

    Args:
        con: DuckDB connection
    """
    con.execute("DELETE FROM adjacent_to")
    con.execute("""
        INSERT INTO adjacent_to
        SELECT
            f1.facility_id AS src,
            f2.facility_id AS dst,
            2 * 3959 * ASIN(SQRT(
                POWER(SIN(RADIANS(f2.latitude - f1.latitude) / 2), 2) +
                COS(RADIANS(f1.latitude)) * COS(RADIANS(f2.latitude)) *
                POWER(SIN(RADIANS(f2.longitude - f1.longitude) / 2), 2)
            )) AS distance_miles
        FROM facilities f1
        JOIN located_in l1 ON f1.facility_id = l1.facility_id
        JOIN located_in l2 ON l1.region_name = l2.region_name
        JOIN facilities f2 ON f2.facility_id = l2.facility_id
        WHERE f1.facility_id < f2.facility_id
          AND l1.region_name != 'Unassigned'
    """)
    edge_count = con.execute("SELECT COUNT(*) FROM adjacent_to").fetchone()[0]
    print(f"Built {edge_count} adjacency edges.")


# ===========================================================================
# DuckPGQ Property Graph
# ===========================================================================

def create_property_graph(con):
    """Create the DuckPGQ property graph from existing tables.

    This provides SQL/PGQ query syntax for graph traversal. If DuckPGQ is not
    available, all queries fall back to standard SQL joins.

    Args:
        con: DuckDB connection
    """
    con.execute("DROP PROPERTY GRAPH IF EXISTS fuel_graph")
    con.execute("""
        CREATE PROPERTY GRAPH fuel_graph
        VERTEX TABLES (
            facilities LABEL Facility,
            regions LABEL Region,
            delivery_methods LABEL DeliveryMethod
        )
        EDGE TABLES (
            located_in
                SOURCE KEY (facility_id) REFERENCES facilities (facility_id)
                DESTINATION KEY (region_name) REFERENCES regions (region_name)
                LABEL LocatedIn,
            uses_method
                SOURCE KEY (facility_id) REFERENCES facilities (facility_id)
                DESTINATION KEY (method_name) REFERENCES delivery_methods (method_name)
                LABEL UsesMethod,
            adjacent_to
                SOURCE KEY (src) REFERENCES facilities (facility_id)
                DESTINATION KEY (dst) REFERENCES facilities (facility_id)
                LABEL AdjacentTo,
            connects_to
                SOURCE KEY (src) REFERENCES facilities (facility_id)
                DESTINATION KEY (dst) REFERENCES facilities (facility_id)
                LABEL ConnectsTo,
            part_of_route
                SOURCE KEY (src) REFERENCES facilities (facility_id)
                DESTINATION KEY (dst) REFERENCES facilities (facility_id)
                LABEL PartOfRoute
        );
    """)
    print("Property graph 'fuel_graph' created successfully.")


# ===========================================================================
# Visualization
# ===========================================================================

def plot_by_regions(con):
    """Plot facilities colored by region using DuckDB data.

    Produces a scatter plot matching the original regionalization.py visual output.

    Args:
        con: DuckDB connection
    """
    df = con.execute("""
        SELECT f.facility_id, f.longitude, f.latitude, li.region_name
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
    """).fetchdf()

    fig, ax = plt.subplots(figsize=(10, 10))
    palette = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
               '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
               '#469990', '#dcbeff', '#9A6324', '#808000']

    regions = df['region_name'].unique()
    color_idx = 0
    for region in sorted(regions):
        subset = df[df['region_name'] == region]
        if region == 'Unassigned':
            ax.scatter(subset['longitude'], subset['latitude'],
                       color='black', label='Unassigned', s=30, marker='x')
        else:
            color = palette[color_idx % len(palette)]
            ax.scatter(subset['longitude'], subset['latitude'],
                       color=color, label=region)
            color_idx += 1

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Bulk Fuel Facilities by Region')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def visualize_graph(con):
    """Visualize the facility graph using NetworkX and matplotlib.

    Renders facility nodes at their geographic positions, colored by region.
    Shows adjacent_to edges (filtered to <= 100 miles for visual clarity).
    Prints graph statistics.

    Args:
        con: DuckDB connection
    """
    # Fetch facility nodes with region info
    nodes_df = con.execute("""
        SELECT f.facility_id, f.longitude, f.latitude,
               li.region_name,
               um.method_name AS delivery_method
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        WHERE li.region_name != 'Unassigned'
    """).fetchdf()

    # Fetch edges (filtered for visualization clarity)
    edges_df = con.execute("""
        SELECT src, dst, distance_miles
        FROM adjacent_to
        WHERE distance_miles <= 100
    """).fetchdf()

    G = nx.Graph()

    # Color setup
    palette = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
               '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
               '#469990', '#dcbeff', '#9A6324', '#808000']
    region_colors = {}
    color_map = {}

    # Add facility nodes
    for _, row in nodes_df.iterrows():
        fid = int(row['facility_id'])
        region = row['region_name']
        if region not in region_colors:
            region_colors[region] = palette[len(region_colors) % len(palette)]
        G.add_node(fid,
                   pos=(row['longitude'], row['latitude']),
                   region=region,
                   delivery_method=row['delivery_method'])
        color_map[fid] = region_colors[region]

    # Add edges
    for _, row in edges_df.iterrows():
        src, dst = int(row['src']), int(row['dst'])
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst, weight=row['distance_miles'])

    # Draw
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.get_node_attributes(G, 'pos')
    colors = [color_map.get(n, 'gray') for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=30, ax=ax)

    # Legend
    for region, color in sorted(region_colors.items()):
        ax.scatter([], [], c=color, label=region, s=50)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Facility Adjacency Graph (edges = same-region, dist <= 100mi)')
    plt.tight_layout()
    plt.show()

    # Statistics
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if G.number_of_nodes() > 0:
        print(f"Connected components: {nx.number_connected_components(G)}")


# ===========================================================================
# Graph Statistics
# ===========================================================================

def get_region_statistics(con):
    """Print region statistics from the graph database.

    Args:
        con: DuckDB connection
    """
    stats_df = con.execute("""
        SELECT li.region_name, COUNT(*) AS facility_count
        FROM located_in li
        GROUP BY li.region_name
        ORDER BY facility_count DESC
    """).fetchdf()
    print("Facilities per Region:")
    for _, row in stats_df.iterrows():
        print(f"  {row['region_name']}: {row['facility_count']} facilities")
    return stats_df


# ===========================================================================
# Agent Tools (Graph-Aware)
# ===========================================================================

@tool("get_facility_dictionary")
def get_facility_dictionary() -> str:
    """Retrieves the complete facility data from the graph database as JSON.
    Returns all facilities with their region and delivery method."""
    global duckdb_con
    result = duckdb_con.execute("""
        SELECT f.facility_id, f.longitude, f.latitude,
               li.region_name AS region,
               um.method_name AS delivery_method
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        ORDER BY li.region_name, f.facility_id
    """).fetchdf()
    return result.to_json(orient='records', indent=2)


@tool("get_adjacent_facilities")
def get_adjacent_facilities(region: str) -> str:
    """Get all facilities in a given region with their delivery methods and
    distances to neighbors. This helps identify what delivery methods are
    common in a region.

    Args:
        region: The region name to query
    """
    global duckdb_con
    result = duckdb_con.execute("""
        SELECT
            f1.facility_id AS source_id,
            um1.method_name AS source_method,
            f2.facility_id AS neighbor_id,
            um2.method_name AS neighbor_method,
            a.distance_miles
        FROM facilities f1
        JOIN located_in l1 ON f1.facility_id = l1.facility_id
        JOIN adjacent_to a ON f1.facility_id = a.src
        JOIN facilities f2 ON f2.facility_id = a.dst
        LEFT JOIN uses_method um1 ON f1.facility_id = um1.facility_id
        LEFT JOIN uses_method um2 ON f2.facility_id = um2.facility_id
        WHERE l1.region_name = ?
        ORDER BY a.distance_miles
    """, [region]).fetchdf()
    return result.to_json(orient='records', indent=2)


@tool("get_unassigned_delivery_methods")
def get_unassigned_delivery_methods() -> str:
    """Find all facilities without a delivery method and show the delivery methods
    of their adjacent facilities in the same region, ranked by frequency.
    This is the primary tool for determining what method to assign."""
    global duckdb_con
    result = duckdb_con.execute("""
        SELECT
            f1.facility_id,
            l1.region_name AS region,
            um2.method_name AS neighbor_method,
            COUNT(*) AS neighbor_count,
            ROUND(AVG(a.distance_miles), 1) AS avg_distance_miles
        FROM facilities f1
        JOIN located_in l1 ON f1.facility_id = l1.facility_id
        LEFT JOIN uses_method um1 ON f1.facility_id = um1.facility_id
        JOIN adjacent_to a ON (f1.facility_id = a.src OR f1.facility_id = a.dst)
        JOIN facilities f2 ON (
            (f2.facility_id = a.dst AND f1.facility_id = a.src) OR
            (f2.facility_id = a.src AND f1.facility_id = a.dst)
        )
        JOIN uses_method um2 ON f2.facility_id = um2.facility_id
        WHERE um1.method_name IS NULL
        GROUP BY f1.facility_id, l1.region_name, um2.method_name
        ORDER BY f1.facility_id, neighbor_count DESC
    """).fetchdf()
    return result.to_json(orient='records', indent=2)


@tool("update_facility_delivery_method")
def update_facility_delivery_method(facility_id: int, delivery_method: str) -> str:
    """Update a facility's delivery method in the graph database.
    Creates the delivery_method node if it doesn't exist and adds the uses_method edge.

    Args:
        facility_id: The facility ID to update
        delivery_method: The delivery method to assign
    """
    global duckdb_con
    # Ensure the delivery method node exists
    duckdb_con.execute(
        "INSERT OR IGNORE INTO delivery_methods VALUES (?)",
        [delivery_method]
    )
    # Remove existing assignment if any
    duckdb_con.execute(
        "DELETE FROM uses_method WHERE facility_id = ?",
        [facility_id]
    )
    # Insert new assignment
    duckdb_con.execute(
        "INSERT INTO uses_method VALUES (?, ?)",
        [facility_id, delivery_method]
    )
    return f"Updated facility {facility_id} to delivery method: {delivery_method}"


# ===========================================================================
# Output Functions (backward compatibility)
# ===========================================================================

def build_output_dict(con):
    """Build the nested output dict from DuckDB for backward compatibility.

    Returns:
        Dict with structure: {Region: {DeliveryMethod: {FacilityID: {Longitude, Latitude}}}}
    """
    rows = con.execute("""
        SELECT li.region_name, COALESCE(um.method_name, 'Unknown') AS method,
               f.facility_id, f.longitude, f.latitude
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        ORDER BY li.region_name, method, f.facility_id
    """).fetchall()

    result = {}
    for region, method, fid, lon, lat in rows:
        result.setdefault(region, {}).setdefault(method, {})[str(fid)] = {
            "Longitude": lon,
            "Latitude": lat
        }
    return result


def save_as_csv(con):
    """Save facility data as CSV from DuckDB, matching the original output format.

    Args:
        con: DuckDB connection

    Returns:
        DataFrame with the exported data
    """
    df = con.execute("""
        SELECT li.region_name AS Region,
               COALESCE(um.method_name, 'Unknown') AS Delivery_method,
               f.facility_id AS Facility_id,
               f.longitude AS Longitude,
               f.latitude AS Latitude
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        ORDER BY Region, Delivery_method, Facility_id
    """).fetchdf()
    df.to_csv('regionalized_df.csv', index=False)
    return df


# ===========================================================================
# CrewAI Agent Setup
# ===========================================================================

# Set LLM and API Key
with open('.env', 'w', encoding='utf-8') as f:
    f.write(f"GEMINI_API_KEY={pipeline.get_api_key()}\n")
    f.write("MODEL=gemini/gemini-2.5-flash-preview-04-17\n")

load_dotenv()
os.environ["GEMINI_API_KEY"] = pipeline.get_api_key()
llm = LLM(model='gemini/gemini-2.5-flash')

# Delivery Method Coordinator Agent
delivery_method_agent = Agent(
    role="Delivery Method Coordinator",
    goal="Analyze each bulk fuel facility site's delivery method using the graph "
         "database to ensure each has a set delivery method.",
    backstory="""Expert in logistics and market analysis, with a focus on regional
    fuel delivery in Alaska. You have access to a DuckDB graph database where
    facilities, regions, and delivery methods are distinct nodes connected by typed
    edges (located_in, uses_method, adjacent_to). Use graph queries to find what
    delivery methods are common among a facility's neighbors in the same region.""",
    verbose=True,
    llm=llm,
    tools=[get_facility_dictionary, get_adjacent_facilities,
           get_unassigned_delivery_methods, update_facility_delivery_method]
)

delivery_task = Task(
    description="""
    Use the get_unassigned_delivery_methods tool to find all facilities that do
    not have a delivery method assigned. This tool returns each unassigned facility
    along with the delivery methods used by its graph neighbors, ranked by frequency.

    Complete the following tasks:
    1. For each facility without a delivery method:
       - Review the neighbor delivery methods and their frequencies
       - Assign the most common delivery method from adjacent facilities
       - If there's a tie, prefer the method used by the closest neighbors
         (lowest avg_distance_miles)
       - Use update_facility_delivery_method to set each assignment
    2. For facilities with multiple delivery methods (e.g., 'Plane or Road'):
       - Keep these as-is; do not modify them
    3. After all assignments, use get_facility_dictionary to verify the final state

    IMPORTANT: Process ALL unassigned facilities. Do not stop early.
    """,
    agent=delivery_method_agent,
    expected_output="""In JSON format:
    - Summary of each newly assigned delivery method (facility_id -> method)
    - Summary of facilities that already had multiple delivery methods
    - Total facilities processed""",
    verbose=True
)


# ===========================================================================
# Main Orchestration
# ===========================================================================

def run_regionalization(bulk_fuel_csv_path, shapefile_path, region_column=None):
    """Execute the full regionalization workflow with DuckDB graph database.

    Args:
        bulk_fuel_csv_path: Path to bulk fuel facility CSV
        shapefile_path: Path to Alaska regions shapefile
        region_column: Column name for region in shapefile

    Returns:
        Dict with facility assignments (backward-compatible format)
    """
    global duckdb_con, pgq_available

    # Step 1: Initialize DuckDB (file-backed)
    print("Initializing DuckDB graph database...")
    duckdb_con, pgq_available = init_duckdb('regionalization.duckdb')

    # Step 2: Create schema
    create_graph_schema(duckdb_con)

    # Step 3: Spatial join + load into DuckDB
    group_sites_by_region(bulk_fuel_csv_path, shapefile_path, region_column, duckdb_con)

    # Step 4: Build adjacency edges
    print("Building adjacency edges...")
    build_adjacency_edges(duckdb_con)

    # Step 5: Create property graph (if PGQ available)
    if pgq_available:
        create_property_graph(duckdb_con)

    # Step 6: Print statistics
    get_region_statistics(duckdb_con)

    # Step 7: Plot facilities by region
    plot_by_regions(duckdb_con)

    # Step 8: Visualize graph
    visualize_graph(duckdb_con)

    # Step 9: Run Delivery Method Agent
    print("Running CrewAI Delivery Method Agent...")
    print("=" * 50)
    crew = Crew(
        agents=[delivery_method_agent],
        tasks=[delivery_task],
        process=Process.sequential,
        verbose=True
    )
    result = crew.kickoff(
        inputs={'duckdb_con': duckdb_con}
    )
    print("\n" + "=" * 50)
    print("CREW EXECUTION COMPLETE")
    print("=" * 50)
    print(result)

    # Step 10: Build backward-compatible output
    output_dict = build_output_dict(duckdb_con)
    return output_dict


def main():
    """Main entry point for the regionalization graph module."""
    region_column = 'NAME'
    output_dict = run_regionalization(
        bulk_fuel_csv_path, shapefile_path, region_column
    )

    # Write output dictionary (backward-compatible JSON)
    with open("output_regionalization_dictionary.json", "w") as outfile:
        outfile.write(json.dumps(output_dict, indent=4))

    # Save CSV
    save_as_csv(duckdb_con)

    # Write placeholder logistics_report.json for downstream compatibility
    with open('logistics_report.json', 'w') as f:
        json.dump({
            "note": "Logistics coordinator removed in graph-based version. "
                    "Route assessment will be handled by downstream agents "
                    "using the graph database directly."
        }, f, indent=4)

    # Print graph summary
    print("\nFinal graph database summary:")
    for table in ['facilities', 'regions', 'delivery_methods',
                  'located_in', 'uses_method', 'adjacent_to',
                  'connects_to', 'part_of_route']:
        count = duckdb_con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    # Close connection (file persists for downstream modules)
    duckdb_con.close()
    print("\nDuckDB connection closed. Database saved to 'regionalization.duckdb'.")


if __name__ == "__main__":
    main()
