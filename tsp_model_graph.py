# -*- coding: utf-8 -*-
"""tsp_model_graph.py

DuckDB Graph Database version of the TSP model module.

This module replaces tsp_model.py. Instead of reading nested JSON
dictionaries, it queries the shared DuckDB graph database for facility
data and writes computed routes back to the graph as edges.

Agents:
    - TSP Route Optimizer: Queries graph DB for computed route data
    - Cost Estimator: Assigns cost values to route segments
    - Operational Risk Agent: Assesses segment risks
    - Route Analyzer: Assesses route quality using cost + risk inputs
    - TSP Route Adjuster: Fixes flagged routes in the graph database
    - Writing Agent: Synthesizes findings into a report
    - Contrarian Agent: Provides critical review

Input data:
    - regionalization.duckdb (graph database)
    - market_cost_analysis_report.json (from market & cost analysis module)

Outputs:
    - tsp_final_report.json: Comprehensive route analysis report
    - Routes written to regionalization.duckdb (connects_to, part_of_route)
    - Matplotlib visualizations: regional tours + final graph state
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import json
import math
import time
import random
import functools
import itertools
import duckdb
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from collections import defaultdict, namedtuple
from typing import Set, List, Tuple, Iterable, Callable, Dict

from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from dotenv import load_dotenv
import pipeline

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
pipeline.set_cwd('/media/volume/Preliminary_mas_runs')

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
graph_con = None

# ===========================================================================
# Type Aliases (unchanged from original)
# ===========================================================================
City = complex
Cities = frozenset
Tour = list
TSP = callable
Link = Tuple[City, City]
Segment = list

# ===========================================================================
# Core TSP Algorithm Functions (all unchanged from original tsp_model.py)
# ===========================================================================

def distance(A: City, B: City) -> float:
    """Distance between two cities in miles using the Haversine formula.

    Source: https://community.esri.com/t5/coordinate-reference-systems-blog/
    distance-on-a-sphere-the-haversine-formula/ba-p/902128
    """
    lon1, lat1 = A.real, A.imag
    lon2, lat2 = B.real, B.imag

    R = 3959  # Earth's radius in miles

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def shortest(tours: Iterable[Tour]) -> Tour:
    "The tour with the smallest tour length."
    return min(tours, key=tour_length)


def tour_length(tour: Tour) -> float:
    "The total distance of each link in the tour, including last to first."
    return sum(distance(tour[i], tour[i - 1]) for i in range(len(tour)))


def valid_tour(tour: Tour, cities: Cities) -> bool:
    "Does `tour` visit every city in `cities` exactly once?"
    from collections import Counter
    return Counter(tour) == Counter(cities)


def nearest_neighbor(A: City, cities) -> City:
    "Find the city C in cities that is nearest to city A."
    return min(cities, key=lambda C: distance(C, A))


def nearest_tsp(cities, start=None) -> Tour:
    """Greedy nearest-neighbor TSP. Extend partial tour to nearest unvisited."""
    start = start or next(iter(cities))
    tour = [start]
    unvisited = set(cities) - {start}
    def extend_to(C): tour.append(C); unvisited.remove(C)
    while unvisited:
        extend_to(nearest_neighbor(tour[-1], unvisited))
    return tour


def rep_nearest_tsp(cities, k=10):
    "Repeat nearest_tsp starting from k different cities; pick shortest."
    return shortest(
        nearest_tsp(cities, start)
        for start in random.sample(list(cities), min(k, len(cities)))
    )


def sample(population, n, seed=42) -> Iterable:
    "Return a list of n elements sampled from population."
    random.seed((n, seed))
    return random.sample(population, min(n, len(population)))


def opt2(tour) -> Tour:
    "Perform 2-opt segment reversals to optimize tour."
    changed = False
    for (i, j) in subsegments(len(tour)):
        if reversal_is_improvement(tour, i, j):
            tour[i:j] = reversed(tour[i:j])
            changed = True
    return (tour if not changed else opt2(tour))


def reversal_is_improvement(tour, i, j) -> bool:
    "Would reversing the segment `tour[i:j]` make the tour shorter?"
    A, B, C, D = tour[i - 1], tour[i], tour[j - 1], tour[j % len(tour)]
    return distance(A, B) + distance(C, D) > distance(A, C) + distance(B, D)


cache = functools.lru_cache(None)


@cache
def subsegments(N) -> Tuple[Tuple[int, int]]:
    "Return (i, j) index pairs denoting tour[i:j] subsegments of length N."
    return tuple(
        (i, i + length)
        for length in reversed(range(2, N - 1))
        for i in range(N - length)
    )


def rep_opt2_nearest_tsp(cities, k=10) -> Tour:
    "Apply 2-opt to each of the repeated nearest neighbor tours."
    return shortest(
        opt2(nearest_tsp(cities, start))
        for start in sample(cities, k)
    )


def greedy_tsp(cities):
    "Go through links, shortest first. If a link can join segments, do it."
    endpoints = {C: [C] for C in cities}
    links = itertools.combinations(cities, 2)
    for (A, B) in sorted(links, key=lambda link: distance(*link)):
        if A in endpoints and B in endpoints and endpoints[A] != endpoints[B]:
            joined_segment = join_segments(endpoints, A, B)
            if len(joined_segment) == len(cities):
                return joined_segment


def join_segments(endpoints, A, B):
    "Join segments [...,A] + [B,...] into one segment."
    Aseg, Bseg = endpoints[A], endpoints[B]
    if Aseg[-1] is not A: Aseg.reverse()
    if Bseg[0] is not B: Bseg.reverse()
    Aseg += Bseg
    del endpoints[A], endpoints[B]
    endpoints[Aseg[0]] = endpoints[Aseg[-1]] = Aseg
    return Aseg


def possible_tours(cities) -> List[Tour]:
    "Return all non-redundant tours (permutations with first city first)."
    start, *others = cities
    return [[start, *perm] for perm in itertools.permutations(others)]


def first(collection):
    "The first element of a collection."
    return next(iter(collection))


# ===========================================================================
# Named Tuples (unchanged)
# ===========================================================================

class RegionalTourResult(namedtuple('_', 'region, tour, length, secs, num_sites')):
    """Result for a single region's TSP tour."""
    def __repr__(self):
        return (f"Region: {self.region:>20} | Sites: {self.num_sites:>3} | "
                f"Length: {round(self.length):>6,d} miles | Time: {self.secs:6.3f}s")


all_results = defaultdict(list)


class Result(namedtuple('_', 'tsp, opt, tour, cities, secs')):
    """A Result records the results of a run on a TSP."""
    def __repr__(self):
        best = min(
            [tour_length(r.tour) for r in all_results[self.cities]],
            default=tour_length(self.tour)
        )
        return (
            f"{name(self.tsp, self.opt):>25}: length "
            f"{round(tour_length(self.tour)):,d} tour "
            f"({tour_length(self.tour) / best:5.1%}) in {self.secs:6.3f} secs"
        )


def name(tsp, opt=None) -> str:
    return tsp.__name__ + (('+' + opt.__name__) if opt else '')


# ===========================================================================
# Visualization Helpers (unchanged)
# ===========================================================================

def Xs(cities) -> List[float]:
    "X coordinates"
    return [c.real for c in cities]


def Ys(cities) -> List[float]:
    "Y coordinates"
    return [c.imag for c in cities]


def plot_segment(segment: Segment, style='bo:', color=None):
    "Plot every city and link in the segment."
    if color is not None:
        plt.plot(Xs(segment), Ys(segment), style, linewidth=2 / 3,
                 markersize=4, clip_on=False, color=color)
    else:
        plt.plot(Xs(segment), Ys(segment), style, linewidth=2 / 3,
                 markersize=4, clip_on=False)
    plt.axis('scaled')
    plt.axis('off')


# ===========================================================================
# DuckDB Data Layer (replaces dict-based parse_regional_locations)
# ===========================================================================

def get_regional_cities(con):
    """Query DuckDB for facilities grouped by region and delivery method.

    Returns:
        groups: dict mapping (region, method) -> frozenset of City (complex)
        facility_maps: dict mapping (region, method) -> {complex_city: facility_id}
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

    groups = {}
    facility_maps = {}

    for fid, lon, lat, region, method in rows:
        key = (region, method)
        city = complex(lon, lat)

        if key not in groups:
            groups[key] = set()
            facility_maps[key] = {}

        groups[key].add(city)
        facility_maps[key][city] = fid

    # Convert sets to frozensets
    groups = {k: frozenset(v) for k, v in groups.items()}

    return groups, facility_maps


def run_regional_tsp(con, tsp_algo=greedy_tsp, optimize=True):
    """Run TSP for each region/delivery method group from the graph DB.

    Args:
        con: DuckDB connection
        tsp_algo: TSP algorithm to use (default: greedy_tsp)
        optimize: Whether to apply 2-opt optimization (default: True)

    Returns:
        results: nested dict results[region][method] = RegionalTourResult
        facility_maps: dict mapping (region, method) -> {complex: facility_id}
    """
    groups, facility_maps = get_regional_cities(con)
    results = {}

    for (region, method), cities in groups.items():
        if len(cities) < 2:
            print(f"Skipping {region}/{method}: only {len(cities)} site(s)")
            continue

        if region not in results:
            results[region] = {}

        try:
            t0 = time.perf_counter()
            tour = tsp_algo(cities)
            if optimize:
                tour = opt2(tour)
            t1 = time.perf_counter()

            length = tour_length(tour)
            results[region][method] = RegionalTourResult(
                region=region,
                tour=tour,
                length=length,
                secs=t1 - t0,
                num_sites=len(cities)
            )
            print(f"  {region} / {method}: {len(cities)} sites, "
                  f"{length:.0f} mi, {t1 - t0:.3f}s")
        except Exception as e:
            print(f"Error processing {region}/{method}: {e}")

    return results, facility_maps


def write_routes_to_graph(con, results, facility_maps):
    """Write computed TSP routes back to the graph database.

    - part_of_route: The actual optimal route edges from the TSP solution

    Args:
        con: DuckDB connection (must be read-write)
        results: nested dict results[region][method] = RegionalTourResult
        facility_maps: dict mapping (region, method) -> {complex: facility_id}
    """
    # Clear previous route data
    con.execute("DELETE FROM part_of_route")

    route_id = 0
    total_edges = 0

    for region in results:
        for method in results[region]:
            result = results[region][method]
            tour = result.tour
            route_id += 1
            fmap = facility_maps.get((region, method), {})

            # Write part_of_route edges (sequential tour edges)
            for seq in range(len(tour)):
                city_a = tour[seq]
                city_b = tour[(seq + 1) % len(tour)]
                fid_a = fmap.get(city_a)
                fid_b = fmap.get(city_b)

                if fid_a is None or fid_b is None:
                    continue

                dist = distance(city_a, city_b)
                con.execute(
                    "INSERT INTO part_of_route VALUES (?, ?, ?, ?, ?)",
                    [fid_a, fid_b, route_id, seq, dist]
                )
                total_edges += 1

    print(f"Wrote {total_edges} route edges across {route_id} routes "
          f"to part_of_route table.")


def convert_results_for_crewai(results):
    """Convert RegionalTourResult namedtuples to dicts for CrewAI.

    Input: results[region][delivery_method] = RegionalTourResult
    Output: Same nested structure but with JSON-serializable dicts
    """
    converted = {}
    for region, delivery_methods in results.items():
        converted[region] = {}
        for dm, result in delivery_methods.items():
            if isinstance(result, dict):
                converted[region][dm] = result
            else:
                converted[region][dm] = {
                    'region': result.region,
                    'tour': [{'longitude': city.real, 'latitude': city.imag}
                             for city in result.tour],
                    'length': result.length,
                    'secs': result.secs,
                    'num_sites': result.num_sites
                }
    return converted


# ===========================================================================
# Visualization
# ===========================================================================

def plot_regional_tours(con, results):
    """Plot all regional tours on separate subplots.

    Args:
        con: DuckDB connection (for facility count context)
        results: nested dict results[region][method] = RegionalTourResult
    """
    num_regions = len(results)
    if num_regions == 0:
        print("No regional tour results to plot.")
        return

    cols = min(3, num_regions)
    rows = (num_regions + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15 * cols, 12 * rows))

    if num_regions == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_regions > 1 else [axes]

    # Load Alaska boundary
    try:
        url = ("https://www2.census.gov/geo/tiger/GENZ2018/shp/"
               "cb_2018_us_state_20m.zip")
        states = gpd.read_file(url)
        alaska = states[states['NAME'] == 'Alaska']
    except Exception:
        alaska = None
        print("Could not load Alaska boundary")

    # Plot each region
    for idx, (region, groups) in enumerate(sorted(results.items())):
        ax = axes[idx]
        plt.sca(ax)

        if alaska is not None:
            alaska.boundary.plot(ax=ax, color='black', linewidth=0.5, zorder=0)

        total_sites = sum(
            results[region][gn].num_sites
            for gn in groups if gn in results[region]
        )
        ax.set_title(f"{region}\n{total_sites} sites, {len(groups)} tours")

        legend_list = []
        for group_name in sorted(groups.keys()):
            tour = results[region][group_name].tour
            num_sites = results[region][group_name].num_sites
            length = results[region][group_name].length

            # Color by delivery method
            color_map = {
                'Road': 'orange', 'Plane': 'blue', 'Barge': 'green',
                'Plane or Road': 'purple', 'Unknown': 'black', 'NaN': 'black'
            }
            color = color_map.get(group_name, 'gray')

            if group_name in ('NaN', 'Unknown'):
                ax.scatter(Xs(tour), Ys(tour), color=color, s=10,
                           marker='x')
                label = (f"Unknown: {num_sites} sites, {length:.1f} mi")
                legend_list.append(
                    plt.Line2D([0], [0], color=color, marker='x',
                               linestyle='None', label=label))
            else:
                plot_segment(tour, style='o-', color=color)
                plot_segment([tour[0]], style='s', color=color)
                label = f"{group_name}: {num_sites} sites, {length:.1f} mi"
                legend_list.append(
                    plt.Line2D([0], [0], color=color, marker='o',
                               linestyle='-', label=label))

        if legend_list:
            ax.legend(handles=legend_list, loc='best', fontsize=8)

        ax.axis('on')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim(-180, -130)
        ax.set_ylim(50, 75)

    # Turn off extra subplots
    for idx in range(num_regions, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_final_graph(con):
    """Visualize the final graph state with TSP routes overlaid on adjacency.

    After TSP routes are written to part_of_route, this renders:
    - Facility nodes at geographic positions, colored by region
    - adjacent_to edges as faint gray lines (context)
    - part_of_route edges as bold colored lines (optimized routes)
    - Route edges colored by delivery method

    Args:
        con: DuckDB connection
    """
    # Fetch facilities with positions and metadata
    facilities_df = con.execute("""
        SELECT f.facility_id, f.longitude, f.latitude,
               li.region_name,
               COALESCE(um.method_name, 'Unknown') AS method
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        WHERE li.region_name != 'Unassigned'
    """).fetchdf()

    # Fetch connectivity edges (background context)
    adjacency_df = con.execute("""
        SELECT src, dst, distance_miles
        FROM connects_to
        WHERE distance_miles <= 100
    """).fetchdf()

    # Fetch optimized routes
    routes_df = con.execute("""
        SELECT pr.src, pr.dst, pr.route_id, pr.sequence,
               pr.distance_miles
        FROM part_of_route pr
        ORDER BY pr.route_id, pr.sequence
    """).fetchdf()

    # Build NetworkX graph
    G = nx.Graph()

    # Color setup
    palette = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
               '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
               '#469990', '#dcbeff', '#9A6324', '#808000']
    region_colors = {}
    node_color_map = {}

    # Build facility lookup for delivery method on edges
    fid_to_method = {}

    for _, row in facilities_df.iterrows():
        fid = int(row['facility_id'])
        region = row['region_name']
        method = row['method']

        if region not in region_colors:
            region_colors[region] = palette[len(region_colors) % len(palette)]

        G.add_node(fid, pos=(row['longitude'], row['latitude']),
                   region=region, method=method)
        node_color_map[fid] = region_colors[region]
        fid_to_method[fid] = method

    # Add adjacency edges (faint background)
    adj_edges = []
    for _, row in adjacency_df.iterrows():
        src, dst = int(row['src']), int(row['dst'])
        if src in G.nodes and dst in G.nodes:
            adj_edges.append((src, dst))

    # Add route edges (bold foreground)
    route_edges = []
    route_edge_colors = []
    method_color_map = {
        'Road': 'orange', 'Plane': 'blue', 'Barge': 'green',
        'Plane or Road': 'purple', 'Unknown': 'black'
    }

    for _, row in routes_df.iterrows():
        src, dst = int(row['src']), int(row['dst'])
        if src in G.nodes and dst in G.nodes:
            route_edges.append((src, dst))
            method = fid_to_method.get(src, 'Unknown')
            route_edge_colors.append(method_color_map.get(method, 'gray'))

    # Draw
    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [node_color_map.get(n, 'gray') for n in G.nodes()]

    # Background: adjacency edges
    if adj_edges:
        nx.draw_networkx_edges(G, pos, edgelist=adj_edges, alpha=0.08,
                               edge_color='gray', ax=ax)

    # Foreground: route edges
    if route_edges:
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, alpha=0.7,
                               edge_color=route_edge_colors, width=2, ax=ax)

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30,
                           ax=ax)

    # Legend: regions
    for region, color in sorted(region_colors.items()):
        ax.scatter([], [], c=color, label=f"Region: {region}", s=50)

    # Legend: delivery method edges
    for method, color in method_color_map.items():
        if method != 'Unknown':
            ax.plot([], [], color=color, linewidth=2,
                    label=f"Route: {method}")

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Final Graph State: TSP Routes Overlaid on Facility Network')
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    route_stats = con.execute("""
        SELECT COUNT(*) AS total_edges,
               COUNT(DISTINCT route_id) AS total_routes,
               ROUND(SUM(distance_miles), 1) AS total_distance
        FROM part_of_route
    """).fetchone()
    print(f"\nFinal Graph Summary:")
    print(f"  Facility nodes: {G.number_of_nodes()}")
    print(f"  Adjacency edges (<=100mi): {len(adj_edges)}")
    print(f"  Route edges: {route_stats[0]}")
    print(f"  Total routes: {route_stats[1]}")
    print(f"  Total route distance: {route_stats[2]} miles")


# ===========================================================================
# Graph-Aware Agent Tools
# ===========================================================================

@tool("query_tsp_routes")
def query_tsp_routes() -> str:
    """Query the graph database for all computed TSP routes with region,
    delivery method, distances, and route statistics."""
    global graph_con
    result = graph_con.execute("""
        SELECT
            pr.route_id,
            li_src.region_name AS region,
            COALESCE(um_src.method_name, 'Unknown') AS delivery_method,
            COUNT(*) AS segments,
            ROUND(SUM(pr.distance_miles), 1) AS total_distance,
            ROUND(AVG(pr.distance_miles), 1) AS avg_segment_distance,
            ROUND(MAX(pr.distance_miles), 1) AS max_segment_distance,
            ROUND(MIN(pr.distance_miles), 1) AS min_segment_distance
        FROM part_of_route pr
        JOIN located_in li_src ON pr.src = li_src.facility_id
        LEFT JOIN uses_method um_src ON pr.src = um_src.facility_id
        GROUP BY pr.route_id, li_src.region_name, delivery_method
        ORDER BY total_distance DESC
    """).fetchdf()
    return result.to_json(orient='records', indent=2)


@tool("query_facility_connections")
def query_facility_connections(region: str) -> str:
    """Query the graph for all facility connections in a specific region,
    including delivery methods, adjacency distances, and route membership.

    Args:
        region: The region name to query
    """
    global graph_con
    result = graph_con.execute("""
        SELECT
            f1.facility_id AS source_id,
            COALESCE(um1.method_name, 'Unknown') AS source_method,
            f2.facility_id AS dest_id,
            COALESCE(um2.method_name, 'Unknown') AS dest_method,
            a.distance_miles AS adjacency_distance,
            pr.route_id,
            pr.sequence AS route_sequence,
            pr.distance_miles AS route_distance
        FROM facilities f1
        JOIN located_in li ON f1.facility_id = li.facility_id
        JOIN connects_to a ON f1.facility_id = a.src
        JOIN facilities f2 ON f2.facility_id = a.dst
        LEFT JOIN uses_method um1 ON f1.facility_id = um1.facility_id
        LEFT JOIN uses_method um2 ON f2.facility_id = um2.facility_id
        LEFT JOIN part_of_route pr ON (
            (pr.src = f1.facility_id AND pr.dst = f2.facility_id) OR
            (pr.src = f2.facility_id AND pr.dst = f1.facility_id)
        )
        WHERE li.region_name = ?
        ORDER BY a.distance_miles
        LIMIT 200
    """, [region]).fetchdf()
    return result.to_json(orient='records', indent=2)


@tool("get_route_summary")
def get_route_summary() -> str:
    """Get a summary of all computed routes with total distance per
    region/method, number of facilities, and route efficiency metrics."""
    global graph_con
    # Route-level summary
    route_summary = graph_con.execute("""
        SELECT
            li.region_name AS region,
            COALESCE(um.method_name, 'Unknown') AS delivery_method,
            pr.route_id,
            COUNT(DISTINCT pr.src) + 1 AS facilities_in_route,
            ROUND(SUM(pr.distance_miles), 1) AS total_distance,
            ROUND(AVG(pr.distance_miles), 1) AS avg_segment,
            ROUND(MAX(pr.distance_miles), 1) AS longest_segment,
            ROUND(MIN(pr.distance_miles), 1) AS shortest_segment
        FROM part_of_route pr
        JOIN located_in li ON pr.src = li.facility_id
        LEFT JOIN uses_method um ON pr.src = um.facility_id
        GROUP BY li.region_name, delivery_method, pr.route_id
        ORDER BY total_distance DESC
    """).fetchdf()

    # Overall totals
    totals = graph_con.execute("""
        SELECT
            COUNT(DISTINCT route_id) AS total_routes,
            COUNT(*) AS total_segments,
            ROUND(SUM(distance_miles), 1) AS total_distance,
            ROUND(AVG(distance_miles), 1) AS avg_segment_distance
        FROM part_of_route
    """).fetchdf()

    return json.dumps({
        "route_details": json.loads(route_summary.to_json(orient='records')),
        "overall_totals": json.loads(totals.to_json(orient='records'))
    }, indent=2)


# ===========================================================================
# Graph-Write Tools for TSP Adjuster
# ===========================================================================

@tool("remove_route_segment")
def remove_route_segment(route_id: int, sequence: int) -> str:
    """Remove a specific segment from a route by route_id and sequence number.

    Args:
        route_id: The route ID to modify
        sequence: The sequence number of the segment to remove
    """
    global graph_con
    graph_con.execute(
        "DELETE FROM part_of_route WHERE route_id = ? AND sequence = ?",
        [route_id, sequence]
    )
    remaining = graph_con.execute(
        "SELECT COUNT(*) FROM part_of_route WHERE route_id = ?",
        [route_id]
    ).fetchone()[0]
    return f"Removed segment {sequence} from route {route_id}. {remaining} segments remaining."


@tool("insert_route_segment")
def insert_route_segment(src_facility_id: int, dst_facility_id: int,
                         route_id: int, sequence: int,
                         distance_miles: float) -> str:
    """Insert a new segment into a route.

    Args:
        src_facility_id: Source facility ID
        dst_facility_id: Destination facility ID
        route_id: The route ID to add to
        sequence: The sequence position for this segment
        distance_miles: Distance in miles for this segment
    """
    global graph_con
    graph_con.execute(
        "INSERT INTO part_of_route VALUES (?, ?, ?, ?, ?)",
        [src_facility_id, dst_facility_id, route_id, sequence, distance_miles]
    )
    return (f"Inserted segment: {src_facility_id} -> {dst_facility_id} "
            f"(route {route_id}, seq {sequence}, {distance_miles:.1f} mi)")


@tool("split_route")
def split_route(route_id: int, split_after_sequence: int) -> str:
    """Split a route into two separate routes at the given sequence point.
    Segments with sequence <= split_after_sequence stay in the original route.
    Segments with sequence > split_after_sequence get a new route_id.

    Args:
        route_id: The route ID to split
        split_after_sequence: Split after this sequence number
    """
    global graph_con
    # Find next available route_id
    max_id = graph_con.execute(
        "SELECT COALESCE(MAX(route_id), 0) FROM part_of_route"
    ).fetchone()[0]
    new_route_id = max_id + 1

    # Update segments after the split point to the new route
    graph_con.execute("""
        UPDATE part_of_route
        SET route_id = ?, sequence = sequence - ? - 1
        WHERE route_id = ? AND sequence > ?
    """, [new_route_id, split_after_sequence, route_id, split_after_sequence])

    # Count segments in each route
    orig_count = graph_con.execute(
        "SELECT COUNT(*) FROM part_of_route WHERE route_id = ?", [route_id]
    ).fetchone()[0]
    new_count = graph_con.execute(
        "SELECT COUNT(*) FROM part_of_route WHERE route_id = ?", [new_route_id]
    ).fetchone()[0]

    return (f"Split route {route_id} after sequence {split_after_sequence}. "
            f"Original route: {orig_count} segments. "
            f"New route {new_route_id}: {new_count} segments.")


# ===========================================================================
# Agent & Task Setup
# ===========================================================================

def setup_agents(llm, tsp_results_dict, input_report):
    """Create all agents and tasks for the TSP analysis.

    Args:
        llm: CrewAI LLM instance
        tsp_results_dict: Converted TSP results for agent context
        input_report: Market & cost analysis report JSON

    Returns:
        Tuple of (agents_list, tasks_list)
    """

    # ----- Agent 1: TSP Route Optimizer -----
    tsp_agent = Agent(
        role="TSP Route Optimizer",
        goal="Query the graph database for computed TSP routes and present "
             "accurate route data including distances, segments, and "
             "facility connections for each region and delivery method.",
        backstory="Expert in Traveling Salesperson solutions with access to "
                  "a DuckDB graph database containing facility data and "
                  "computed routes. You must query the database and report "
                  "only what the data shows. Do not fabricate route details, "
                  "distances, or facility information — only report what "
                  "the tools return.",
        verbose=True,
        llm=llm,
        tools=[query_tsp_routes, query_facility_connections, get_route_summary]
    )

    tsp_task = Task(
        description=f"""Retrieve and summarize the optimized delivery routes
        from the graph database.

        IMPORTANT: Base your output strictly on data returned by your tools.
        Do not invent or assume route details not present in the data.

        1. Use get_route_summary to get an overview of all routes
           (total distance, segment counts, facilities per route).
        2. Use query_tsp_routes for detailed per-route statistics
           (avg/min/max segment distances by region and delivery method).
        3. For each region, use query_facility_connections to report
           facility connections and route membership.

        The TSP results summary: {json.dumps(tsp_results_dict, indent=2)[:3000]}

        Note: Delivery method assignment was already handled in the
        regionalization step. Do not reassign or recommend delivery methods.

        Present the route data in a clear, structured format. Report the
        facts — leave analysis and recommendations to other agents.""",
        agent=tsp_agent,
        expected_output="A structured summary of all computed routes with "
                       "distances, segment counts, and facility details "
                       "from the graph database."
    )

    # ----- Agent 2: Cost Estimator -----
    cost_estimator_agent = Agent(
        role="Cost Estimator Agent",
        goal="Assign cost values to each segment of each route based on "
             "distance, delivery method, and operational factors.",
        backstory="Energy delivery cost specialist in Alaska with expertise "
                  "in fuel transportation economics.",
        verbose=True,
        llm=llm
    )

    cost_estimation_task = Task(
        description="""Estimate the cost for each delivery segment from the
        TSP route data.

        Consider these factors:
        - **Distance:** Longer segments cost more. Use distances from routes.
        - **Delivery Method:** Cost per mile varies significantly:
          - Road: ~$2-5/mile (fuel truck)
          - Barge: ~$1-3/mile (but seasonal, bulk quantities)
          - Plane: ~$8-15/mile (small aircraft, limited cargo)
        - **Fuel Price:** Current Alaska fuel prices as baseline
        - **Operational Costs:** Labor, maintenance, infrastructure per method

        Provide cost estimates per segment and per route. Present in a
        structured format usable by other agents.""",
        agent=cost_estimator_agent,
        context=[tsp_task],
        expected_output="A structured list of segments with estimated costs."
    )

    # ----- Agent 3: Operational Risk Agent -----
    operational_risk_agent = Agent(
        role="Operational Risk Analyst",
        goal="Analyze fuel delivery segment risks by region considering "
             "weather, delivery method, and cost factors.",
        backstory="Expert in logistics risk assessment with deep knowledge "
                  "of Arctic operations and supply chain vulnerabilities.",
        verbose=True,
        llm=llm
    )

    operational_risk_task = Task(
        description=f"""Assess the risk for each fuel delivery segment.
        Use the TSP Agent's route analysis and Cost Estimator's cost data.
        Reference the market analysis: {str(input_report)[:2000]}

        For each segment/route, evaluate:
        - **Weather:** Storm, ice, fog impact on segment feasibility
        - **Delivery Method:** Inherent risks (road conditions, ice for
          barges, visibility for air)
        - **Cost:** High cost segments may indicate challenging routes

        Classify each route into risk categories:
        - High risk: Significant potential for delays/failure
        - Moderate risk: Some risks but manageable
        - Low risk: Minimal anticipated risks

        Suggest alternatives for high-risk segments.""",
        agent=operational_risk_agent,
        context=[tsp_task],
        expected_output="A detailed risk analysis for each route/segment."
    )

    # Discussion between Risk and Cost agents
    operational_cost_discussion = Task(
        description="""Review cost estimates and factor them into risk
        assessment. Adjust costs based on operational risk. For routes
        with dual delivery methods (e.g., 'Plane or Road'), select the
        optimal method based on cost-risk balance.""",
        agent=operational_risk_agent,
        expected_output="An informed cost-risk assessment for each route."
    )

    # ----- Agent 4: Route Analyzer -----
    route_analyzer_agent = Agent(
        role="Route Analyzer",
        goal="Assess whether computed delivery routes are practical, "
             "cost-effective, and operationally sound by synthesizing "
             "route data, cost estimates, and risk assessments.",
        backstory="Expert in logistics network analysis with deep knowledge "
                  "of Alaska's geography and fuel delivery constraints. "
                  "You evaluate route quality by combining route data with "
                  "cost and risk inputs from other agents. You identify "
                  "routes that are too long, too costly, or ineffective "
                  "and recommend improvements.",
        verbose=True,
        llm=llm
    )

    route_analysis_task = Task(
        description="""Analyze the computed delivery routes using the route
        data from the TSP Route Optimizer, cost estimates from the Cost
        Estimator, and risk assessments from the Operational Risk Analyst.

        For each region/delivery method route, assess:
        1. **Efficiency:** Are there unusually long segments that suggest
           the route could be improved? Does the route make geographic sense?
        2. **Cost-effectiveness:** Based on cost estimates, which routes
           have the highest cost per mile or per facility? Are there
           cheaper alternatives?
        3. **Operational viability:** Based on risk assessments, which
           routes face the highest operational risk? Are high-cost routes
           also high-risk?
        4. **Recommendations:** Identify routes that might benefit from
           alternative groupings, delivery methods, or splitting into
           sub-routes.

        Note: Delivery method assignment and resolution of dual methods
        was already handled in the regionalization step. Do not reassign
        delivery methods — focus on route quality assessment.

        Provide a clear assessment for each route with actionable
        recommendations.""",
        agent=route_analyzer_agent,
        context=[tsp_task, cost_estimation_task, operational_risk_task],
        expected_output="A route-by-route assessment with efficiency, cost, "
                       "and risk evaluations plus actionable recommendations."
    )

    # ----- Agent 5: TSP Adjuster -----
    tsp_adjuster_agent = Agent(
        role="TSP Route Adjuster",
        goal="Fix unrealistic or inefficient routes identified by the Route "
             "Analyzer by modifying route segments in the graph database.",
        backstory="Expert in route optimization and graph database operations. "
                  "You take specific recommendations from the Route Analyzer "
                  "and apply corrections to routes in the graph database. "
                  "You can remove inefficient segments, insert better "
                  "connections, and split overly large routes. You must query "
                  "routes before modifying them to understand the current "
                  "state, and verify changes after making them. Only modify "
                  "routes that were flagged as problematic — do not change "
                  "routes that are working well.",
        verbose=True,
        llm=llm,
        tools=[query_tsp_routes, get_route_summary, query_facility_connections,
               remove_route_segment, insert_route_segment, split_route]
    )

    tsp_adjuster_task = Task(
        description="""Review the Route Analyzer's assessment and fix routes
        that were flagged as unrealistic, too costly, or ineffective.

        IMPORTANT: Query routes BEFORE modifying them. Verify changes AFTER
        making them. Only modify routes that were specifically flagged.

        For each flagged route:
        1. Use query_tsp_routes or get_route_summary to examine the current
           route state
        2. Based on the Route Analyzer's recommendation, apply fixes:
           - For routes that are too long: use split_route to break them
             into smaller sub-routes
           - For segments that are inefficient: use remove_route_segment
             to remove the problematic segment, then insert_route_segment
             to add a better connection
           - For routes with geographic issues: restructure segments as
             needed
        3. After each modification, use get_route_summary to verify the
           change improved the route

        Report all changes made and their impact on route distances.""",
        agent=tsp_adjuster_agent,
        context=[route_analysis_task],
        expected_output="A summary of all route modifications made, with "
                       "before/after distances and verification that changes "
                       "improved route quality."
    )

    # ----- Agent 6: Writing Agent -----
    writing_agent = Agent(
        role="Fuel Delivery Analyst and Report Writer",
        goal="Write an engaging report with analysis and actionable "
             "recommendations for fuel delivery leaders in Alaska.",
        backstory="Expert in fuel delivery economics, policy, logistics, "
                  "and technical writing. Skilled at synthesizing complex "
                  "information from multiple agents.",
        verbose=True,
        llm=llm
    )

    multi_agent_discussion_task = Task(
        description="""Lead a discussion synthesizing findings from the
        TSP Route Optimizer, Route Analyzer, TSP Route Adjuster, Cost
        Estimator, and Operational Risk Agent.

        * TSP Route Optimizer: Route data from the graph database
        * Route Analyzer: Route efficiency, cost-effectiveness, and
          viability assessments with recommendations
        * TSP Route Adjuster: Route modifications made and their impact
        * Cost Estimator: Cost estimates per segment/route
        * Operational Risk Agent: Risk assessments and alternatives

        As moderator:
        1. Synthesize route data, analysis, adjustments, cost, and risk
        2. Identify key trade-offs (cost vs. risk vs. efficiency)
        3. Evaluate whether route adjustments addressed the concerns
        4. Identify the most important findings for the final report""",
        agent=writing_agent,
        expected_output="A structured summary with key points and insights."
    )

    # ----- Agent 5: Contrarian Agent -----
    contrarian_agent = Agent(
        role="Supply Chain and Logistics Contrarian",
        goal="Critically evaluate route analyses and recommendations, "
             "identifying logical inconsistencies and oversights.",
        backstory="Seasoned supply chain expert with a contrarian mindset. "
                  "Excels at identifying reasoning flaws and finding "
                  "alternative explanations.",
        verbose=True,
        llm=llm
    )

    contrarian_task = Task(
        description="""Critically review the TSP route analysis, cost
        estimates, risk assessments, and recommendations.

        Identify:
        1. Logical inconsistencies or contradictions
        2. Potential oversights or missing considerations
        3. Alternative route strategies not considered
        4. Unrealistic cost or risk assumptions
        5. Areas needing additional data or analysis

        Present findings in a structured format.""",
        agent=contrarian_agent,
        expected_output="A critical review with specific areas of concern."
    )

    writing_response_task = Task(
        description="""Respond to the contrarian critique. For each point:
        1. Justify your analysis where critique is unfounded
        2. Acknowledge valid concerns
        3. Propose revisions where warranted""",
        agent=writing_agent,
        expected_output="A response with justifications and revisions."
    )

    contrarian_followup_task = Task(
        description="""Review the Writing Agent's response. Provide
        follow-up critiques or confirm concerns have been addressed.""",
        agent=contrarian_agent,
        expected_output="Follow-up critiques or confirmation of resolution."
    )

    writing_task = Task(
        description="""Produce a comprehensive final report integrating
        all agent analyses, discussions, and critiques.

        Include:
        1. An engaging narrative on Alaska fuel delivery routes
        2. Route analysis with optimization findings
        3. Cost-risk assessment for each region's routes
        4. Clear, actionable recommendations
        5. Agent Discussion Summary
        6. Limitations (contrarian critique summary)

        ***Return a plain text document, NOT JSON.***""",
        agent=writing_agent,
        expected_output="A comprehensive plain text report with analysis, "
                       "recommendations, and contrarian review."
    )

    agents = [tsp_agent, cost_estimator_agent, operational_risk_agent,
              route_analyzer_agent, tsp_adjuster_agent, writing_agent,
              contrarian_agent]

    tasks = [
        tsp_task,                      # Phase 1: Route data retrieval (data-grounded)
        cost_estimation_task,          # Phase 1: Cost estimation
        operational_risk_task,         # Phase 1: Risk assessment
        operational_cost_discussion,   # Phase 2: Cost-risk discussion
        route_analysis_task,           # Phase 2: Route analysis (uses cost + risk)
        tsp_adjuster_task,             # Phase 3: Fix flagged routes in graph DB
        multi_agent_discussion_task,   # Phase 4: Multi-agent synthesis
        contrarian_task,               # Phase 5: Contrarian review
        writing_response_task,         # Phase 5: Writing response
        contrarian_followup_task,      # Phase 5: Contrarian follow-up
        writing_task                   # Phase 6: Final report
    ]

    return agents, tasks


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Run the TSP model pipeline with DuckDB graph database."""
    global graph_con

    # LLM and API setup
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(f"GEMINI_API_KEY={pipeline.get_api_key()}\n")
        f.write("MODEL=gemini/gemini-2.5-flash-preview-04-17\n")

    load_dotenv()
    os.environ["GEMINI_API_KEY"] = pipeline.get_api_key()
    llm = LLM(model='gemini/gemini-2.5-flash')

    # Connect to graph database (read-write for writing routes)
    graph_con = duckdb.connect('regionalization.duckdb')
    print("Connected to graph database: regionalization.duckdb")

    # Load market analysis report
    input_report = {}
    try:
        with open('market_cost_analysis_report.json', 'r') as f:
            input_report = json.load(f)
        print("Loaded market_cost_analysis_report.json")
    except FileNotFoundError:
        print("Warning: market_cost_analysis_report.json not found. "
              "Proceeding without market analysis context.")

    # Print graph summary
    print("\nGraph Database Summary:")
    for table in ['facilities', 'regions', 'delivery_methods',
                  'located_in', 'uses_method', 'adjacent_to',
                  'connects_to', 'part_of_route']:
        count = graph_con.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        print(f"  {table}: {count} rows")

    # Run regional TSP
    print("\nRunning TSP for each region/delivery method group...")
    print("=" * 60)
    regional_results, facility_maps = run_regional_tsp(
        graph_con, tsp_algo=greedy_tsp, optimize=True
    )

    # Write routes to graph database
    print("\nWriting routes to graph database...")
    write_routes_to_graph(graph_con, regional_results, facility_maps)

    # Visualize regional tours
    print("\nPlotting regional tours...")
    plot_regional_tours(graph_con, regional_results)

    # Visualize final graph state
    print("\nVisualizing final graph state...")
    visualize_final_graph(graph_con)

    # Convert results for CrewAI
    tsp_results_dict = convert_results_for_crewai(regional_results)

    # Set up agents and tasks
    agents, tasks = setup_agents(llm, tsp_results_dict, input_report)

    # Configure crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        llm=llm
    )

    print("\nRunning TSP Analysis Crew...")
    print("=" * 60)

    result = crew.kickoff(inputs={
        'regional_tsp_results': tsp_results_dict
    })

    print("\n" + "=" * 60)
    print("CREW EXECUTION COMPLETE")
    print("=" * 60)
    print(result)

    # Save report
    with open('tsp_final_report.json', 'w', encoding='utf-8') as f:
        json.dump({"result": str(result)}, f, indent=4)

    # Print final graph state
    print("\nFinal graph database state:")
    for table in ['facilities', 'regions', 'delivery_methods',
                  'located_in', 'uses_method', 'adjacent_to',
                  'connects_to', 'part_of_route']:
        count = graph_con.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        print(f"  {table}: {count} rows")

    # Close connection
    graph_con.close()
    print("\nDuckDB connection closed. Routes saved to regionalization.duckdb.")


if __name__ == "__main__":
    main()
