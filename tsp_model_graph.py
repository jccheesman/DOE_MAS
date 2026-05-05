# -*- coding: utf-8 -*-
"""tsp_model_graph.py

DuckDB Graph Database version of the TSP model module. Queries the
shared DuckDB graph database for facility data and writes computed
routes back to the graph as edges.

Agents:
    - TSP Route Optimizer: Queries graph DB for computed route data
    - Operational Risk Agent: Assesses segment risks using friction-based
      costs and terrain data from the graph database
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
from datetime import date
warnings.filterwarnings('ignore', category=DeprecationWarning)

import json
import math
import time
import random
import functools
import itertools
import duckdb
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import geopandas as gpd
from typing import Set, List, Tuple, Iterable, Callable, Dict

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import pipeline

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
pipeline.set_cwd('/media/volume/GraphDB_Runs')

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
# Core TSP Algorithm Functions
# ===========================================================================

# Friction-weighted distance lookup populated before TSP runs.
# Maps (City, City) -> friction-based cost.  Falls back to Haversine
# when no friction data is available for an edge.
_friction_weights: Dict[Tuple[City, City], float] = {}


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


def weighted_distance(A: City, B: City) -> float:
    """Friction-weighted distance between two cities.

    Uses friction-based delivery cost from the graph database if available,
    falling back to Haversine distance when no friction data exists.
    """
    key = (A, B)
    if key in _friction_weights:
        return _friction_weights[key]
    key_rev = (B, A)
    if key_rev in _friction_weights:
        return _friction_weights[key_rev]
    return distance(A, B)


def shortest(tours: Iterable[Tour]) -> Tour:
    "The tour with the smallest tour length."
    return min(tours, key=tour_length)


def tour_length(tour: Tour) -> float:
    "The total of weighted distance for each link in the tour, including last to first."
    return sum(weighted_distance(tour[i], tour[i - 1]) for i in range(len(tour)))


def valid_tour(tour: Tour, cities: Cities) -> bool:
    "Does `tour` visit every city in `cities` exactly once?"
    from collections import Counter
    return Counter(tour) == Counter(cities)


def nearest_neighbor(A: City, cities) -> City:
    "Find the city C in cities that is nearest to city A (using weighted distance)."
    return min(cities, key=lambda C: weighted_distance(C, A))


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
    return (weighted_distance(A, B) + weighted_distance(C, D) >
            weighted_distance(A, C) + weighted_distance(B, D))


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
    for (A, B) in sorted(links, key=lambda link: weighted_distance(*link)):
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
               um.method_name AS method
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        JOIN uses_method um ON f.facility_id = um.facility_id
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


def _load_friction_weights(con, facility_maps):
    """Populate the global _friction_weights lookup from connects_to edges.

    Maps (City, City) pairs to their friction-based delivery_cost.  Falls
    back to avg_friction * path_length_miles when delivery_cost is NULL.
    """
    global _friction_weights
    _friction_weights.clear()

    # Build reverse map: facility_id -> City (complex)
    fid_to_city = {}
    for (region, method), city_map in facility_maps.items():
        for city, fid in city_map.items():
            fid_to_city[fid] = city

    rows = con.execute("""
        SELECT src, dst, delivery_cost, avg_friction, path_length_miles
        FROM connects_to
        WHERE avg_friction IS NOT NULL
    """).fetchall()

    loaded = 0
    for src, dst, cost, avg_f, path_mi in rows:
        src_city = fid_to_city.get(src)
        dst_city = fid_to_city.get(dst)
        if src_city is None or dst_city is None:
            continue

        # Prefer delivery_cost; fall back to friction * path length
        if cost is not None and cost < 999:
            weight = cost
        elif avg_f is not None and path_mi is not None:
            weight = avg_f * path_mi
        else:
            continue

        _friction_weights[(src_city, dst_city)] = weight
        loaded += 1

    print(f"  Loaded {loaded} friction-weighted edges for TSP")


def run_regional_tsp(con, tsp_algo=greedy_tsp, optimize=True):
    """Run TSP for each region/delivery method group from the graph DB.

    Args:
        con: DuckDB connection
        tsp_algo: TSP algorithm to use (default: greedy_tsp)
        optimize: Whether to apply 2-opt optimization (default: True)

    Returns:
        results: nested dict results[region][method] = dict with keys:
                 region, tour, length, secs, num_sites
        facility_maps: dict mapping (region, method) -> {complex: facility_id}
    """
    groups, facility_maps = get_regional_cities(con)

    # Load friction-based weights if available
    _load_friction_weights(con, facility_maps)

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
            results[region][method] = {
                'region': region,
                'tour': tour,
                'length': length,
                'secs': t1 - t0,
                'num_sites': len(cities)
            }
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
        results: nested dict results[region][method] = dict
        facility_maps: dict mapping (region, method) -> {complex: facility_id}
    """
    # Clear previous route data
    con.execute("DELETE FROM part_of_route")

    route_id = 0
    total_edges = 0

    for region in results:
        for method in results[region]:
            result = results[region][method]
            tour = result['tour']
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


def serialize_tours(results):
    """Convert complex City objects in tour dicts to JSON-serializable format.

    Input: results[region][delivery_method] = dict with 'tour' as list of complex
    Output: Same structure but with tour as list of {longitude, latitude} dicts
    """
    converted = {}
    for region, methods in results.items():
        converted[region] = {}
        for dm, result in methods.items():
            converted[region][dm] = {
                **result,
                'tour': [{'longitude': c.real, 'latitude': c.imag}
                         for c in result['tour']]
            }
    return converted


# ===========================================================================
# Visualization
# ===========================================================================

def plot_regional_tours(con, results):
    """Plot all regional tours on separate subplots.

    Args:
        con: DuckDB connection (for facility count context)
        results: nested dict results[region][method] = dict
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
            results[region][gn]['num_sites']
            for gn in groups if gn in results[region]
        )
        ax.set_title(f"{region}\n{total_sites} sites, {len(groups)} tours")

        legend_list = []
        for group_name in sorted(groups.keys()):
            tour = results[region][group_name]['tour']
            num_sites = results[region][group_name]['num_sites']
            length = results[region][group_name]['length']

            # Color by delivery method
            color_map = {
                'Road': 'orange', 'Plane': 'blue', 'Barge': 'green',
                'Unknown': 'black', 'NaN': 'black'
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
    plt.savefig("outputs/regional_tours.png", dpi=150, bbox_inches="tight")
    plt.close()


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
        'Unknown': 'black'
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
    plt.savefig("outputs/final_graph.png", dpi=150, bbox_inches="tight")
    plt.close()

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
# Directed Regional Maps (arrows on routes)
# ===========================================================================

def _load_alaska_boundary():
    """Load Alaska state boundary as a GeoDataFrame (WGS84)."""
    try:
        url = ("https://www2.census.gov/geo/tiger/GENZ2018/shp/"
               "cb_2018_us_state_20m.zip")
        states = gpd.read_file(url)
        return states[states['NAME'] == 'Alaska']
    except Exception:
        print("Could not load Alaska boundary from Census TIGER")
        return None


def plot_regional_directed_maps(con):
    """Plot one map per region showing directed routes with arrows.

    Each map shows:
    - Alaska state boundary (light gray)
    - Facility nodes colored by delivery method
    - part_of_route edges as arrows (src -> dst) colored by method
    - connects_to edges as faint background arrows

    Saves to outputs/regional_directed_{region_name}.png
    """
    os.makedirs("outputs", exist_ok=True)

    alaska = _load_alaska_boundary()

    regions = con.execute("""
        SELECT DISTINCT li.region_name
        FROM located_in li
        WHERE li.region_name != 'Unassigned'
        ORDER BY li.region_name
    """).fetchall()

    method_colors = {
        'Road': '#f58231', 'Plane': '#4363d8', 'Barge': '#3cb44b',
        'Unknown': '#333333',
    }

    for (region_name,) in regions:
        facilities = con.execute("""
            SELECT f.facility_id, f.longitude, f.latitude,
                   COALESCE(um.method_name, 'Unknown') AS method
            FROM facilities f
            JOIN located_in li ON f.facility_id = li.facility_id
            LEFT JOIN uses_method um ON f.facility_id = um.facility_id
            WHERE li.region_name = ?
        """, [region_name]).fetchdf()

        if facilities.empty:
            continue

        fids = set(facilities['facility_id'].astype(int))
        fid_pos = {
            int(r['facility_id']): (r['longitude'], r['latitude'])
            for _, r in facilities.iterrows()
        }
        fid_method = {
            int(r['facility_id']): r['method']
            for _, r in facilities.iterrows()
        }

        routes = con.execute("""
            SELECT pr.src, pr.dst, pr.route_id, pr.sequence
            FROM part_of_route pr
            WHERE pr.src IN (SELECT facility_id FROM located_in
                             WHERE region_name = ?)
            ORDER BY pr.route_id, pr.sequence
        """, [region_name]).fetchdf()

        connects = con.execute("""
            SELECT ct.src, ct.dst
            FROM connects_to ct
            WHERE ct.src IN (SELECT facility_id FROM located_in
                             WHERE region_name = ?)
              AND ct.distance_miles <= 150
        """, [region_name]).fetchdf()

        fig, ax = plt.subplots(figsize=(14, 10))

        if alaska is not None:
            alaska.boundary.plot(ax=ax, color='#999999', linewidth=0.5,
                                zorder=0)

        # Background: connects_to as faint arrows
        for _, row in connects.iterrows():
            src, dst = int(row['src']), int(row['dst'])
            if src in fid_pos and dst in fid_pos:
                x0, y0 = fid_pos[src]
                x1, y1 = fid_pos[dst]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>", color='#cccccc',
                                            lw=0.5, mutation_scale=6),
                            zorder=1)

        # Foreground: part_of_route as bold arrows
        for _, row in routes.iterrows():
            src, dst = int(row['src']), int(row['dst'])
            if src in fid_pos and dst in fid_pos:
                x0, y0 = fid_pos[src]
                x1, y1 = fid_pos[dst]
                method = fid_method.get(src, 'Unknown')
                color = method_colors.get(method, '#333333')
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=1.8, mutation_scale=12),
                            zorder=3)

        # Nodes
        for method, color in method_colors.items():
            subset = facilities[facilities['method'] == method]
            if not subset.empty:
                ax.scatter(subset['longitude'], subset['latitude'],
                           c=color, s=40, zorder=5, label=method,
                           edgecolors='white', linewidths=0.3)

        # Zoom to region with padding
        lons = facilities['longitude']
        lats = facilities['latitude']
        pad_lon = max((lons.max() - lons.min()) * 0.15, 0.5)
        pad_lat = max((lats.max() - lats.min()) * 0.15, 0.3)
        ax.set_xlim(lons.min() - pad_lon, lons.max() + pad_lon)
        ax.set_ylim(lats.min() - pad_lat, lats.max() + pad_lat)

        n_routes = routes['route_id'].nunique() if not routes.empty else 0
        ax.set_title(f"{region_name}\n"
                     f"{len(facilities)} facilities, {n_routes} routes",
                     fontsize=13)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(loc='best', fontsize=8, title='Delivery Method')

        plt.tight_layout()
        safe_name = region_name.replace(' ', '_').replace('/', '_')
        path = f"outputs/regional_directed_{safe_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")

    print(f"Regional directed maps: {len(regions)} regions saved.")


# ===========================================================================
# Overall Graph with Gradient Edges
# ===========================================================================

def _make_gradient_line(x0, y0, x1, y1, cmap, n_segments=50, lw=1.5,
                        alpha=0.8):
    """Create a LineCollection with a color gradient from (x0,y0) to (x1,y1).

    Returns a matplotlib.collections.LineCollection.
    """
    x = np.linspace(x0, x1, n_segments + 1)
    y = np.linspace(y0, y1, n_segments + 1)
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = cmap(np.linspace(0, 1, n_segments))
    lc = mcoll.LineCollection(segments, colors=colors, linewidths=lw,
                              alpha=alpha)
    return lc


def visualize_graph_gradient(con):
    """Visualize the full graph database with gradient-colored directed edges.

    Each edge is drawn as a color gradient from dark (source) to light
    (destination), making flow direction visible without arrows.

    - connects_to edges: faint gray gradient (background context)
    - part_of_route edges: bold method-colored gradient (optimized routes)
    - Nodes colored by region

    Saves to outputs/graph_gradient.png
    """
    os.makedirs("outputs", exist_ok=True)

    alaska = _load_alaska_boundary()

    facilities_df = con.execute("""
        SELECT f.facility_id, f.longitude, f.latitude,
               li.region_name,
               COALESCE(um.method_name, 'Unknown') AS method
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        WHERE li.region_name != 'Unassigned'
    """).fetchdf()

    adjacency_df = con.execute("""
        SELECT src, dst, distance_miles
        FROM connects_to
        WHERE distance_miles <= 100
    """).fetchdf()

    routes_df = con.execute("""
        SELECT pr.src, pr.dst, pr.route_id, pr.sequence
        FROM part_of_route pr
        ORDER BY pr.route_id, pr.sequence
    """).fetchdf()

    fid_pos = {}
    fid_method = {}
    region_palette = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9A6324', '#808000',
    ]
    region_colors = {}
    node_colors = {}

    for _, row in facilities_df.iterrows():
        fid = int(row['facility_id'])
        region = row['region_name']
        fid_pos[fid] = (row['longitude'], row['latitude'])
        fid_method[fid] = row['method']

        if region not in region_colors:
            region_colors[region] = region_palette[
                len(region_colors) % len(region_palette)]
        node_colors[fid] = region_colors[region]

    # Gradient colormaps per delivery method (dark -> light)
    method_cmaps = {
        'Road': mcolors.LinearSegmentedColormap.from_list(
            'road', ['#8B4513', '#FFD700']),
        'Plane': mcolors.LinearSegmentedColormap.from_list(
            'plane', ['#00008B', '#87CEEB']),
        'Barge': mcolors.LinearSegmentedColormap.from_list(
            'barge', ['#006400', '#90EE90']),
        'Unknown': mcolors.LinearSegmentedColormap.from_list(
            'unknown', ['#333333', '#CCCCCC']),
    }
    gray_cmap = mcolors.LinearSegmentedColormap.from_list(
        'gray_grad', ['#AAAAAA', '#EEEEEE'])

    fig, ax = plt.subplots(figsize=(18, 14))

    if alaska is not None:
        alaska.boundary.plot(ax=ax, color='#999999', linewidth=0.5, zorder=0)

    # Background: adjacency edges as faint gradients
    for _, row in adjacency_df.iterrows():
        src, dst = int(row['src']), int(row['dst'])
        if src in fid_pos and dst in fid_pos:
            x0, y0 = fid_pos[src]
            x1, y1 = fid_pos[dst]
            lc = _make_gradient_line(x0, y0, x1, y1, gray_cmap,
                                     n_segments=20, lw=0.4, alpha=0.15)
            ax.add_collection(lc)

    # Foreground: route edges as bold method-colored gradients
    for _, row in routes_df.iterrows():
        src, dst = int(row['src']), int(row['dst'])
        if src in fid_pos and dst in fid_pos:
            x0, y0 = fid_pos[src]
            x1, y1 = fid_pos[dst]
            method = fid_method.get(src, 'Unknown')
            cmap = method_cmaps.get(method, method_cmaps['Unknown'])
            lc = _make_gradient_line(x0, y0, x1, y1, cmap,
                                     n_segments=40, lw=2.0, alpha=0.85)
            ax.add_collection(lc)

    # Nodes
    for fid in fid_pos:
        x, y = fid_pos[fid]
        ax.scatter(x, y, c=node_colors.get(fid, '#333333'), s=25,
                   zorder=5, edgecolors='white', linewidths=0.3)

    # Legend: regions
    for region, color in sorted(region_colors.items()):
        ax.scatter([], [], c=color, label=f"{region}", s=50)

    # Legend: delivery method gradients
    for method, cmap in method_cmaps.items():
        if method == 'Unknown':
            continue
        dark = cmap(0.0)
        light = cmap(1.0)
        ax.plot([], [], color=dark, linewidth=3,
                label=f"{method} (dark=src)")
        ax.plot([], [], color=light, linewidth=3,
                label=f"{method} (light=dst)")

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8,
              title='Regions & Routes')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Graph Database: Directed Routes with Gradient Flow\n'
                 '(dark = source, light = destination)',
                 fontsize=14)
    ax.autoscale_view()
    plt.tight_layout()
    plt.savefig("outputs/graph_gradient.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Graph gradient visualization saved: outputs/graph_gradient.png")
    print(f"  Nodes: {len(fid_pos)}")
    print(f"  Adjacency edges: {len(adjacency_df)}")
    print(f"  Route edges: {len(routes_df)}")


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


@tool("query_friction_costs")
def query_friction_costs() -> str:
    """Query friction-based delivery costs and terrain data from the graph
    database.  Returns cost, friction, and path data for all connects_to
    edges, grouped by region and delivery method.

    Data comes from the friction surface computation pipeline which
    computed least-cost paths through real terrain (slope, land cover,
    permafrost, road networks, rivers).
    """
    global graph_con
    result = graph_con.execute("""
        SELECT
            li.region_name AS region,
            COALESCE(um.method_name, 'Unknown') AS delivery_method,
            COUNT(*) AS edge_count,
            ROUND(AVG(ct.avg_friction), 3) AS mean_friction,
            ROUND(MAX(ct.avg_friction), 3) AS max_friction,
            ROUND(AVG(ct.path_length_miles), 1) AS mean_path_miles,
            ROUND(AVG(ct.distance_miles), 1) AS mean_haversine_miles,
            ROUND(AVG(ct.delivery_cost), 2) AS mean_delivery_cost,
            ROUND(MIN(ct.delivery_cost), 2) AS min_delivery_cost,
            ROUND(MAX(ct.delivery_cost), 2) AS max_delivery_cost,
            ROUND(AVG(ct.cost_summer), 2) AS mean_cost_summer,
            ROUND(AVG(ct.cost_shoulder), 2) AS mean_cost_shoulder,
            ROUND(AVG(ct.cost_winter), 2) AS mean_cost_winter,
            SUM(CASE WHEN ct.friction_winter >= 999 THEN 1 ELSE 0 END)
                AS winter_impassable_count
        FROM connects_to ct
        JOIN facilities f ON ct.src = f.facility_id
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        WHERE ct.avg_friction IS NOT NULL
        GROUP BY li.region_name, um.method_name
        ORDER BY li.region_name, um.method_name
    """).fetchdf()

    return result.to_json(orient='records', indent=2)


@tool("save_report")
def save_report(json_report: str) -> str:
    """Save the final TSP analysis report as JSON.

    Args:
        json_report: The complete report as a JSON string.
    """
    with open('tsp_final_report.json', 'w', encoding='utf-8') as f:
        f.write(json_report)
    return "Report saved to tsp_final_report.json"


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

def setup_agents(llm_haiku, llm_sonnet, tsp_results_dict, input_report):
    """Create all agents and tasks for the TSP analysis.

    Args:
        llm_haiku:  CrewAI LLM instance for fast/cheap agents
                    (TSP Route Optimizer, Operational Risk).
        llm_sonnet: CrewAI LLM instance for reasoning/writing agents
                    (Route Analyzer, TSP Route Adjuster, Writer, Contrarian).
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
        llm=llm_haiku,
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

        The TSP results summary: {json.dumps(tsp_results_dict, indent=2)}

        Note: Delivery method assignment was already handled in the
        regionalization step. Do not reassign or recommend delivery methods.

        Present the route data in a clear, structured format. Report the
        facts — leave analysis and recommendations to other agents.""",
        agent=tsp_agent,
        expected_output="A structured summary of all computed routes with "
                       "distances, segment counts, and facility details "
                       "from the graph database."
    )

    # ----- Agent 2: Operational Risk Agent -----
    operational_risk_agent = Agent(
        role="Operational Risk Analyst",
        goal="Analyze fuel delivery segment risks by region using friction-"
             "based costs and terrain data from the graph database, combined "
             "with the market cost analysis report.",
        backstory="Expert in logistics risk assessment with deep knowledge "
                  "of Arctic operations and supply chain vulnerabilities. "
                  "You ground your analysis in real delivery cost data "
                  "computed from terrain friction surfaces (slope, land cover, "
                  "permafrost, road networks, rivers) stored in the graph "
                  "database.",
        verbose=True,
        llm=llm_haiku,
        tools=[query_friction_costs],
    )

    operational_risk_task = Task(
        description=f"""Assess the risk for each fuel delivery segment.

        Use the query_friction_costs tool to retrieve friction-based delivery
        costs and terrain data from the graph database. These costs were
        computed from real terrain data (slope, land cover, permafrost, road
        networks, rivers) using least-cost path analysis.

        Also reference the market analysis: {str(input_report)}

        For each region/delivery method, evaluate:
        - **Terrain difficulty:** Use avg_friction and max_friction from the
          graph. Higher friction = harder terrain (steep slopes, permafrost,
          poor road surface). Values near 1.0 are ideal; above 2.0 is
          challenging.
        - **Seasonal accessibility:** Check cost_summer vs cost_shoulder vs
          cost_winter. Routes with winter_impassable_count > 0 are shut down
          in winter (frozen rivers/sea ice for barge routes).
        - **Weather:** Storm, ice, fog impact on segment feasibility
        - **Delivery method risks:** Road conditions (permafrost damage),
          ice for barges (seasonal), visibility for air
        - **Cost indicators:** High delivery_cost relative to path_length
          indicates difficult terrain. Compare mean_delivery_cost across
          methods within a region.

        Classify each route into risk categories:
        - High risk: Significant potential for delays/failure
        - Moderate risk: Some risks but manageable
        - Low risk: Minimal anticipated risks

        Suggest alternatives for high-risk segments.""",
        agent=operational_risk_agent,
        context=[tsp_task],
        expected_output="A detailed risk analysis for each route/segment, "
                       "grounded in friction-based terrain and cost data."
    )

    # ----- Agent 3: Route Analyzer -----
    route_analyzer_agent = Agent(
        role="Route Analyzer",
        goal="Assess whether computed delivery routes are practical, "
             "cost-effective, and operationally sound by synthesizing "
             "route data, friction-based costs from the graph database, "
             "and risk assessments.",
        backstory="Expert in logistics network analysis with deep knowledge "
                  "of Alaska's geography and fuel delivery constraints. "
                  "You evaluate route quality by combining route data with "
                  "friction-based delivery costs and terrain data stored in "
                  "the graph database. You identify routes that are too "
                  "long, too costly, or ineffective and recommend improvements.",
        verbose=True,
        llm=llm_sonnet,
        tools=[query_friction_costs],
    )

    route_analysis_task = Task(
        description="""Analyze the computed delivery routes using:
        - Route data from the TSP Route Optimizer
        - Friction-based delivery costs from the graph database (use the
          query_friction_costs tool)
        - Risk assessments from the Operational Risk Analyst

        The friction costs were computed from real terrain: least-cost paths
        through slope, land cover, permafrost, road networks, and rivers.
        Higher avg_friction means harder terrain. The delivery_cost field
        is WAF × path_length_miles × baseline rate per delivery method.

        For each region/delivery method route, assess:
        1. **Efficiency:** Are there unusually long segments? Compare
           path_length_miles (friction path) to distance_miles (Haversine).
           High detour ratios indicate terrain forcing long diversions.
        2. **Cost-effectiveness:** Which routes have the highest
           mean_delivery_cost? Compare costs across methods within a region.
        3. **Seasonal viability:** Check winter_impassable_count — routes
           with many impassable winter edges need alternative seasonal plans.
        4. **Operational viability:** Cross-reference with risk assessments.
           Are high-cost routes also high-risk?
        5. **Recommendations:** Identify routes that might benefit from
           splitting into sub-routes or alternative groupings.

        Note: Delivery method assignment was handled in regionalization.
        Do not reassign delivery methods — focus on route quality.

        Provide a clear assessment for each route with actionable
        recommendations.""",
        agent=route_analyzer_agent,
        context=[tsp_task, operational_risk_task],
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
        llm=llm_sonnet,
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
        llm=llm_sonnet,
        tools=[save_report]
    )

    multi_agent_discussion_task = Task(
        description="""Lead a discussion synthesizing findings from the
        TSP Route Optimizer, Route Analyzer, TSP Route Adjuster, and
        Operational Risk Agent.

        * TSP Route Optimizer: Route data from the graph database
        * Route Analyzer: Route efficiency, cost-effectiveness, and
          viability assessments with recommendations
        * TSP Route Adjuster: Route modifications made and their impact
        * Operational Risk Agent: Risk assessments grounded in friction-
          based terrain and cost data from the graph database

        Note: Delivery costs are computed from the friction surface
        pipeline (terrain-aware least-cost paths) and stored directly
        in the graph database — not estimated by an LLM agent.

        As moderator:
        1. Synthesize route data, analysis, adjustments, and risk
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
        llm=llm_sonnet
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

    _regions_list = list(tsp_results_dict.keys())
    _regions_bullet = "\n    ".join(f"- {r}" for r in _regions_list)
    _report_desc = (
        """Produce a comprehensive final report integrating all agent
        analyses, discussions, and critiques.

        IMPORTANT: You MUST cover ALL of the following regions individually:
    """ + _regions_bullet + """

        The report should be a single JSON document. Use the save_report
        tool to save it. The JSON must follow this structure:

        {
        "title": "Alaska Fuel Delivery TSP Route Optimization Report",
        "date_generated": \"""" + date.today().isoformat() + """\",
        "executive_summary": "2-3 paragraphs summarizing key findings across ALL regions",
        "market_dynamics": "Current market trends, fuel prices, and demand drivers relevant to route optimization",
        "environmental_concerns": "Climate, weather, and seasonal constraints affecting delivery routes across Alaska",
        "regional_route_analysis": {
            "<region_name>": {
                "optimized_routes": "Description of optimized route paths, distances, and number of facilities",
                "delivery_methods": "Methods used in this region and rationale",
                "cost_analysis": "Estimated costs and cost drivers for this region",
                "risk_assessment": "Key risks and mitigation strategies for this region",
                "key_findings": "Notable observations for this region"
            }
        },
        "recommendations": {
            "strategic_priorities": [
                {"priority": "Name", "description": "Details",
                 "implementation_steps": ["Step 1", "Step 2"],
                 "expected_impact": "Outcome"}
            ],
            "operational_improvements": [
                {"improvement": "Name", "description": "Details",
                 "expected_impact": "Outcome"}
            ]
        },
        "agent_discussion_summary": {
            "overview": "How the multi-agent discussion shaped the analysis",
            "key_insights": ["Insight 1", "Insight 2"],
            "impact_on_recommendations": "How discussion influenced final recommendations"
        },
        "limitations": {
            "contrarian_critique": "Summary of contrarian critique",
            "response_to_critique": "How concerns were addressed",
            "acknowledged_limitations": ["Limitation 1"],
            "areas_for_further_research": ["Area 1"]
        },
        "metadata": {
            "regions_covered": """ + json.dumps(_regions_list) + """,
            "agents_involved": ["TSP Route Optimizer",
                "Operational Risk Agent", "Route Analyzer",
                "Writing Agent", "Contrarian Agent"],
            "data_sources": ["regionalization.duckdb graph database",
                "market_cost_analysis_report.json",
                "friction_analysis_report.json"],
            "confidence_level": "High/Medium/Low"
        }
        }

        You MUST include a separate entry in "regional_route_analysis" for
        EACH region listed above. Do not skip or combine regions."""
    )
    writing_task = Task(
        description=_report_desc,
        agent=writing_agent,
        expected_output="A comprehensive JSON report saved to "
                       "tsp_final_report.json via the save_report tool."
    )

    agents = [tsp_agent, operational_risk_agent, route_analyzer_agent,
              tsp_adjuster_agent, writing_agent, contrarian_agent]

    tasks = [
        tsp_task,                      # Phase 1: Route data retrieval (data-grounded)
        operational_risk_task,         # Phase 1: Risk assessment (uses friction costs from graph)
        route_analysis_task,           # Phase 2: Route analysis (uses friction costs + risk)
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

    # LLM setup (per-agent tiers via OpenRouter by default)
    llm_haiku = pipeline.get_llm("haiku")
    llm_sonnet = pipeline.get_llm("sonnet")

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
    tsp_results_dict = serialize_tours(regional_results)

    # Set up agents and tasks
    agents, tasks = setup_agents(llm_haiku, llm_sonnet, tsp_results_dict, input_report)

    # Configure crew (agent-level llms take precedence; this is just a fallback)
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        llm=llm_haiku
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

    # Save report (fallback if the agent didn't use the save_report tool)
    if not os.path.exists('tsp_final_report.json'):
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
