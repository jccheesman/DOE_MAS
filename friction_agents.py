# -*- coding: utf-8 -*-
"""friction_agents.py

CrewAI agents for friction surface analysis, seasonal adjustment,
cost estimation, and validation.

Agents:
    - Friction Modeler: Orchestrates friction raster assembly and
      WhiteboxTools least-cost path computation for all connects_to edges
    - Seasonal Enhancer: Applies seasonal multipliers to friction values
    - Cost Estimator: Computes delivery costs from friction, distance, and
      baseline rates
    - Validation Agent: Validates costs against ISER / AEA benchmarks

Input data:
    - regionalization.duckdb (graph database with connects_to edges)
    - GEE-exported rasters in RASTER_DIR
    - Vector layers (airports, ports, facilities) in VECTOR_DIR

Outputs:
    - friction_analysis_report.json
    - Updated connects_to edges with friction, seasonal, and cost columns
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import json
import logging
import warnings
from datetime import date

import duckdb

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=DeprecationWarning)

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import pipeline
import friction_config

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
pipeline.set_cwd('/media/volume/GraphDB_Runs')

# ---------------------------------------------------------------------------
# DuckDB Graph Database Connection
# ---------------------------------------------------------------------------
graph_con = None


def connect_graph_db(db_path='regionalization.duckdb'):
    """Open a read-write connection to the shared DuckDB graph database."""
    global graph_con
    graph_con = duckdb.connect(db_path, read_only=False)
    print(f"Connected to graph database: {db_path} (read-write)")
    return graph_con


# ===========================================================================
# Tools — Friction Modeler
# ===========================================================================

@tool("run_friction_computation")
def run_friction_computation() -> str:
    """Trigger the full friction surface computation pipeline.

    Builds composite friction rasters (friction_road.tif, friction_barge.tif)
    from GEE-exported layers, then computes least-cost paths for Road and
    Barge edges using WhiteboxTools cost_distance. Plane edges use direct
    Haversine distance (airport-to-airport, no friction raster).

    Updates the graph database with avg_friction, max_friction, and
    path_length_miles on each connects_to edge.

    Returns:
        str: Summary of the friction computation results.
    """
    import friction_surface
    global graph_con

    friction_surface.main(con=graph_con)

    # Query summary statistics
    stats = graph_con.execute("""
        SELECT
            COUNT(*) AS total_edges,
            COUNT(avg_friction) AS edges_with_friction,
            ROUND(AVG(avg_friction), 3) AS mean_avg_friction,
            ROUND(MIN(avg_friction), 3) AS min_avg_friction,
            ROUND(MAX(avg_friction), 3) AS max_avg_friction,
            ROUND(AVG(path_length_miles), 1) AS mean_path_miles,
            ROUND(MAX(path_length_miles), 1) AS max_path_miles
        FROM connects_to
    """).fetchdf()

    return (
        f"Friction computation complete.\n"
        f"Summary:\n{stats.to_string(index=False)}"
    )


@tool("query_friction_stats")
def query_friction_stats() -> str:
    """Query friction statistics from the graph database, grouped by
    delivery method and region.

    Returns:
        str: JSON summary of friction statistics per region and method.
    """
    global graph_con
    result = graph_con.execute("""
        SELECT
            li.region_name AS region,
            COALESCE(um.method_name, 'Unknown') AS delivery_method,
            COUNT(*) AS edge_count,
            ROUND(AVG(ct.avg_friction), 3) AS mean_friction,
            ROUND(MIN(ct.avg_friction), 3) AS min_friction,
            ROUND(MAX(ct.avg_friction), 3) AS max_friction,
            ROUND(AVG(ct.path_length_miles), 1) AS mean_path_miles,
            ROUND(AVG(ct.distance_miles), 1) AS mean_haversine_miles,
            ROUND(
                AVG(ct.path_length_miles) / NULLIF(AVG(ct.distance_miles), 0),
                2
            ) AS path_detour_ratio
        FROM connects_to ct
        JOIN facilities f ON ct.src = f.facility_id
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        WHERE ct.avg_friction IS NOT NULL
        GROUP BY li.region_name, um.method_name
        ORDER BY li.region_name, um.method_name
    """).fetchdf()

    return result.to_json(orient='records', indent=2)


# ===========================================================================
# Tools — Seasonal Enhancer
# ===========================================================================

def _get_seasonal_mult(region, feature, season):
    """Look up seasonal multiplier with regional override fallback."""
    override = friction_config.REGIONAL_SEASONAL_OVERRIDES.get(
        (region, feature, season))
    if override is not None:
        return override
    return friction_config.SEASONAL_MULTIPLIERS.get(
        (feature, season), 1.0)

@tool("apply_seasonal_multipliers")
def apply_seasonal_multipliers() -> str:
    """Apply seasonal multipliers to avg_friction for all connects_to edges.

    For each edge, determines the appropriate seasonal multiplier based on
    the delivery method and geographic context (river, coastal, etc.).
    Writes friction_summer, friction_shoulder, and friction_winter to
    the connects_to edges.

    Returns:
        str: Summary of seasonal friction values written.
    """
    global graph_con

    # Read all edges with their delivery method, friction, and region
    edges = graph_con.execute("""
        SELECT ct.src, ct.dst, ct.avg_friction,
               COALESCE(um.method_name, 'Road') AS method,
               li.region_name
        FROM connects_to ct
        JOIN uses_method um ON ct.src = um.facility_id
        JOIN located_in li ON ct.src = li.facility_id
        WHERE ct.avg_friction IS NOT NULL
    """).fetchall()

    updated = 0
    skipped_barge = 0
    for src, dst, avg_friction, method, region in edges:
        if avg_friction is None:
            continue

        # Barge: seasonal friction already populated from per-pixel surfaces
        if method == 'Barge':
            skipped_barge += 1
            continue
        elif method == 'Plane':
            summer_mult = 1.0
            shoulder_mult = 1.0
            winter_mult = 1.0
        else:
            summer_mult = 1.0
            shoulder_mult = 1.1
            winter_mult = 1.3

        friction_summer = avg_friction * summer_mult
        friction_shoulder = avg_friction * shoulder_mult
        friction_winter = (
            friction_config.IMPASSABLE
            if winter_mult >= friction_config.IMPASSABLE
            else avg_friction * winter_mult
        )

        graph_con.execute("""
            UPDATE connects_to
            SET friction_summer = ?,
                friction_shoulder = ?,
                friction_winter = ?
            WHERE src = ? AND dst = ?
        """, [friction_summer, friction_shoulder, friction_winter, src, dst])
        updated += 1

    # Query summary
    summary = graph_con.execute("""
        SELECT
            COUNT(*) AS edges_updated,
            ROUND(AVG(friction_summer), 3) AS mean_summer,
            ROUND(AVG(friction_shoulder), 3) AS mean_shoulder,
            ROUND(AVG(friction_winter), 3) AS mean_winter
        FROM connects_to
        WHERE friction_summer IS NOT NULL
    """).fetchdf()

    return (
        f"Seasonal multipliers applied to {updated} edges "
        f"({skipped_barge} Barge edges already have per-pixel seasonal friction).\n"
        f"Summary:\n{summary.to_string(index=False)}"
    )


@tool("query_seasonal_friction")
def query_seasonal_friction() -> str:
    """Query seasonal friction values from the graph database, grouped
    by delivery method and region.

    Returns:
        str: JSON summary of seasonal friction per region and method.
    """
    global graph_con
    result = graph_con.execute("""
        SELECT
            li.region_name AS region,
            COALESCE(um.method_name, 'Unknown') AS delivery_method,
            COUNT(*) AS edge_count,
            ROUND(AVG(ct.friction_summer), 3) AS mean_summer,
            ROUND(AVG(ct.friction_shoulder), 3) AS mean_shoulder,
            ROUND(AVG(ct.friction_winter), 3) AS mean_winter
        FROM connects_to ct
        JOIN facilities f ON ct.src = f.facility_id
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        WHERE ct.friction_summer IS NOT NULL
        GROUP BY li.region_name, um.method_name
        ORDER BY li.region_name, um.method_name
    """).fetchdf()

    return result.to_json(orient='records', indent=2)


# ===========================================================================
# Tools — Cost Estimator
# ===========================================================================

@tool("compute_delivery_costs")
def compute_delivery_costs() -> str:
    """Compute delivery costs for all connects_to edges using the formula:

        DeliveryCost = WAF_seasonal * path_length_miles * BaselineRate

    Reads seasonal friction + path_length from the graph. Writes
    delivery_cost and seasonal cost variants (cost_summer, cost_shoulder,
    cost_winter) back to each edge. Also computes directional costs
    (cost_fwd, cost_rev).

    Returns:
        str: Summary of computed delivery costs.
    """
    global graph_con

    edges = graph_con.execute("""
        SELECT ct.src, ct.dst, ct.path_length_miles,
               ct.friction_summer, ct.friction_shoulder, ct.friction_winter,
               COALESCE(um.method_name, 'Road') AS method
        FROM connects_to ct
        JOIN uses_method um ON ct.src = um.facility_id
        WHERE ct.path_length_miles IS NOT NULL
          AND ct.friction_summer IS NOT NULL
    """).fetchall()

    updated = 0
    for src, dst, path_miles, f_summer, f_shoulder, f_winter, method in edges:
        rate = friction_config.BASELINE_RATES.get(method, 3.5)

        cost_summer = f_summer * path_miles * rate
        cost_shoulder = f_shoulder * path_miles * rate
        if f_winter is None or f_winter >= friction_config.IMPASSABLE:
            if f_winter is not None and f_winter >= friction_config.IMPASSABLE:
                logger.warning(
                    "Edge %s->%s has f_winter=%s (finite IMPASSABLE). "
                    "Expected NULL under nodata-barrier semantics.",
                    src, dst, f_winter)
            cost_winter = None
        else:
            cost_winter = f_winter * path_miles * rate

        # Base delivery cost uses summer (primary delivery season)
        delivery_cost = cost_summer

        # Directional costs: fwd uses avg_friction, rev may differ slightly
        # For now, fwd=cost_summer (src->dst), rev=cost_summer (dst->src)
        # These become asymmetric once terrain directionality is computed
        cost_fwd = cost_summer
        cost_rev = cost_summer

        graph_con.execute("""
            UPDATE connects_to
            SET delivery_cost = ?,
                cost_summer = ?,
                cost_shoulder = ?,
                cost_winter = ?,
                cost_fwd = ?,
                cost_rev = ?
            WHERE src = ? AND dst = ?
        """, [delivery_cost, cost_summer, cost_shoulder, cost_winter,
              cost_fwd, cost_rev, src, dst])
        updated += 1

    summary = graph_con.execute("""
        SELECT
            COUNT(*) AS edges_with_cost,
            ROUND(AVG(delivery_cost), 2) AS mean_cost,
            ROUND(MIN(delivery_cost), 2) AS min_cost,
            ROUND(MAX(delivery_cost), 2) AS max_cost,
            ROUND(SUM(delivery_cost), 2) AS total_cost
        FROM connects_to
        WHERE delivery_cost IS NOT NULL
    """).fetchdf()

    return (
        f"Delivery costs computed for {updated} edges.\n"
        f"Summary:\n{summary.to_string(index=False)}"
    )


@tool("query_delivery_costs")
def query_delivery_costs() -> str:
    """Query delivery cost statistics from the graph, grouped by region
    and delivery method.

    Returns:
        str: JSON summary of delivery costs per region and method.
    """
    global graph_con
    result = graph_con.execute("""
        SELECT
            li.region_name AS region,
            COALESCE(um.method_name, 'Unknown') AS delivery_method,
            COUNT(*) AS edge_count,
            ROUND(AVG(ct.delivery_cost), 2) AS mean_cost,
            ROUND(MIN(ct.delivery_cost), 2) AS min_cost,
            ROUND(MAX(ct.delivery_cost), 2) AS max_cost,
            ROUND(AVG(ct.path_length_miles), 1) AS mean_path_miles,
            ROUND(AVG(ct.avg_friction), 3) AS mean_friction
        FROM connects_to ct
        JOIN facilities f ON ct.src = f.facility_id
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        WHERE ct.delivery_cost IS NOT NULL
        GROUP BY li.region_name, um.method_name
        ORDER BY li.region_name, um.method_name
    """).fetchdf()

    return result.to_json(orient='records', indent=2)


# ===========================================================================
# Tools — Validation Agent
# ===========================================================================

@tool("validate_costs_against_benchmarks")
def validate_costs_against_benchmarks() -> str:
    """Compare computed delivery costs against Alaska fuel price benchmarks
    from AEDG, DCRA Fuel Price Reports, and ISER/AEA studies.

    Checks cost reasonableness by region and delivery method, and derives
    calibration multipliers where computed costs deviate from benchmarks.

    Returns:
        str: JSON report with benchmark comparisons and calibration factors.
    """
    global graph_con

    # Query computed costs by region and method
    computed = graph_con.execute("""
        SELECT
            li.region_name AS region,
            COALESCE(um.method_name, 'Road') AS delivery_method,
            COUNT(*) AS edge_count,
            ROUND(AVG(ct.delivery_cost / NULLIF(ct.path_length_miles, 0)), 2)
                AS computed_cost_per_mile,
            ROUND(AVG(ct.path_length_miles), 1) AS mean_path_miles
        FROM connects_to ct
        JOIN facilities f ON ct.src = f.facility_id
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        WHERE ct.delivery_cost IS NOT NULL
        GROUP BY li.region_name, um.method_name
        ORDER BY li.region_name, um.method_name
    """).fetchdf()

    # Benchmark ranges ($/mile) from multiple Alaska energy cost sources:
    #   - Alaska Energy Data Gateway (AEDG): akenergygateway.alaska.edu
    #     Community-level fuel prices from ISER/UAA
    #   - DCRA Alaska Fuel Price Reports (semi-annual surveys of ~100
    #     communities): storymaps.arcgis.com/stories/6d6a33a3d9a74723a2f476c26ecfdf21
    #   - ISER / AEA published energy cost studies
    benchmarks = {
        "Road":  {"low": 2.0, "mid": 3.5, "high": 5.0},
        "Barge": {"low": 1.0, "mid": 2.0, "high": 3.0},
        "Plane": {"low": 8.0, "mid": 11.5, "high": 15.0},
    }

    validation_results = []
    for _, row in computed.iterrows():
        method = row['delivery_method']
        cpm = row['computed_cost_per_mile']
        bench = benchmarks.get(method, benchmarks["Road"])

        if cpm is not None:
            calibration = bench["mid"] / cpm if cpm > 0 else None
            status = "OK"
            if cpm < bench["low"] * 0.7:
                status = "LOW - may need upward calibration"
            elif cpm > bench["high"] * 1.3:
                status = "HIGH - may need downward calibration"
        else:
            calibration = None
            status = "NO DATA"

        validation_results.append({
            "region": row['region'],
            "method": method,
            "computed_cost_per_mile": float(cpm) if cpm else None,
            "benchmark_range": bench,
            "calibration_multiplier": round(calibration, 3) if calibration else None,
            "status": status,
        })

    return json.dumps(validation_results, indent=2)


@tool("save_friction_report")
def save_friction_report(report_json: str) -> str:
    """Save the friction analysis report to a JSON file.

    Args:
        report_json: The report content as a JSON string.

    Returns:
        str: Confirmation message with file path.
    """
    output_path = "friction_analysis_report.json"
    try:
        parsed = json.loads(report_json)
        content = parsed
    except (json.JSONDecodeError, TypeError):
        content = {"report": report_json, "date_generated": str(date.today())}

    with open(output_path, 'w') as f:
        json.dump(content, f, indent=4)

    return f"Friction analysis report saved to {output_path}"


# ===========================================================================
# Agent & Task Setup
# ===========================================================================

def setup_agents(llm_haiku, llm_sonnet):
    """Create and configure all friction-related CrewAI agents and tasks.

    Args:
        llm_haiku:  CrewAI LLM instance for the Cost Estimator agent.
        llm_sonnet: CrewAI LLM instance for the Friction Modeler, Seasonal
                    Enhancer, and Validation agents.

    Returns:
        tuple: (agents_list, tasks_list) for Crew initialization
    """

    # --- Agent 1: Friction Modeler ---
    friction_modeler = Agent(
        role="Friction Modeler",
        goal=(
            "Derive friction values for all pre-existing connections stored "
            "in the 'connects_to' edge of the graph database. Assemble "
            "composite friction rasters from GEE-exported layers, run "
            "WhiteboxTools cost_distance and cost_pathway for each facility "
            "pair, and write avg_friction, max_friction, and path_length_miles "
            "to the graph."
        ),
        backstory=(
            "You are a high-level geospatial analyst based in Alaska with "
            "advanced knowledge of creating friction surfaces on complex "
            "terrain. You specialize in integrating multiple geospatial data "
            "layers (slope, land cover, permafrost, road networks, rivers) "
            "into composite cost surfaces using WhiteboxTools. Your job is "
            "to take raw rasters and produce a single friction number per "
            "edge in the graph database."
        ),
        verbose=True,
        llm=llm_sonnet,
        tools=[run_friction_computation, query_friction_stats],
    )

    friction_task = Task(
        description=(
            "Execute the friction surface computation pipeline:\n\n"
            "1. Use the run_friction_computation tool. It builds two "
            "friction rasters and handles plane routes separately:\n"
            "   - friction_road.tif: for Road delivery (land traversal)\n"
            "   - friction_barge.tif: for Barge delivery (water navigation)\n"
            "   - Plane delivery: uses direct Haversine distance between "
            "facilities (airport-to-airport, no terrain friction raster)\n"
            "   The methodology for assembling the rasters:\n"
            "   a) Reclassify slope into friction: flat (<2°)=1.0, "
            "rolling (2-8°)=1.4, mountain (>8°)=1.75.\n"
            "   b) Combine Dynamic World land cover with Pastick et al. "
            "permafrost zonation via a lookup matrix — e.g. grass on "
            "continuous permafrost=1.88, trees on sporadic=1.87.\n"
            "   c) Per pixel, take max(slope friction, LULC-permafrost "
            "friction) as the base ground friction.\n"
            "   d) Where roads exist (GRIP4 binary presence), apply the "
            "uniform ROAD_PRESENT_FRICTION value (1.0) to override the "
            "underlying terrain friction.\n"
            "   e) Set rivers and water to 999 (impassable) for road; "
            "navigable (1.0) for barge.\n"
            "   f) Run WhiteboxTools cost_distance from each source "
            "facility to produce accumulated cost surfaces, then trace "
            "least-cost paths via backlink rasters.\n"
            "   g) Sample friction along each traced path to compute "
            "weighted average friction (WAF) and path length in miles.\n"
            "   h) Plane edges are set to avg_friction=1.0 and "
            "path_length_miles=Haversine distance (no terrain penalty).\n\n"
            "2. Once complete, use query_friction_stats to review the "
            "results by region and delivery method.\n\n"
            "3. Report a summary table of all criteria (slope, land cover, "
            "road type, port access, etc.), conditions, and the "
            "corresponding friction weights used.\n\n"
            "4. Flag any regions or methods with unexpectedly high or low "
            "friction values."
        ),
        expected_output=(
            "A detailed summary including:\n"
            "- Table of friction criteria, conditions, and weights\n"
            "- Statistics: edge count, mean/min/max friction by region and method\n"
            "- Path detour ratios (friction path vs Haversine straight line)\n"
            "- Any flagged anomalies or regions needing attention"
        ),
        agent=friction_modeler,
    )

    # --- Agent 2: Seasonal Enhancer ---
    seasonal_enhancer = Agent(
        role="Seasonal Enhancer",
        goal=(
            "Determine the seasonal friction values for each route in the "
            "connects_to edges. Read avg_friction and the edge's delivery "
            "method from the graph, apply appropriate seasonal multipliers, "
            "and write friction_summer, friction_shoulder, and "
            "friction_winter back to the edges."
        ),
        backstory=(
            "You are an Alaska logistics specialist who understands the "
            "dramatic seasonal variations in transportation accessibility. "
            "You know that river and coastal barge routes freeze shut in "
            "winter, that shoulder seasons (May and October) bring "
            "unpredictable ice conditions, and that road conditions degrade "
            "significantly in winter. You apply calibrated seasonal "
            "multipliers based on the Alaska Historical Sea Ice Atlas and "
            "river freeze/breakup data."
        ),
        verbose=True,
        llm=llm_sonnet,
        tools=[apply_seasonal_multipliers, query_seasonal_friction],
    )

    seasonal_task = Task(
        description=(
            "Apply seasonal friction multipliers to all connects_to edges:\n"
            "1. Use apply_seasonal_multipliers to compute friction_summer, "
            "friction_shoulder, and friction_winter for each edge.\n"
            "2. Seasonal multipliers depend on delivery method:\n"
            "   - Barge: major river summer=1.0, shoulder=1.3, winter=999\n"
            "   - Road: summer=1.0, shoulder=1.1, winter=1.3\n"
            "   - Plane: minimal seasonal variation\n"
            "3. Use query_seasonal_friction to verify results by region.\n"
            "4. Report which regions and methods are most affected by "
            "seasonality, and identify routes that become impassable in winter."
        ),
        expected_output=(
            "A summary including:\n"
            "- Count of edges updated per season\n"
            "- Mean seasonal friction by region and method\n"
            "- List of routes that become impassable (999) in winter\n"
            "- Assessment of seasonal accessibility per region"
        ),
        agent=seasonal_enhancer,
    )

    # --- Agent 3: Cost Estimator ---
    cost_estimator = Agent(
        role="Cost Estimator",
        goal=(
            "Calculate the cost of fuel delivery based on baseline cost "
            "rates in Alaska, seasonal friction values, and route distance. "
            "Formula: DeliveryCost = WAF_seasonal x path_length_miles x "
            "BaselineRate. Write delivery costs back to the graph."
        ),
        backstory=(
            "You are an energy economist specializing in Alaska fuel "
            "delivery costs. You understand that delivery costs in Alaska "
            "are driven by route difficulty (friction), distance, and the "
            "mode of transport. Baseline rates are approximately $3.50/mi "
            "for road, $2.00/mi for barge, and $11.50/mi for plane. You "
            "compute per-edge costs and seasonal variants to support "
            "route optimization by the TSP model."
        ),
        verbose=True,
        llm=llm_haiku,
        tools=[compute_delivery_costs, query_delivery_costs],
    )

    cost_task = Task(
        description=(
            "Compute delivery costs for all connects_to edges:\n"
            "1. Use compute_delivery_costs to calculate costs using the "
            "formula: DeliveryCost = WAF_seasonal * path_length_miles * "
            "BaselineRate. For Plane edges, WAF=1.0 and path_length is "
            "Haversine distance, so cost = distance_miles * $11.50.\n"
            "2. Baseline rates: Road=$3.50/mi, Barge=$2.00/mi, Plane=$11.50/mi\n"
            "3. Write delivery_cost, cost_summer, cost_shoulder, cost_winter, "
            "cost_fwd, and cost_rev to each edge.\n"
            "4. Use query_delivery_costs to review costs by region and method.\n"
            "5. Identify the most expensive routes and the cheapest routes.\n"
            "6. Report total estimated delivery cost across all routes."
        ),
        expected_output=(
            "A cost analysis report including:\n"
            "- Mean/min/max delivery cost by region and method\n"
            "- Top 5 most expensive routes with cost breakdown\n"
            "- Top 5 cheapest routes\n"
            "- Total estimated network delivery cost\n"
            "- Seasonal cost comparison (summer vs shoulder vs winter)"
        ),
        agent=cost_estimator,
    )

    # --- Agent 4: Validation ---
    validation_agent = Agent(
        role="Validation Agent",
        goal=(
            "Validate computed delivery costs against published Alaska "
            "fuel price data from the Alaska Energy Data Gateway (AEDG), "
            "DCRA Fuel Price Reports, and ISER/AEA studies. Derive "
            "regional calibration multipliers where computed costs "
            "deviate from observed community fuel prices."
        ),
        backstory=(
            "You are a research analyst at the Institute of Social and "
            "Economic Research (ISER) who has published extensively on "
            "Alaska energy costs. You cross-reference model outputs "
            "against multiple real-world data sources:\n"
            "- Alaska Energy Data Gateway (akenergygateway.alaska.edu): "
            "community-level fuel prices from ISER/UAA, covering "
            "heating fuel and gasoline across hundreds of communities.\n"
            "- DCRA Alaska Fuel Price Reports: semi-annual surveys of "
            "~100 communities with current and historical fuel prices, "
            "providing per-gallon costs that reflect total delivered "
            "cost including transport, storage, and margins.\n"
            "- ISER/AEA published benchmark studies on Alaska energy "
            "costs by region and delivery method.\n"
            "You understand that per-gallon community prices capture "
            "total delivered cost, so comparing them to per-mile "
            "computed costs requires accounting for route distance and "
            "freight volume. You derive calibration factors to align "
            "the friction-based cost model with observed prices."
        ),
        verbose=True,
        llm=llm_sonnet,
        tools=[
            validate_costs_against_benchmarks,
            query_delivery_costs,
            save_friction_report,
        ],
    )

    validation_task = Task(
        description=(
            "Validate the computed delivery costs against real-world "
            "Alaska fuel price data:\n\n"
            "1. Use validate_costs_against_benchmarks to compare computed "
            "costs against benchmark data from:\n"
            "   - Alaska Energy Data Gateway (akenergygateway.alaska.edu): "
            "community-level heating fuel and gasoline prices\n"
            "   - DCRA Alaska Fuel Price Reports: semi-annual surveys of "
            "~100 communities with per-gallon fuel costs\n"
            "   - ISER/AEA published Alaska energy cost studies\n\n"
            "2. Review the calibration multipliers for each region and method.\n"
            "   Note: community fuel prices are per-gallon (total delivered "
            "   cost) while computed costs are per-mile. To compare, consider "
            "   that price differentials between communities on the same "
            "   delivery route reflect the per-mile transport cost component.\n\n"
            "3. Flag regions where costs deviate more than 30%% from benchmarks.\n"
            "4. Use query_delivery_costs to get detailed cost breakdowns.\n"
            "5. Compile a final friction analysis report with:\n"
            "   - Friction criteria and weights table\n"
            "   - Seasonal adjustment summary\n"
            "   - Cost computation results\n"
            "   - Benchmark validation results citing AEDG and DCRA data\n"
            "   - Recommended calibration multipliers\n"
            f"6. Today's date is {date.today()}.\n"
            "7. Use save_friction_report to save the final report."
        ),
        expected_output=(
            "A comprehensive friction analysis report (JSON) including:\n"
            "- Benchmark comparison table by region and method\n"
            "- Calibration multipliers with confidence assessment\n"
            "- Regions flagged for cost anomalies\n"
            "- Recommendations for improving cost accuracy\n"
            "- The report saved to friction_analysis_report.json"
        ),
        agent=validation_agent,
    )

    agents = [friction_modeler, seasonal_enhancer, cost_estimator, validation_agent]
    tasks = [friction_task, seasonal_task, cost_task, validation_task]

    return agents, tasks


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Run the friction agents pipeline.

    Executes four agents sequentially:
    1. Friction Modeler — builds rasters and computes least-cost paths
    2. Seasonal Enhancer — applies seasonal multipliers
    3. Cost Estimator — computes delivery costs
    4. Validation Agent — validates against benchmarks
    """
    global graph_con

    llm_haiku = pipeline.get_llm("haiku")
    llm_sonnet = pipeline.get_llm("sonnet")
    graph_con = connect_graph_db()

    agents, tasks = setup_agents(llm_haiku, llm_sonnet)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    print("Starting friction agents pipeline...")
    result = crew.kickoff()

    # Save the crew result
    report_path = "friction_analysis_report.json"
    if not os.path.exists(report_path):
        with open(report_path, 'w') as f:
            json.dump(
                {"result": str(result), "date_generated": str(date.today())},
                f,
                indent=4,
            )
        print(f"Friction analysis report saved to {report_path}")

    graph_con.close()
    print("Friction agents pipeline complete.")


if __name__ == "__main__":
    main()
