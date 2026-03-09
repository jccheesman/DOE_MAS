# -*- coding: utf-8 -*-
"""market_cost_analysis.py

Market & Cost Analysis module for the DOE MAS pipeline.

Conducts a broad-scale analysis & overview of the current market dynamics,
cost, delivery methods, and seasonality of fuel delivery in Alaska.

This module replaces broad_overview_agent_discussion.py with:
- DuckDB graph database as the primary data source (no nested dicts)
- Rebranded agents focused on market dynamics, economics, and delivery costs
- Preserved multi-agent discussion structure with contrarian review

Agents:
    - Market Dynamics Analyst: Fuel prices, demand, transportation rates, key drivers
    - Economic & Environmental Factors Agent: Seasonality & infrastructure
    - Delivery Method Analyst: Cost/status of delivery methods + graph DB analysis
    - Writing Agent: Synthesizes findings into a written report
    - Contrarian Agent: Critiques and provides alternative perspectives

Input data:
    - regionalization.duckdb (graph database from regionalization_graph.py)

Outputs:
    - market_cost_analysis_report.json: Comprehensive analysis report
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import json
import warnings
import duckdb

warnings.filterwarnings('ignore', category=DeprecationWarning)

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import pipeline

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
pipeline.set_cwd('/media/volume/Preliminary_mas_runs')

# ---------------------------------------------------------------------------
# DuckDB Graph Database Connection
# ---------------------------------------------------------------------------
graph_con = None


def connect_graph_db(db_path='regionalization.duckdb'):
    """Open a read-only connection to the shared DuckDB graph database.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        duckdb.DuckDBPyConnection
    """
    global graph_con
    graph_con = duckdb.connect(db_path, read_only=True)
    print(f"Connected to graph database: {db_path} (read-only)")
    return graph_con


# ===========================================================================
# Graph-Aware Tools for Delivery Method Analyst
# ===========================================================================

@tool("query_graph_facilities")
def query_graph_facilities() -> str:
    """Query the graph database for all facilities with their regions and
    delivery methods. Returns a JSON array of facility records."""
    global graph_con
    result = graph_con.execute("""
        SELECT f.facility_id, f.longitude, f.latitude,
               li.region_name AS region,
               COALESCE(um.method_name, 'Unassigned') AS delivery_method
        FROM facilities f
        JOIN located_in li ON f.facility_id = li.facility_id
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        ORDER BY li.region_name, f.facility_id
    """).fetchdf()
    return result.to_json(orient='records', indent=2)


@tool("query_graph_regions")
def query_graph_regions() -> str:
    """Query the graph database for region statistics including facility
    counts, delivery method distributions, and adjacency edge counts."""
    global graph_con
    # Region overview with facility count
    region_stats = graph_con.execute("""
        SELECT
            li.region_name,
            COUNT(DISTINCT li.facility_id) AS facility_count,
            COUNT(DISTINCT um.method_name) AS unique_methods
        FROM located_in li
        LEFT JOIN uses_method um ON li.facility_id = um.facility_id
        GROUP BY li.region_name
        ORDER BY facility_count DESC
    """).fetchdf()

    # Delivery method distribution per region
    method_dist = graph_con.execute("""
        SELECT
            li.region_name,
            COALESCE(um.method_name, 'Unassigned') AS delivery_method,
            COUNT(*) AS count
        FROM located_in li
        LEFT JOIN uses_method um ON li.facility_id = um.facility_id
        GROUP BY li.region_name, delivery_method
        ORDER BY li.region_name, count DESC
    """).fetchdf()

    # Connectivity stats per region
    adj_stats = graph_con.execute("""
        SELECT
            li.region_name,
            COUNT(*) AS adjacency_edges,
            ROUND(AVG(a.distance_miles), 1) AS avg_distance_miles,
            ROUND(MIN(a.distance_miles), 1) AS min_distance_miles,
            ROUND(MAX(a.distance_miles), 1) AS max_distance_miles
        FROM connects_to a
        JOIN located_in li ON a.src = li.facility_id
        GROUP BY li.region_name
        ORDER BY adjacency_edges DESC
    """).fetchdf()

    return json.dumps({
        "region_overview": json.loads(region_stats.to_json(orient='records')),
        "method_distribution": json.loads(method_dist.to_json(orient='records')),
        "adjacency_statistics": json.loads(adj_stats.to_json(orient='records'))
    }, indent=2)


@tool("query_delivery_method_stats")
def query_delivery_method_stats() -> str:
    """Query the graph database for delivery method statistics: how many
    facilities use each method, regional breakdown, and average distances
    between facilities using each method."""
    global graph_con
    # Overall method counts
    method_counts = graph_con.execute("""
        SELECT
            COALESCE(um.method_name, 'Unassigned') AS delivery_method,
            COUNT(*) AS facility_count
        FROM facilities f
        LEFT JOIN uses_method um ON f.facility_id = um.facility_id
        GROUP BY delivery_method
        ORDER BY facility_count DESC
    """).fetchdf()

    # Method by region
    method_by_region = graph_con.execute("""
        SELECT
            li.region_name,
            COALESCE(um.method_name, 'Unassigned') AS delivery_method,
            COUNT(*) AS count
        FROM located_in li
        LEFT JOIN uses_method um ON li.facility_id = um.facility_id
        GROUP BY li.region_name, delivery_method
        ORDER BY li.region_name, count DESC
    """).fetchdf()

    # Avg distances between facilities using same delivery method
    method_distances = graph_con.execute("""
        SELECT
            COALESCE(um1.method_name, 'Unknown') AS method,
            COUNT(*) AS edge_count,
            ROUND(AVG(a.distance_miles), 1) AS avg_distance,
            ROUND(MIN(a.distance_miles), 1) AS min_distance,
            ROUND(MAX(a.distance_miles), 1) AS max_distance
        FROM connects_to a
        JOIN uses_method um1 ON a.src = um1.facility_id
        JOIN uses_method um2 ON a.dst = um2.facility_id
        WHERE um1.method_name = um2.method_name
        GROUP BY method
        ORDER BY avg_distance
    """).fetchdf()

    return json.dumps({
        "overall_method_counts": json.loads(method_counts.to_json(orient='records')),
        "method_by_region": json.loads(method_by_region.to_json(orient='records')),
        "same_method_distances": json.loads(method_distances.to_json(orient='records'))
    }, indent=2)


@tool("query_friction_cost_stats")
def query_friction_cost_stats() -> str:
    """Query the graph database for friction-adjusted cost statistics.
    Returns per-delivery-method friction cost summaries including average
    friction ratios (how much longer realistic routes are vs straight-line),
    and per-region breakdowns showing where terrain adds the most cost."""
    global graph_con

    # Check if friction_costs table exists
    try:
        graph_con.execute("SELECT 1 FROM friction_costs LIMIT 1")
    except Exception:
        return json.dumps({
            "status": "no_friction_data",
            "message": "Friction costs have not been computed yet. "
                       "Only Haversine distances are available."
        })

    # Per-method summary
    method_summary = graph_con.execute("""
        SELECT
            delivery_method,
            COUNT(*) AS num_pairs,
            ROUND(AVG(haversine_miles), 1) AS avg_haversine_mi,
            ROUND(AVG(friction_cost), 1) AS avg_friction_cost,
            ROUND(AVG(friction_ratio), 3) AS avg_friction_ratio,
            ROUND(MIN(friction_ratio), 3) AS min_friction_ratio,
            ROUND(MAX(friction_ratio), 3) AS max_friction_ratio
        FROM friction_costs
        GROUP BY delivery_method
        ORDER BY avg_friction_ratio DESC
    """).fetchdf()

    # Per-region per-method breakdown
    region_breakdown = graph_con.execute("""
        SELECT
            li.region_name,
            fc.delivery_method,
            COUNT(*) AS num_pairs,
            ROUND(AVG(fc.friction_ratio), 3) AS avg_friction_ratio,
            ROUND(MAX(fc.friction_ratio), 3) AS max_friction_ratio,
            ROUND(AVG(fc.haversine_miles), 1) AS avg_haversine_mi,
            ROUND(AVG(fc.friction_cost), 1) AS avg_friction_cost
        FROM friction_costs fc
        JOIN located_in li ON fc.src = li.facility_id
        GROUP BY li.region_name, fc.delivery_method
        ORDER BY avg_friction_ratio DESC
    """).fetchdf()

    # Highest friction pairs (terrain bottlenecks)
    top_friction_pairs = graph_con.execute("""
        SELECT
            fc.src, fc.dst, fc.delivery_method,
            f1.community_name AS src_community,
            f2.community_name AS dst_community,
            li.region_name,
            ROUND(fc.haversine_miles, 1) AS haversine_mi,
            ROUND(fc.friction_cost, 1) AS friction_cost,
            ROUND(fc.friction_ratio, 3) AS friction_ratio
        FROM friction_costs fc
        JOIN facilities f1 ON fc.src = f1.facility_id
        JOIN facilities f2 ON fc.dst = f2.facility_id
        JOIN located_in li ON fc.src = li.facility_id
        ORDER BY fc.friction_ratio DESC
        LIMIT 15
    """).fetchdf()

    # Connects_to edges with friction vs haversine comparison
    friction_coverage = graph_con.execute("""
        SELECT
            COUNT(*) AS total_edges,
            COUNT(friction_cost) AS edges_with_friction,
            ROUND(100.0 * COUNT(friction_cost) / COUNT(*), 1) AS coverage_pct
        FROM connects_to
    """).fetchdf()

    return json.dumps({
        "method_summary": json.loads(method_summary.to_json(orient='records')),
        "region_breakdown": json.loads(region_breakdown.to_json(orient='records')),
        "highest_friction_pairs": json.loads(top_friction_pairs.to_json(orient='records')),
        "friction_coverage": json.loads(friction_coverage.to_json(orient='records'))
    }, indent=2)


# ===========================================================================
# Report Save Tool
# ===========================================================================

@tool("save_report")
def save_report(json_report: str) -> str:
    """Save the final analysis report as a JSON file.

    Args:
        json_report: A JSON string containing the analysis report
    """
    with open('market_cost_analysis_report.json', 'w') as f:
        json.dump(json_report, f, indent=4)
    return "Report saved to market_cost_analysis_report.json"


# ===========================================================================
# Agent & Task Setup
# ===========================================================================

def setup_agents(llm):
    """Create all agents and tasks for the market & cost analysis.

    Args:
        llm: CrewAI LLM instance

    Returns:
        Tuple of (agents_list, tasks_list)
    """

    # ----- Agent 1: Market Dynamics Analyst -----
    market_dynamics_analyst = Agent(
        role="Market Dynamics Analyst",
        goal="Analyze market trends regarding fuel in Alaska. Research current "
             "fuel prices, demand, transportation rates, and key market drivers.",
        backstory="You are an expert in energy market economics with deep "
                  "knowledge of Alaska's unique fuel market. You specialize in "
                  "analyzing price trends, demand patterns, supply chain "
                  "dynamics, and regulatory factors that affect fuel delivery "
                  "costs in remote and Arctic regions. You are encouraged to "
                  "draw on broad domain knowledge, research, and market "
                  "intelligence to provide a comprehensive macro-level view.",
        verbose=True,
        llm=llm
    )

    market_dynamics_task = Task(
        description="""Analyze market trends regarding fuel in Alaska:

        1. Research current fuel prices in Alaska, including regional variations
           (e.g., urban vs. rural, road-accessible vs. remote communities)
        2. Analyze demand patterns: seasonal fluctuations, population-driven
           demand, industrial/commercial vs. residential consumption
        3. Assess transportation rates for different delivery methods (barge,
           plane, road transport) and how they affect final fuel costs
        4. Identify key market drivers: oil price volatility, regulatory
           changes, infrastructure investments, climate change impacts on
           delivery windows
        5. Discuss how market forces create incentives or barriers for
           different delivery methods

        Present your findings in a clear narrative style with specific data
        points and examples where possible.

        Focus on market-wide economics and macro trends. Leave facility-
        specific and graph-database analysis to the Delivery Method Analyst.""",
        agent=market_dynamics_analyst,
        expected_output="A comprehensive market dynamics analysis covering "
                       "fuel prices, demand patterns, transportation rates, "
                       "and key market drivers in Alaska."
    )

    # ----- Agent 2: Economic & Environmental Factors Agent -----
    economic_environmental_agent = Agent(
        role="Economic & Environmental Analyst",
        goal="Research seasonality and infrastructure in Alaska in relation "
             "to fuel delivery. Analyze how environmental and economic "
             "factors shape delivery logistics.",
        backstory="You are an expert in Arctic infrastructure, environmental "
                  "science, and economic development. You understand how "
                  "Alaska's extreme climate, seasonal variations, and "
                  "infrastructure limitations create unique challenges for "
                  "fuel delivery. You specialize in analyzing the interplay "
                  "between environmental conditions and economic viability. "
                  "You are encouraged to explore broadly, drawing on research, "
                  "policy knowledge, and environmental science to provide "
                  "context that goes beyond the immediate dataset.",
        verbose=True,
        llm=llm
    )

    economic_environmental_task = Task(
        description="""Research seasonality and infrastructure in Alaska in
        relation to fuel delivery:

        1. Analyze seasonal delivery windows: when can barges reach coastal
           communities? When are ice roads operational? When are airstrips
           accessible year-round vs. seasonally?
        2. Assess infrastructure status: road network coverage, port
           facilities, airstrip conditions, fuel storage capacity at
           remote communities
        3. Research environmental factors: permafrost impacts on
           infrastructure, river ice conditions, coastal erosion affecting
           ports, weather-related delivery disruptions
        4. Evaluate economic factors: cost of maintaining infrastructure in
           Arctic conditions, community size vs. delivery economics,
           government subsidies and programs (e.g., Power Cost Equalization)
        5. Discuss how climate change is altering traditional delivery
           patterns and creating both risks and opportunities

        Present your findings in a narrative style with real examples.""",
        agent=economic_environmental_agent,
        expected_output="A comprehensive analysis of seasonal, environmental, "
                       "and infrastructure factors affecting fuel delivery "
                       "in Alaska."
    )

    # ----- Agent 3: Delivery Method Analyst (with graph DB tools) -----
    delivery_method_analyst = Agent(
        role="Delivery Method Analyst",
        goal="Research the cost and current status of fuel delivery methods "
             "in Alaska. Use the graph database to analyze the structure of "
             "fuel delivery for our test case of bulk fuel facility sites.",
        backstory="You are an expert in logistics and fuel transportation "
                  "with hands-on experience in Alaska's delivery network. "
                  "You have access to a DuckDB graph database containing "
                  "real bulk fuel facility data organized as a property "
                  "graph with facilities, regions, and delivery methods as "
                  "distinct nodes connected by typed edges (located_in, "
                  "uses_method, adjacent_to). You must query this database "
                  "and ground your analysis strictly in the returned data. "
                  "Do not fabricate facility names, counts, distances, or "
                  "statistics — only report what the tools return.",
        verbose=True,
        llm=llm,
        tools=[query_graph_facilities, query_graph_regions,
               query_delivery_method_stats, query_friction_cost_stats]
    )

    delivery_method_task = Task(
        description="""Analyze fuel delivery methods in Alaska using the
        graph database as your primary source of truth:

        IMPORTANT: Query the graph database FIRST. Base your analysis on
        the actual data returned by your tools. Do not invent or assume
        facility details, counts, or distances not present in the data.

        1. Use query_graph_regions to understand the regional structure of
           bulk fuel facilities. How many facilities are in each region?
           What delivery methods dominate each region?
        2. Use query_delivery_method_stats to analyze delivery method
           patterns. Which methods are most common? How do distances vary
           by method?
        3. Use query_graph_facilities to examine specific facility data
           and identify patterns in how facilities are distributed
        4. For each delivery method found in the graph data, summarize:
           - How many facilities use it and in which regions
           - Distance patterns (avg, min, max) from the graph
           - Regional concentration or spread
           Do NOT research general market costs — that is the Market
           Dynamics Analyst's responsibility.
        5. Analyze how the graph structure reveals delivery patterns:
           - Which regions are most dependent on a single delivery method?
           - Where do mixed methods (e.g., 'Plane or Road') indicate
             infrastructure flexibility?
           - How do adjacency distances relate to delivery method choices?
        6. Use query_friction_cost_stats to analyze terrain-adjusted costs:
           - How do friction ratios differ by delivery method? (Road routes
             are typically 1.5-3x longer than straight-line due to terrain
             and road networks; Barge may be shorter via water; Plane is
             near straight-line)
           - Which regions have the highest friction ratios? This indicates
             where terrain, lack of roads, or permafrost add the most cost.
           - Identify the highest-friction facility pairs — these are the
             most expensive/difficult connections in the network.
           - How does friction coverage compare to total edges? Are there
             gaps where friction costs haven't been computed?
           - Compare Haversine distances vs friction-adjusted costs to
             quantify how much real-world terrain adds to delivery costs.

        Present your analysis grounded in graph-derived data, using
        domain expertise only to interpret patterns found in the data.""",
        agent=delivery_method_analyst,
        expected_output="A comprehensive delivery method analysis combining "
                       "graph database insights with research on costs, "
                       "infrastructure, and operational status of each "
                       "delivery method in Alaska."
    )

    # ----- Agent 4: Writing Agent -----
    writing_agent = Agent(
        role="Report Writer & Synthesis Analyst",
        goal="Synthesize findings from all agents into a comprehensive "
             "written report on Alaska fuel delivery market dynamics, costs, "
             "and delivery methods.",
        backstory="You are an expert in fuel delivery economics, policy, "
                  "logistics, and technical writing. You are skilled at "
                  "analyzing complex information from multiple sources and "
                  "presenting it in a clear, actionable format. You serve "
                  "as the moderator for multi-agent discussions, drawing "
                  "out key insights and identifying areas of agreement "
                  "and disagreement.",
        verbose=True,
        llm=llm,
        tools=[save_report]
    )

    # ----- Agent 5: Contrarian Agent -----
    contrarian_agent = Agent(
        role="Supply Chain & Logistics Contrarian",
        goal="Critically evaluate the analysis and recommendations of other "
             "agents, identifying logical inconsistencies, potential "
             "oversights, and alternative perspectives from a supply chain "
             "and logistics perspective.",
        backstory="You are a seasoned supply chain and logistics expert with "
                  "a contrarian mindset. You have decades of experience in "
                  "Arctic supply chains and excel at identifying flaws in "
                  "reasoning, finding alternative explanations, and ensuring "
                  "that recommendations are grounded in operational reality. "
                  "Pay special attention to whether claims about facilities, "
                  "regions, and delivery patterns are supported by the graph "
                  "database data rather than assumed.",
        verbose=True,
        llm=llm
    )

    # ===================================================================
    # Tasks
    # ===================================================================

    # ----- Phase 2: Multi-Agent Discussion -----
    multi_agent_discussion_task = Task(
        description="""Lead a structured discussion to synthesize findings
        from the Market Dynamics Analyst, Economic & Environmental Analyst,
        and Delivery Method Analyst.

        * Market Dynamics Analyst contributions:
            - Market trends, fuel prices, demand patterns
            - Key market drivers and their regional impacts
            - Future market outlook and implications for delivery strategies

        * Economic & Environmental Analyst contributions:
            - Seasonal delivery windows and infrastructure constraints
            - Environmental factors affecting operations
            - Economic viability considerations for different regions

        * Delivery Method Analyst contributions:
            - Graph-based analysis of facility distribution and delivery patterns
            - Delivery method costs, reliability, and infrastructure requirements
            - Regional delivery method dependencies and flexibility

        As the moderator, you should:
        1. Synthesize information across all three analyses
        2. Identify key themes and cross-cutting insights
        3. Note areas of agreement and potential conflicts
        4. Ask clarifying questions such as:
           - How do market price fluctuations impact different delivery
             methods in specific regions?
           - What are the most significant logistical bottlenecks?
           - What synergies or conflicts exist between delivery methods?
           - How does the graph data support or challenge market assumptions?
        5. Identify the most important findings for the final report

        The goal is to produce actionable insights for the final report.""",
        agent=writing_agent,
        expected_output="A structured summary of the multi-agent discussion "
                       "highlighting key points, cross-cutting insights, "
                       "specific examples, and actionable findings from each "
                       "agent that should be included in the final report."
    )

    # ----- Phase 3: Contrarian Review & Dialogue -----
    contrarian_task = Task(
        description="""Critically review the comprehensive analyses and
        discussion produced by the Market Dynamics Analyst, Economic &
        Environmental Analyst, Delivery Method Analyst, and Writing Agent.

        From a supply chain management and logistics perspective, evaluate:

        1. Logical inconsistencies or contradictions in the analysis
        2. Potential oversights or missing considerations
        3. Alternative explanations or perspectives
        4. Unrealistic or impractical assumptions
        5. Areas where the graph-based analysis could be interpreted
           differently
        6. Missing supply chain risks or opportunities
        7. Whether the market analysis adequately accounts for Alaska's
           unique operating environment

        Present your findings in a structured format with specific
        citations to the analyses you are critiquing.""",
        agent=contrarian_agent,
        expected_output="A structured critical review identifying logical "
                       "inconsistencies, oversights, alternative perspectives, "
                       "and specific areas for improvement."
    )

    writing_response_task = Task(
        description="""Review the critique provided by the Contrarian Agent.
        Respond to each point, providing:
        1. Justifications for your analysis where the critique is unfounded
        2. Acknowledgment of valid concerns
        3. Proposed revisions or alternative approaches where warranted
        4. Additional evidence or reasoning to strengthen weak areas""",
        agent=writing_agent,
        expected_output="A point-by-point response to the contrarian critique "
                       "with justifications, acknowledgments, and revisions."
    )

    contrarian_followup_task = Task(
        description="""Review the Writing Agent's response to your critique.
        Provide:
        1. Follow-up critiques on any points not adequately addressed
        2. Confirmation of concerns that were satisfactorily resolved
        3. Any final recommendations for strengthening the analysis""",
        agent=contrarian_agent,
        expected_output="Follow-up critiques or confirmation that concerns "
                       "have been addressed, with any final recommendations."
    )

    # ----- Phase 4: Final Report -----
    final_report_task = Task(
        description="""Produce a comprehensive final report integrating all
        agent analyses, the multi-agent discussion, and the contrarian review.

        The report should be a single JSON document with the following
        structure. Use the save_report tool to save it:

        {
        "title": "Alaska Fuel Delivery Market & Cost Analysis",
        "executive_summary": "2-3 paragraphs summarizing key findings",
        "main_report": {
            "introduction": "Overview of Alaska fuel delivery landscape",
            "market_dynamics": "Current market trends, prices, demand drivers",
            "economic_environmental_factors": "Seasonality, infrastructure, climate",
            "delivery_methods_analysis": "Cost, status, and patterns of each method",
            "graph_database_insights": "Key findings from the facility graph data",
            "challenges": "Primary challenges facing fuel delivery in Alaska",
            "opportunities": "Key opportunities for improvement",
            "conclusion": "Synthesis of all findings"
        },
        "recommendations": {
            "strategic_priorities": [
                {"priority": "Name", "description": "Details",
                 "implementation_steps": ["Step 1", "Step 2"],
                 "expected_impact": "Outcome",
                 "supporting_evidence": "Reference to findings"}
            ],
            "operational_improvements": [
                {"improvement": "Name", "description": "Details",
                 "implementation_steps": ["Step 1", "Step 2"],
                 "expected_impact": "Outcome"}
            ]
        },
        "agent_discussion_summary": {
            "overview": "How the multi-agent discussion shaped the analysis",
            "market_dynamics_insights": ["Key insight 1", "Key insight 2"],
            "economic_environmental_insights": ["Key insight 1"],
            "delivery_method_insights": ["Key insight 1"],
            "synthesis_insights": ["Cross-functional considerations"],
            "impact_on_recommendations": "How discussion influenced recommendations"
        },
        "limitations": {
            "contrarian_critique_initial": "Summary of initial critique",
            "response_to_critique": "How concerns were addressed",
            "follow_up_points": ["Point 1", "Point 2"],
            "acknowledged_limitations": ["Limitation 1"],
            "areas_for_further_research": ["Research area 1"]
        },
        "metadata": {
            "date_generated": "YYYY-MM-DD",
            "agents_involved": ["Market Dynamics Analyst",
                "Economic & Environmental Analyst",
                "Delivery Method Analyst", "Writing Agent",
                "Contrarian Agent"],
            "data_sources": ["regionalization.duckdb graph database",
                "Agent domain knowledge"],
            "confidence_level": "High/Medium/Low"
        }
        }""",
        agent=writing_agent,
        expected_output="A comprehensive JSON report saved to "
                       "market_cost_analysis_report.json via the save_report tool."
    )

    agents = [market_dynamics_analyst, economic_environmental_agent,
              delivery_method_analyst, writing_agent, contrarian_agent]

    tasks = [
        market_dynamics_task,                # Phase 1: Individual analysis
        economic_environmental_task,         # Phase 1
        delivery_method_task,                # Phase 1 (uses graph DB tools)
        multi_agent_discussion_task,         # Phase 2: Multi-agent discussion
        contrarian_task,                     # Phase 3: Contrarian review
        writing_response_task,               # Phase 3: Writing response
        contrarian_followup_task,            # Phase 3: Contrarian follow-up
        final_report_task                    # Phase 4: Final report
    ]

    return agents, tasks


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Run the market & cost analysis pipeline."""
    # LLM setup (Ollama)
    llm = pipeline.get_llm()

    # Connect to graph database (read-only)
    connect_graph_db('regionalization.duckdb')

    # Print graph summary for context
    print("\nGraph Database Summary:")
    for table in ['facilities', 'regions', 'delivery_methods',
                  'located_in', 'uses_method', 'adjacent_to']:
        count = graph_con.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        print(f"  {table}: {count} rows")
    print()

    # Set up agents and tasks
    agents, tasks = setup_agents(llm)

    # Configure crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        llm=llm
    )

    print("Running Market & Cost Analysis Crew...")
    print("=" * 60)

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("CREW EXECUTION COMPLETE")
    print("=" * 60)
    print(result)

    # Save result (backup in case save_report tool wasn't called)
    with open('market_cost_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump({"result": str(result)}, f, indent=4)

    # Close graph connection
    graph_con.close()
    print("Graph database connection closed.")


if __name__ == "__main__":
    main()
