"""run_graph.py

Orchestration script for the DuckDB graph-based MAS pipeline.

This replaces run.py and calls the new graph-integrated modules:
1. regionalization_graph.py  - Creates regionalization.duckdb graph database
2. friction_layer_graph.py   - Friction layer computation (terrain/road/delivery costs)
3. market_cost_analysis.py   - Market & cost analysis using graph DB
4. tsp_model_graph.py        - TSP route optimization using graph DB

All modules share the same DuckDB database file (regionalization.duckdb).
"""

import os
import regionalization_graph
import friction_layer_graph
import market_cost_analysis
import tsp_model_graph
import pipeline

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # Step 1: Regionalization - creates the graph database
    print("=" * 60)
    print("STEP 1: Regionalization (Graph Database Creation)")
    print("=" * 60)
    regionalization_graph.main()

    # Step 2: Friction Layer - computes terrain/road/delivery cost surfaces
    print("\n" + "=" * 60)
    print("STEP 2: Friction Layer Computation")
    print("=" * 60)
    friction_layer_graph.main()

    # Step 3: Market & Cost Analysis - reads graph DB (read-only)
    print("\n" + "=" * 60)
    print("STEP 3: Market & Cost Analysis")
    print("=" * 60)
    market_cost_analysis.main()

    # Step 4: TSP Route Optimization - reads & writes to graph DB
    print("\n" + "=" * 60)
    print("STEP 4: TSP Route Optimization")
    print("=" * 60)
    tsp_model_graph.main()

    # Step 5: Copy JSON outputs to outputs folder
    print("\n" + "=" * 60)
    print("STEP 5: Saving outputs")
    print("=" * 60)
    pipeline.save_json()

    print("\nPipeline complete.")
