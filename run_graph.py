"""run_graph.py

Orchestration script for the DuckDB graph-based MAS pipeline.

This replaces run.py and calls the new graph-integrated modules:
1. regionalization_graph.py  - Creates regionalization.duckdb graph database
2. market_cost_analysis.py   - Market & cost analysis using graph DB
3. tsp_model_graph.py        - TSP route optimization using graph DB

All modules share the same DuckDB database file (regionalization.duckdb).
"""

import os
import json
from pathlib import Path
import regionalization_graph
import market_cost_analysis
import tsp_model_graph
import pipeline

CHECKPOINT_FILE = "outputs/.checkpoint"


def load_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        return json.loads(Path(CHECKPOINT_FILE).read_text())
    return {"completed_steps": []}


def save_checkpoint(step):
    cp = load_checkpoint()
    if step not in cp["completed_steps"]:
        cp["completed_steps"].append(step)
    Path(CHECKPOINT_FILE).write_text(json.dumps(cp))
    print(f"  Checkpoint saved: {step}")


def should_run(step):
    if step in load_checkpoint()["completed_steps"]:
        print(f"  Skipping {step} (already complete)")
        return False
    return True

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # Step 1: Regionalization - creates the graph database
    print("=" * 60)
    print("STEP 1: Regionalization (Graph Database Creation)")
    print("=" * 60)
    if should_run("regionalization"):
        regionalization_graph.main()
        save_checkpoint("regionalization")

    # Step 2: Market & Cost Analysis - reads graph DB (read-only)
    print("\n" + "=" * 60)
    print("STEP 2: Market & Cost Analysis")
    print("=" * 60)
    if should_run("market_cost_analysis"):
        market_cost_analysis.main()
        save_checkpoint("market_cost_analysis")

    # Step 3: TSP Route Optimization - reads & writes to graph DB
    print("\n" + "=" * 60)
    print("STEP 3: TSP Route Optimization")
    print("=" * 60)
    if should_run("tsp_optimization"):
        tsp_model_graph.main()
        save_checkpoint("tsp_optimization")

    # Step 4: Copy JSON outputs to outputs folder
    print("\n" + "=" * 60)
    print("STEP 4: Saving outputs")
    print("=" * 60)
    pipeline.save_json()

    print("\nPipeline complete.")

    # Reset checkpoints so the pipeline can be re-run for new output variations.
    # Checkpoints only protect against mid-run crashes; once all steps succeed,
    # clear them to allow fresh runs.
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
        print("Checkpoints cleared for next run.")
