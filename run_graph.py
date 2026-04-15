"""run_graph.py

Orchestration script for the DuckDB graph-based MAS pipeline. Calls:
1. regionalization_graph.py  - Creates regionalization.duckdb graph database
2. market_cost_analysis.py   - Market & cost analysis using graph DB
3. friction_surface.py       - Friction raster computation (no LLM)
4. friction_agents.py        - Seasonal, cost, and validation agents
5. tsp_model_graph.py        - TSP route optimization using graph DB

All modules share the same DuckDB database file (regionalization.duckdb).
"""

import os
import json
import sys
from pathlib import Path
import regionalization_graph
import market_cost_analysis
import friction_surface
import friction_agents
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

def run_step(step_name, step_func):
    """Run a pipeline step with LLM health check and timeout retry."""
    if not should_run(step_name):
        return

    pipeline.check_llm()
    try:
        step_func()
    except Exception as e:
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            print(f"\nTimeout on {step_name}, checking LLM and retrying...")
            pipeline.check_llm()
            step_func()  # retry once
        else:
            raise
    save_checkpoint(step_name)


def run_compute_step(step_name, step_func):
    """Run a compute-only step (no LLM required)."""
    if not should_run(step_name):
        return

    step_func()
    save_checkpoint(step_name)


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # Set up file logging so overnight runs can be fully reviewed
    log_file = pipeline.setup_logging()

    # Tee stdout/stderr to the log file so print() output is also captured
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    _log_fh = open(log_file, "a")
    sys.stdout = Tee(sys.__stdout__, _log_fh)
    sys.stderr = Tee(sys.__stderr__, _log_fh)

    # Step 1: Regionalization - creates the graph database
    print("=" * 60)
    print("STEP 1: Regionalization (Graph Database Creation)")
    print("=" * 60)
    run_step("regionalization", regionalization_graph.main)

    # Step 2: Market & Cost Analysis - reads graph DB (read-only)
    print("\n" + "=" * 60)
    print("STEP 2: Market & Cost Analysis")
    print("=" * 60)
    run_step("market_cost_analysis", market_cost_analysis.main)

    # Step 3: Friction Surface Computation - pure computation, no LLM
    print("\n" + "=" * 60)
    print("STEP 3: Friction Surface Computation")
    print("=" * 60)
    run_compute_step("friction_surface", friction_surface.main)

    # Step 4: Friction Agents - seasonal, cost, and validation via CrewAI
    print("\n" + "=" * 60)
    print("STEP 4: Friction Agents (Seasonal, Cost, Validation)")
    print("=" * 60)
    run_step("friction_agents", friction_agents.main)

    # Step 5: TSP Route Optimization - reads & writes to graph DB
    print("\n" + "=" * 60)
    print("STEP 5: TSP Route Optimization")
    print("=" * 60)
    run_step("tsp_optimization", tsp_model_graph.main)

    # Step 6: Copy JSON outputs to outputs folder
    print("\n" + "=" * 60)
    print("STEP 6: Saving outputs")
    print("=" * 60)
    pipeline.save_json()

    print("\nPipeline complete.")

    # Reset checkpoints so the pipeline can be re-run for new output variations.
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
        print("Checkpoints cleared for next run.")

    _log_fh.close()
