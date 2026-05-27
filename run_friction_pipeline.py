"""run_friction_pipeline.py

Orchestrate the agent-free friction-surface stage end-to-end:
  1. Build 36 mode-month friction surfaces from preprocessed inputs.
  2. Compute cost-distance edges per facility pair.
  3. Write edges to DuckDB (new mode_specific_edges table + connects_to backfill).

No CrewAI, no LLM calls.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pipeline
from friction_surface import MODES, write_friction_stack
from routing_wbt import DEFAULT_REPR_MONTHS, compute_edge_costs, write_edges_to_graph

logger = logging.getLogger(__name__)


REQUIRED_STATIC = ("slope.tif", "lulc.tif", "permafrost.tif")


def _validate_inputs(input_dir: Path) -> None:
    missing: list[str] = []
    for name in REQUIRED_STATIC:
        if not (input_dir / name).exists():
            missing.append(name)
    for month in range(1, 13):
        for kind in ("sea_ice", "river_ice"):
            rel = f"{kind}/{kind}_{month:02d}.tif"
            if not (input_dir / rel).exists():
                missing.append(rel)
    if missing:
        raise FileNotFoundError(
            "Missing required preprocessed inputs in "
            f"{input_dir}:\n  " + "\n  ".join(missing)
        )


def _collect_existing_stack(output_dir: Path) -> dict[tuple[str, int], Path]:
    stack: dict[tuple[str, int], Path] = {}
    for mode in MODES:
        for month in range(1, 13):
            p = output_dir / f"{mode}_{month:02d}.tif"
            if p.exists():
                stack[(mode, month)] = p
    return stack


def _print_summary(edge_df, friction_stack, t0: float) -> None:
    print("\n" + "=" * 56)
    print("Friction pipeline summary")
    print("=" * 56)
    print(f"  Surfaces written: {len(friction_stack)}")
    if edge_df is None or edge_df.empty:
        print("  No edges computed.")
    else:
        per_mode = edge_df.groupby("mode")
        for mode_name, grp in per_mode:
            reachable = grp[~grp["unreachable"]]
            unreachable_n = int(grp["unreachable"].sum())
            n = len(grp)
            mean_cost = reachable["total_cost"].mean() if not reachable.empty else float("nan")
            print(
                f"  {mode_name:>9}: {n:>5} edges  "
                f"unreachable={unreachable_n:>4}  "
                f"mean_cost={mean_cost:.2f}"
            )
    print(f"  Total runtime:    {time.time() - t0:.1f}s")
    print("=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Preprocessed raster inputs (defaults to RASTER_DIR env var).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write friction TIFs (defaults to <RASTER_DIR>/friction).",
    )
    parser.add_argument(
        "--skip-surfaces",
        action="store_true",
        help="Reuse existing friction TIFs in --output-dir; only recompute edges.",
    )
    args = parser.parse_args()

    pipeline.setup_logging("outputs")
    t0 = time.time()

    input_dir = Path(args.input_dir) if args.input_dir else Path(pipeline.get_raster_dir())
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "friction"

    if args.skip_surfaces:
        friction_stack = _collect_existing_stack(output_dir)
        if not friction_stack:
            raise FileNotFoundError(
                f"--skip-surfaces was set but no friction TIFs found in {output_dir}"
            )
        logger.info("reusing %d existing friction surfaces", len(friction_stack))
    else:
        _validate_inputs(input_dir)
        logger.info("building friction surfaces from %s", input_dir)
        friction_stack = write_friction_stack(input_dir, output_dir)
        logger.info("wrote %d friction surfaces", len(friction_stack))

    con = pipeline.get_duckdb_connection()
    try:
        edge_df = compute_edge_costs(
            con, friction_stack, output_dir,
            repr_months=DEFAULT_REPR_MONTHS,
        )
        write_edges_to_graph(edge_df, con, repr_months=DEFAULT_REPR_MONTHS)
    finally:
        con.close()

    _print_summary(edge_df, friction_stack, t0)


if __name__ == "__main__":
    main()
