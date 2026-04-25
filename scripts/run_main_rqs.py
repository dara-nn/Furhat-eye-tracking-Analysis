"""Orchestrator for the main RQ1 and RQ2 pipeline.

Runs the four research-analysis steps (preprocess, RQ1 look-away / look-back,
RQ2 speaking / listening, aggregate + plot) end to end. Does NOT run the §V
additional analyses (heatmap, sub-AOI, pupil, timeline, baseline); those
are independent scripts under scripts/ and each runs on its own.

Usage:
    python3 scripts/run_main_rqs.py                 # all six participants
    python3 scripts/run_main_rqs.py --exclude P5    # drop P5 (tracking-quality outlier)

Outputs (relative to the repo root):
    results/results_per_participant.csv
    figures/rq1_boxplot.png
    figures/rq2_boxplot.png
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from step1_preprocess import (
    EVENTS_CSV,
    REC_TO_PID,
    ROOT,
    load_events,
    preprocess_recording,
)
from step2_rq1_turn_taking import Rq1Result, compute_rq1
from step3_rq3_role_asymmetry import Rq3Result, compute_rq3
from step4_aggregate_and_plot import build_results_df, plot_rq1, plot_rq2


def main() -> int:
    parser = argparse.ArgumentParser(description="RQ1/RQ3 gaze analysis.")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Participant IDs to exclude, e.g. --exclude P5",
    )
    args = parser.parse_args()
    excluded = {p.strip() for p in args.exclude}

    events_by_rec = load_events(EVENTS_CSV)

    if excluded:
        print(f"Excluding participants: {sorted(excluded)}")

    rq1_by_pid: dict[str, Rq1Result] = {}
    rq3_by_pid: dict[str, Rq3Result] = {}

    print("\n=== Running preprocessing + RQ1 + RQ3 per participant ===")
    for rec, pid in REC_TO_PID.items():
        if pid in excluded:
            print(f"  {rec} ({pid}): excluded, skipping")
            continue
        print(f"  {rec} ({pid}): preprocessing ...", flush=True)
        prep = preprocess_recording(rec, events_by_rec[rec])
        rq1_by_pid[pid] = compute_rq1(prep)
        rq3_by_pid[pid] = compute_rq3(prep)

    results = build_results_df(rq1_by_pid, rq3_by_pid)

    print("\n--- Per-participant results ---")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(results.to_string(index=False))

    out_csv = ROOT / "results" / "results_per_participant.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    rq1_png = ROOT / "figures" / "rq1_boxplot.png"
    rq2_png = ROOT / "figures" / "rq2_boxplot.png"
    plot_rq1(results, rq1_png)
    plot_rq2(results, rq2_png)
    print(f"Saved: {rq1_png}")
    print(f"Saved: {rq2_png}")

    print(f"\n--- Aggregate means (N={len(results)} participants) ---")
    for col in ["rq1_claim_pct", "rq1_yield_pct", "rq3_speaking_pct", "rq3_listening_pct"]:
        vals = results[col].dropna()
        print(f"  {col:22s} mean={vals.mean():.3f}  sd={vals.std():.3f}  n={len(vals)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
