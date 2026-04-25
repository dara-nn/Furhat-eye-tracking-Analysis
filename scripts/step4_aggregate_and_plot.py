"""Step 4 — Aggregate the per-participant results and draw the two boxplots.

Takes the per-recording RQ1 and RQ2 results from Steps 2 and 3, reduces
each participant's per-turn lists to means, and writes

- ``results/results_per_participant.csv`` (one row per participant),
- ``figures/rq1_boxplot.png``,
- ``figures/rq2_boxplot.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step2_rq1_turn_taking import Rq1Result
from step3_rq3_role_asymmetry import Rq3Result


# --- Aggregation -------------------------------------------------------------


def build_results_df(
    rq1_by_pid: dict[str, Rq1Result],
    rq3_by_pid: dict[str, Rq3Result],
) -> pd.DataFrame:
    rows = []
    for pid in rq1_by_pid:
        r1 = rq1_by_pid[pid]
        r3 = rq3_by_pid[pid]
        def _mean(xs):
            return float(np.mean(xs)) if xs else float("nan")
        rows.append(
            {
                "participant": pid,
                "rq1_claim_pct": _mean(r1.claim_pcts),                # ±1 s primary
                "rq1_yield_pct": _mean(r1.yield_pcts),                # ±1 s primary
                "rq1_claim_pct_w500": _mean(r1.claim_pcts_w500),      # ±0.5 s sensitivity
                "rq1_yield_pct_w500": _mean(r1.yield_pcts_w500),      # ±0.5 s sensitivity
                "rq3_speaking_pct": r3.speak_pct,
                "rq3_listening_pct": r3.listen_pct,
                "rq3_speak_at_robot_s": r3.speak_at_robot,
                "rq3_speak_total_s": r3.speak_total,
                "rq3_listen_at_robot_s": r3.listen_at_robot,
                "rq3_listen_total_s": r3.listen_total,
                "n_claims": len(r1.claim_pcts),
                "n_yields": len(r1.yield_pcts),
                "n_speak": r3.n_speak_turns,
                "n_listen": r3.n_listen_turns,
            }
        )
    return pd.DataFrame(rows)


# --- Plots -------------------------------------------------------------------


def _boxplot(
    results: pd.DataFrame,
    cols: list[str],
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    data = [results[c].dropna().to_numpy() for c in cols]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True, widths=0.5, showfliers=False)
    for i, arr in enumerate(data, start=1):
        jitter = np.random.default_rng(i).uniform(-0.08, 0.08, size=len(arr))
        ax.scatter(np.full_like(arr, i) + jitter, arr, alpha=0.7, s=36, zorder=3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Proportion of window / turn")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rq1(results: pd.DataFrame, out_path: Path) -> None:
    _boxplot(
        results,
        ["rq1_claim_pct", "rq1_yield_pct"],
        [
            "Look-away at turn start\n% of turns with on-robot \u2192 away\n(within \u00b11 s of turn start)",
            "Look-back at turn end\n% of turns with away \u2192 on-robot\n(within \u00b11 s of turn end)",
        ],
        "RQ1. Does a gaze shift happen around the\nparticipant's turn start / end?",
        out_path,
    )


def plot_rq2(results: pd.DataFrame, out_path: Path) -> None:
    _boxplot(
        results,
        ["rq3_speaking_pct", "rq3_listening_pct"],
        [
            "Participant speaking\n(% time looking at robot)",
            "Participant listening\n(% time looking at robot)",
        ],
        "RQ2. % time participant looked at robot,\nwhile speaking vs. while listening",
        out_path,
    )
