"""Per-participant turn-and-gaze timeline.

One stacked figure, one row per participant (P5 excluded). Each row shows:
  - robot-turn bars along the top band (green)
  - participant-turn bars along the middle band (orange)
  - gaze-state ribbon along the bottom band: binary robot / outside
  - vertical tick marks at every participant_turn_start and _end

Gaze state ribbon shows two colors. Bins with majority tracking loss
are left unpainted (background shows through).
  on-robot = Face AOI hit (Eyes / Mouth / Nose included by AOI nesting)
  away     = valid sample outside the Face AOI (Body, Outside, no-AOI-hit)

Samples are binned into 50 ms windows and the majority state wins.

Output: figures/turn_timeline.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from step1_preprocess import (
    EVENTS_CSV,
    REC_TO_PID,
    load_events,
    preprocess_recording,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "figures" / "turn_timeline.png"

EXCLUDED = {"P5"}
BIN_S = 0.05  # 50 ms ribbon bins

# State codes for the ribbon:
#   0 = away (valid, not Face AOI), 1 = on-robot (Face AOI), 2 = invalid (masked)
STATE_COLORS = ["#fdae61", "#1a9850"]
STATE_LABELS = ["away", "on-robot"]

ROBOT_TURN_COLOR = "#66c2a5"
PART_TURN_COLOR = "#fc8d62"
MARK_COLOR = "#c5363d"


def build_state_ribbon(
    samples, t0: float, t1: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bin samples into BIN_S windows; each bin's state = majority sample.

    Three-way: 0 = away (valid, not Face AOI), 1 = on-robot (Face AOI),
    2 = invalid (tracking loss). Invalid bins are masked in the figure so
    the background shows through.
    """
    edges = np.arange(t0, t1 + BIN_S, BIN_S)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ts = samples["timestamp_s"].to_numpy()
    on_robot = samples["on_robot"].to_numpy().astype(np.int32)
    valid = samples["valid"].to_numpy().astype(bool)
    category = np.where(~valid, 2, on_robot)

    idx = np.clip(np.searchsorted(edges, ts, side="right") - 1, 0, len(centers) - 1)

    counts = np.zeros((len(centers), 3), dtype=np.int32)
    np.add.at(counts, (idx, category), 1)
    bin_state = counts.argmax(axis=1)

    return centers, bin_state


def draw_participant(ax, prep) -> None:
    t0, t1 = prep.recording_start_s, prep.recording_end_s

    # Lane y-extents. Top = robot turns, middle = participant turns, bottom = ribbon.
    ROBOT_Y = (0.72, 0.98)
    PART_Y  = (0.40, 0.66)
    RIB_Y   = (0.06, 0.32)

    for rs, re in prep.robot_turns:
        ax.axvspan(rs, re, ymin=ROBOT_Y[0], ymax=ROBOT_Y[1],
                   color=ROBOT_TURN_COLOR, lw=0)
    for ps, pe in prep.participant_turns:
        ax.axvspan(ps, pe, ymin=PART_Y[0], ymax=PART_Y[1],
                   color=PART_TURN_COLOR, lw=0)

    # Participant turn boundary ticks
    for ps, pe in prep.participant_turns:
        for tmark in (ps, pe):
            ax.axvline(tmark, color=MARK_COLOR, lw=0.6, alpha=0.6,
                       ymin=0.0, ymax=1.0)

    # Gaze state ribbon (invalid bins masked; background shows through)
    centers, bin_state = build_state_ribbon(prep.samples, t0, t1)
    masked = np.ma.masked_where(bin_state == 2, bin_state)
    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.pcolormesh(
        np.r_[centers - BIN_S / 2, centers[-1] + BIN_S / 2],
        np.array([RIB_Y[0], RIB_Y[1]]),
        masked[np.newaxis, :],
        cmap=cmap,
        norm=norm,
        shading="flat",
    )

    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 1)
    ax.set_yticks([
        (ROBOT_Y[0] + ROBOT_Y[1]) / 2,
        (PART_Y[0] + PART_Y[1]) / 2,
        (RIB_Y[0] + RIB_Y[1]) / 2,
    ])
    ax.set_yticklabels(["robot turn", "participant turn", "gaze"])
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel(prep.pid, rotation=0, labelpad=22, fontsize=11, va="center")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)


def main() -> None:
    events_by_rec = load_events(EVENTS_CSV)
    preps = []
    for rec, pid in REC_TO_PID.items():
        if pid in EXCLUDED:
            continue
        preps.append(preprocess_recording(rec, events_by_rec[rec]))

    fig, axes = plt.subplots(
        len(preps), 1, figsize=(14, 2.0 * len(preps)), sharex=False
    )
    if len(preps) == 1:
        axes = [axes]

    for ax, prep in zip(axes, preps):
        draw_participant(ax, prep)

    axes[-1].set_xlabel("Recording time (s)")

    legend_handles = [
        Patch(facecolor=ROBOT_TURN_COLOR, label="robot turn"),
        Patch(facecolor=PART_TURN_COLOR,  label="participant turn"),
        Patch(facecolor=STATE_COLORS[1],  label="gaze: on-robot"),
        Patch(facecolor=STATE_COLORS[0],  label="gaze: away"),
        Patch(facecolor="none", edgecolor=MARK_COLOR, label="participant turn start/end"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH.name}")


if __name__ == "__main__":
    main()
