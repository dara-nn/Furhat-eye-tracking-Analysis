"""Pie-chart version of the sub-AOI fixation time figure.

Reads ``subaoi_dwell_per_participant.csv`` and renders three pies
(overall / speaking / listening), each showing the mean across N=5
participants of fixation-time share per sub-AOI.

Output: figures/subaoi_dwell_pies.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "subaoi_dwell_per_participant.csv"
OUT_PATH = ROOT / "figures" / "subaoi_dwell_pies.png"

EXCLUDED = {"P5"}
ROLES = ["overall", "speaking", "listening"]

SLICES = [
    ("eyes_pct",       "Eyes",         "#1f77b4"),
    ("mouth_pct",      "Mouth",        "#d62728"),
    ("nose_pct",       "Nose",         "#9467bd"),
    ("face_other_pct", "Face (other)", "#2ca02c"),
    ("body_pct",       "Body",         "#8c564b"),
    ("outside_pct",    "Outside",      "#7f7f7f"),
    ("none_pct",       "No AOI",       "#d9d9d9"),
]


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df[~df["participant"].isin(EXCLUDED)].copy()

    cols = [c for c, _, _ in SLICES]
    labels = [lab for _, lab, _ in SLICES]
    colors = [col for _, _, col in SLICES]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for ax, role in zip(axes, ROLES):
        sub = df[df["role"] == role]
        means = sub[cols].mean().to_numpy() * 100.0
        total = means.sum()
        means_norm = means / total * 100.0 if total > 0 else means

        def autopct(pct: float) -> str:
            return f"{pct:.1f}%" if pct >= 2.0 else ""

        wedges, _texts, autotexts = ax.pie(
            means_norm,
            colors=colors,
            autopct=autopct,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(edgecolor="white", linewidth=1.2),
            pctdistance=0.72,
        )
        for t in autotexts:
            t.set_color("white")
            t.set_fontsize(10)
            t.set_fontweight("bold")
        ax.set_title(role.capitalize(), fontsize=13)

    axes[0].legend(
        labels,
        loc="center left",
        bbox_to_anchor=(-0.35, 0.5),
        frameon=False,
        fontsize=10,
        title="Sub-AOI",
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH.name}")


if __name__ == "__main__":
    main()
