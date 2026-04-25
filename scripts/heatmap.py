"""Fixation heatmap overlaid on the scene-camera snapshot.

For each excluded-P5 participant, read the Pro Lab mapped fixation
coordinates from the raw Tobii TSV, bin them into a 2D histogram over the
snap.jpg pixel grid, Gaussian-smooth (sigma ~25 px), and render as an
alpha-overlay on the snapshot. Weighted by sample count (one vote per
fixation-classified gaze sample) — the longer a fixation, the heavier it
contributes, matching Pro Lab's default "Heat map by gaze" behaviour.

Outputs (all under ``figures/``):
    heatmap_all.png        pooled across all valid samples
    heatmap_speaking.png   samples inside participant turns
    heatmap_listening.png  samples inside robot turns
    heatmap_per_participant.png  2x3 grid, one overall heatmap per P

Run from the repo root:  ``python3 scripts/heatmap.py``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from step1_preprocess import (
    EVENTS_CSV,
    GAZE_DIR,
    REC_TO_PID,
    load_events,
    preprocess_recording,
)

ROOT = Path(__file__).resolve().parent.parent
SNAP_PATH = ROOT / "data" / "from-glass-data" / "snap.jpg"

EXCLUDED = {"P5"}
SIGMA_PX = 25.0
CMAP = "inferno"

FIX_X_COL = "Mapped fixation X [snap]"
FIX_Y_COL = "Mapped fixation Y [snap]"


def load_mapped_fixation_samples(tsv_path: Path) -> pd.DataFrame:
    """Return one row per fixation-classified sample with mapped (x, y) and timestamp_s."""
    cols = [
        "Recording timestamp",
        "Eye movement type",
        FIX_X_COL,
        FIX_Y_COL,
    ]
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False, usecols=cols)
    df = df[df["Eye movement type"] == "Fixation"]
    df = df.dropna(subset=[FIX_X_COL, FIX_Y_COL])
    df = df.rename(columns={FIX_X_COL: "x", FIX_Y_COL: "y"})
    df["timestamp_s"] = df["Recording timestamp"].astype(float) / 1_000_000
    return df[["timestamp_s", "x", "y"]].reset_index(drop=True)


def mask_in_windows(ts: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(len(ts), dtype=bool)
    for a, b in windows:
        m |= (ts >= a) & (ts < b)
    return m


def compute_heatmap(xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> np.ndarray:
    """1-px-bin 2D histogram, Gaussian-smoothed. Shape (h, w)."""
    if len(xs) == 0:
        return np.zeros((h, w), dtype=float)
    counts, _, _ = np.histogram2d(
        ys, xs, bins=[h, w], range=[[0, h], [0, w]]
    )
    return gaussian_filter(counts, sigma=SIGMA_PX)


def overlay_on_ax(ax, snap: np.ndarray, heat: np.ndarray, title: str) -> None:
    h, w = snap.shape[:2]
    ax.imshow(snap, extent=(0, w, h, 0))
    if heat.max() > 0:
        norm = heat / heat.max()
        cmap = plt.get_cmap(CMAP)
        rgba = cmap(norm)
        # Transparent where no density; opaque where peaks. sqrt keeps mid tones readable.
        rgba[..., 3] = np.sqrt(norm) * 0.75
        ax.imshow(rgba, extent=(0, w, h, 0))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def main() -> None:
    snap = np.array(Image.open(SNAP_PATH).convert("RGB"))
    h, w = snap.shape[:2]

    events_by_rec = load_events(EVENTS_CSV)

    per_participant: dict[str, dict[str, np.ndarray]] = {}
    pooled: dict[str, np.ndarray] = {
        "all": np.zeros((h, w), dtype=float),
        "speaking": np.zeros((h, w), dtype=float),
        "listening": np.zeros((h, w), dtype=float),
    }

    for rec, pid in REC_TO_PID.items():
        if pid in EXCLUDED:
            continue
        tsv = GAZE_DIR / f"Furhat-eye-tracking {rec}.tsv"
        fix = load_mapped_fixation_samples(tsv)
        prep = preprocess_recording(rec, events_by_rec[rec])

        ts = fix["timestamp_s"].to_numpy()
        xs = fix["x"].to_numpy()
        ys = fix["y"].to_numpy()

        m_speak = mask_in_windows(ts, prep.participant_turns)
        m_listen = mask_in_windows(ts, prep.robot_turns)

        heats = {
            "all":       compute_heatmap(xs, ys, w, h),
            "speaking":  compute_heatmap(xs[m_speak], ys[m_speak], w, h),
            "listening": compute_heatmap(xs[m_listen], ys[m_listen], w, h),
        }
        per_participant[pid] = heats
        for role in pooled:
            pooled[role] += heats[role]

        print(f"{pid}: {len(fix)} mapped fixation samples "
              f"(speaking={int(m_speak.sum())}, listening={int(m_listen.sum())})")

    # --- Pooled figures ------------------------------------------------------
    for role, heat in pooled.items():
        fig, ax = plt.subplots(figsize=(10, 10 * h / w))
        overlay_on_ax(ax, snap, heat, "")
        fig.tight_layout()
        out = ROOT / "figures" / f"heatmap_{role}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out.name}")

    # --- Per-participant 2x3 grid (overall) ----------------------------------
    pids = [p for p in REC_TO_PID.values() if p not in EXCLUDED]
    fig, axes = plt.subplots(2, 3, figsize=(15, 2 * 5 * h / w))
    axes = axes.ravel()
    for ax, pid in zip(axes, pids):
        overlay_on_ax(ax, snap, per_participant[pid]["all"], pid)
    for ax in axes[len(pids):]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.2, top=0.95, bottom=0.02, left=0.02, right=0.98)
    out = ROOT / "figures" / "heatmap_per_participant.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


if __name__ == "__main__":
    main()
