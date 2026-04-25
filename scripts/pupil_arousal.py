"""Pupil diameter as arousal signal: role comparison + engagement coupling.

Two questions:
  1. Does mean pupil differ between participant-speaking and robot-speaking
     (listening) windows? We know from the raw timeline that speaking >
     listening on average; this quantifies the gap per participant.
  2. Is pupil larger when the participant is looking at the robot (on-robot,
     AOI Face hit) than when looking away? If yes, engagement -> arousal.
     We split this further by role to see whether the coupling holds
     separately during speaking and listening.

Samples included: both validities == "Valid" AND Pupil diameter filtered
not NaN. No baseline correction; reports raw filtered pupil in mm.

Outputs:
  figures/pupil_arousal.png   two-panel figure (role; role x gaze-state).
  results/pupil_arousal.csv   per-(pid, role, gaze) mean / SD / n.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step1_preprocess import (
    EVENTS_CSV,
    GAZE_DIR,
    REC_TO_PID,
    collapse_turns,
    load_events,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_FIG = ROOT / "figures" / "pupil_arousal.png"
OUT_CSV = ROOT / "results" / "pupil_arousal.csv"

EXCLUDED = {"P5"}
AOI_FACE = "AOI hit [snap - Face]"

C_SPEAK = "#fc8d62"
C_LISTEN = "#66c2a5"
C_SPEAK_OFF = "#fdbf6f"
C_LISTEN_OFF = "#a6d96a"


def in_windows(ts: np.ndarray, wins: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(len(ts), dtype=bool)
    for a, b in wins:
        m |= (ts >= a) & (ts < b)
    return m


def load_pupil_and_face(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        low_memory=False,
        decimal=",",
        usecols=[
            "Recording timestamp",
            "Pupil diameter filtered",
            "Validity left",
            "Validity right",
            AOI_FACE,
        ],
    )
    df["timestamp_s"] = df["Recording timestamp"].astype(float) / 1_000_000
    df["pupil_mm"] = pd.to_numeric(df["Pupil diameter filtered"], errors="coerce")
    valid = (
        (df["Validity left"] == "Valid")
        & (df["Validity right"] == "Valid")
        & df["pupil_mm"].notna()
    )
    df = df[valid].copy()
    face = pd.to_numeric(df[AOI_FACE], errors="coerce").fillna(0).astype(int)
    df["on_robot"] = face == 1
    return df[["timestamp_s", "pupil_mm", "on_robot"]].reset_index(drop=True)


def summarize(pupil: np.ndarray) -> tuple[float, float, int]:
    if len(pupil) == 0:
        return float("nan"), float("nan"), 0
    return float(pupil.mean()), float(pupil.std(ddof=1)) if len(pupil) > 1 else 0.0, int(len(pupil))


def main() -> None:
    events_by_rec = load_events(EVENTS_CSV)

    rows: list[dict] = []
    for rec, pid in REC_TO_PID.items():
        if pid in EXCLUDED:
            continue
        robot_turns, part_turns = collapse_turns(events_by_rec[rec])
        df = load_pupil_and_face(GAZE_DIR / f"Furhat-eye-tracking {rec}.tsv")
        ts = df["timestamp_s"].to_numpy()
        in_speak = in_windows(ts, part_turns)
        in_listen = in_windows(ts, robot_turns)
        on_robot = df["on_robot"].to_numpy()
        pupil = df["pupil_mm"].to_numpy()

        for role_name, m_role in (("speaking", in_speak), ("listening", in_listen)):
            mean, sd, n = summarize(pupil[m_role])
            rows.append(dict(pid=pid, role=role_name, gaze="any", mean=mean, sd=sd, n=n))
            for gaze_name, m_gaze in (("on_robot", on_robot), ("off_robot", ~on_robot)):
                sel = m_role & m_gaze
                mean, sd, n = summarize(pupil[sel])
                rows.append(dict(pid=pid, role=role_name, gaze=gaze_name,
                                  mean=mean, sd=sd, n=n))

    summary = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)

    pids = sorted(summary["pid"].unique())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # --- Panel 1: role comparison ---
    d1 = summary[summary["gaze"] == "any"].pivot(index="pid", columns="role", values="mean")
    sd1 = summary[summary["gaze"] == "any"].pivot(index="pid", columns="role", values="sd")
    x = np.arange(len(pids))
    w = 0.38
    ax1.bar(x - w/2, d1.loc[pids, "speaking"],  w,
            yerr=sd1.loc[pids, "speaking"],  label="Speaking",  color=C_SPEAK,
            error_kw=dict(lw=0.8, ecolor="#555"))
    ax1.bar(x + w/2, d1.loc[pids, "listening"], w,
            yerr=sd1.loc[pids, "listening"], label="Listening", color=C_LISTEN,
            error_kw=dict(lw=0.8, ecolor="#555"))
    ax1.set_xticks(x)
    ax1.set_xticklabels(pids)
    ax1.set_ylabel("Mean pupil diameter (mm)")
    speak_mean = d1["speaking"].mean()
    listen_mean = d1["listening"].mean()
    ax1.axhline(speak_mean, ls=":", lw=0.8, color=C_SPEAK)
    ax1.axhline(listen_mean, ls=":", lw=0.8, color=C_LISTEN)
    ax1.legend(frameon=False, loc="upper right")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)

    # --- Panel 2: role x gaze-state ---
    d2 = summary[summary["gaze"] != "any"].pivot_table(
        index="pid", columns=["role", "gaze"], values="mean"
    )
    groups = [
        (("speaking", "on_robot"),  "Speaking - on-robot",  C_SPEAK),
        (("speaking", "off_robot"), "Speaking - away", C_SPEAK_OFF),
        (("listening", "on_robot"), "Listening - on-robot", C_LISTEN),
        (("listening", "off_robot"),"Listening - away",C_LISTEN_OFF),
    ]
    w2 = 0.20
    for i, (key, lab, col) in enumerate(groups):
        offset = (i - 1.5) * w2
        vals = d2.loc[pids, key]
        ax2.bar(x + offset, vals, w2, color=col, label=lab, edgecolor="white", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pids)
    ax2.set_ylabel("Mean pupil diameter (mm)")
    # Engagement delta annotation: (on - off) per role, averaged across participants
    delta_speak = (d2.xs(("speaking","on_robot"),axis=1) - d2.xs(("speaking","off_robot"),axis=1)).mean()
    delta_listen = (d2.xs(("listening","on_robot"),axis=1) - d2.xs(("listening","off_robot"),axis=1)).mean()
    ax2.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.set_axisbelow(True)

    # Tight y-axis around the data, both panels use the same lower bound
    all_means = summary[summary["mean"].notna()]["mean"]
    ymin = max(0.0, all_means.min() - 0.3)
    ymax = all_means.max() + 0.5
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_FIG.name}, {OUT_CSV.name}")


if __name__ == "__main__":
    main()
