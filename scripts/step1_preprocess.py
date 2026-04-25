"""Step 1 — Load raw data, label each gaze sample with AOI, collapse turns.

This step is everything that happens before either research question. It
reads a recording's gaze TSV, annotates each sample as on-robot or looking
away according to the Face / Outside+Body AOI rules, collapses the raw
transcript segment boundaries into complete speaker turns, and builds the
fixation-event table used by RQ2.

The result is a :class:`PreprocessedRecording` — a single object that Steps
2–4 consume without re-reading any raw files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --- Project constants -------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
EVENTS_CSV = ROOT / "data" / "turn_events.csv"
GAZE_DIR = ROOT / "data" / "gaze-data-trimmed"

REC_TO_PID = {f"Recording {i}": f"P{i}" for i in range(1, 7)}

# AOI semantics:
#   "at robot" = gaze landed on the robot's Face (Face is nested, so it
#                automatically covers Eyes / Mouth / Nose).
#   "looking away" = gaze landed on Outside OR Body.
AOI_FACE_COL = "AOI hit [snap - Face]"
AOI_BODY_COL = "AOI hit [snap - Body]"
AOI_OUTSIDE_COL = "AOI hit [snap - Outside]"
AOI_ALL_COLS = [
    "AOI hit [snap - Body]",
    "AOI hit [snap - Eyes]",
    "AOI hit [snap - Face]",
    "AOI hit [snap - Mouth]",
    "AOI hit [snap - Nose]",
    AOI_OUTSIDE_COL,
]


# --- Public output -----------------------------------------------------------


@dataclass
class PreprocessedRecording:
    rec_name: str                                  # e.g. "Recording 1"
    pid: str                                       # e.g. "P1"
    samples: pd.DataFrame
    fixations: pd.DataFrame
    robot_turns: list[tuple[float, float]]
    participant_turns: list[tuple[float, float]]
    recording_start_s: float
    recording_end_s: float


# --- Loaders -----------------------------------------------------------------


def load_events(csv_path: Path = EVENTS_CSV) -> dict[str, list[tuple[str, float]]]:
    """Return {recording_name: [(event, t_s), ...]} sorted by timestamp."""
    events: dict[str, list[tuple[str, float]]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = row["Recording name"]
            events.setdefault(rec, []).append(
                (row["Event"], int(row["Timestamp [\u03bcs]"]) / 1_000_000)
            )
    for rec in events:
        events[rec].sort(key=lambda x: x[1])
    return events


def load_gaze_samples(tsv_path: Path) -> pd.DataFrame:
    """Read a Tobii TSV and attach valid / on_robot / on_away columns."""
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    keep = [
        "Recording timestamp",
        "Eye movement type",
        "Eye movement event duration",
        "Eye movement type index",
        *AOI_ALL_COLS,
    ]
    df = df[keep].copy()
    df["timestamp_s"] = df["Recording timestamp"].astype(float) / 1_000_000
    aoi = df[AOI_ALL_COLS].apply(pd.to_numeric, errors="coerce")
    df["valid"] = aoi.notna().any(axis=1)
    df["on_robot"] = (aoi[AOI_FACE_COL].fillna(0) == 1) & df["valid"]
    df["on_away"] = (
        (aoi[AOI_OUTSIDE_COL].fillna(0) == 1) | (aoi[AOI_BODY_COL].fillna(0) == 1)
    ) & df["valid"]
    return df


def build_fixation_events(samples: pd.DataFrame) -> pd.DataFrame:
    """Collapse samples into one row per fixation (keyed by Eye movement type index)."""
    fix = samples[samples["Eye movement type"] == "Fixation"].copy()
    fix["Eye movement type index"] = pd.to_numeric(
        fix["Eye movement type index"], errors="coerce"
    )
    fix = fix.dropna(subset=["Eye movement type index"])
    grouped = fix.groupby("Eye movement type index", sort=True)

    rows = []
    for idx, g in grouped:
        g_valid = g[g["valid"]]
        if len(g_valid) == 0:
            continue
        on_robot = bool(g_valid["on_robot"].sum() > (len(g_valid) / 2))
        fix_start = float(g["timestamp_s"].min())
        duration_ms = float(g["Eye movement event duration"].iloc[0])
        duration_s = duration_ms / 1000.0
        rows.append(
            {
                "fix_index": int(idx),
                "fix_start_s": fix_start,
                "duration_s": duration_s,
                "fix_end_s": fix_start + duration_s,
                "on_robot": on_robot,
            }
        )
    return pd.DataFrame(rows)


# --- Turn collapse -----------------------------------------------------------


def collapse_turns(
    events: list[tuple[str, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Merge consecutive same-speaker segments into (start, end) turns.

    Returns (robot_turns, participant_turns).
    """
    robot_turns: list[tuple[float, float]] = []
    participant_turns: list[tuple[float, float]] = []

    current_speaker: str | None = None
    current_start: float | None = None
    current_end: float | None = None

    for et, t in events:
        speaker = "robot" if et.startswith("robot") else "participant"
        if et.endswith("_start"):
            if current_speaker != speaker:
                if current_speaker is not None:
                    (robot_turns if current_speaker == "robot" else participant_turns).append(
                        (current_start, current_end)
                    )
                current_speaker = speaker
                current_start = t
            current_end = t
        elif et.endswith("_end"):
            if current_speaker == speaker:
                current_end = t
            else:
                if current_speaker is not None:
                    (robot_turns if current_speaker == "robot" else participant_turns).append(
                        (current_start, current_end)
                    )
                current_speaker = speaker
                current_start = t
                current_end = t

    if current_speaker is not None:
        (robot_turns if current_speaker == "robot" else participant_turns).append(
            (current_start, current_end)
        )

    return robot_turns, participant_turns


# --- Step entry point --------------------------------------------------------


def preprocess_recording(
    rec_name: str, events_for_rec: list[tuple[str, float]]
) -> PreprocessedRecording:
    """Run the full Step 1 pipeline for one recording."""
    pid = REC_TO_PID[rec_name]
    tsv = GAZE_DIR / f"Furhat-eye-tracking {rec_name}.tsv"
    samples = load_gaze_samples(tsv)
    fixations = build_fixation_events(samples)
    robot_turns, participant_turns = collapse_turns(events_for_rec)
    return PreprocessedRecording(
        rec_name=rec_name,
        pid=pid,
        samples=samples,
        fixations=fixations,
        robot_turns=robot_turns,
        participant_turns=participant_turns,
        recording_start_s=float(samples["timestamp_s"].min()),
        recording_end_s=float(samples["timestamp_s"].max()),
    )
