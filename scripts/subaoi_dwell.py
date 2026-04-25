"""Sub-AOI fixation time — break the "on robot" category into Eyes / Mouth /
Nose / (Face-only = Face hit but none of E/M/N) / Body / Outside.

Numerator matches Tobii Pro Lab's default "Total duration of fixations"
metric: only samples with ``Eye movement type == "Fixation"`` contribute.
Saccades, EyesNotFound, and Unclassified rows are excluded, mirroring
Pro Lab's I-VT-filter behavior.

For each participant, pool all fixation samples and report the fraction
whose gaze fell on each sub-AOI. Also split by role:

- Speaking   = fixation sample timestamp inside a participant turn
- Listening  = fixation sample timestamp inside a robot turn
- Overall    = all fixation samples regardless of role

AOI nesting: Face is a superset of Eyes / Mouth / Nose. To avoid
double-counting, sub-AOI columns are made *mutually exclusive*:

- Eyes       = Eyes AOI hit
- Mouth      = Mouth AOI hit and not Eyes
- Nose       = Nose AOI hit and not Eyes and not Mouth
- Face_other = Face AOI hit and none of Eyes / Mouth / Nose
- Body       = Body AOI hit and not Face
- Outside    = Outside AOI hit and not Face and not Body
- None       = fixation-classified sample with no AOI hit (genuine gap
               between AOIs — not tracking loss, which was filtered out)

Runs on the same N=5 cohort as the main pipeline (excluding P5) and writes
``results/subaoi_dwell_per_participant.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from step1_preprocess import (
    EVENTS_CSV,
    REC_TO_PID,
    ROOT,
    load_events,
    preprocess_recording,
)

EXCLUDED = {"P5"}

AOI_COLS = {
    "eyes":    "AOI hit [snap - Eyes]",
    "mouth":   "AOI hit [snap - Mouth]",
    "nose":    "AOI hit [snap - Nose]",
    "face":    "AOI hit [snap - Face]",
    "body":    "AOI hit [snap - Body]",
    "outside": "AOI hit [snap - Outside]",
}


def classify_subaoi(df: pd.DataFrame) -> pd.Series:
    """Return exclusive sub-AOI label per valid sample."""
    eyes  = df[AOI_COLS["eyes"]].fillna(0) == 1
    mouth = df[AOI_COLS["mouth"]].fillna(0) == 1
    nose  = df[AOI_COLS["nose"]].fillna(0) == 1
    face  = df[AOI_COLS["face"]].fillna(0) == 1
    body  = df[AOI_COLS["body"]].fillna(0) == 1
    outs  = df[AOI_COLS["outside"]].fillna(0) == 1

    lab = np.full(len(df), "none", dtype=object)
    lab[outs & ~face & ~body] = "outside"
    lab[body & ~face] = "body"
    lab[face & ~eyes & ~mouth & ~nose] = "face_other"
    lab[nose & ~eyes & ~mouth] = "nose"
    lab[mouth & ~eyes] = "mouth"
    lab[eyes] = "eyes"
    return pd.Series(lab, index=df.index)


def mask_in_windows(ts: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(len(ts), dtype=bool)
    for a, b in windows:
        m |= (ts >= a) & (ts < b)
    return m


def main() -> None:
    events_by_rec = load_events(EVENTS_CSV)

    labels = ["eyes", "mouth", "nose", "face_other", "body", "outside", "none"]
    rows = []

    for rec, pid in REC_TO_PID.items():
        if pid in EXCLUDED:
            continue
        prep = preprocess_recording(rec, events_by_rec[rec])
        s = prep.samples[
            prep.samples["valid"]
            & (prep.samples["Eye movement type"] == "Fixation")
        ].copy()
        s["subaoi"] = classify_subaoi(s)
        ts = s["timestamp_s"].to_numpy()

        # Sample period (seconds) — nominal 50 Hz; compute empirically to be
        # robust to any slight sample-rate drift in the recording.
        if len(ts) >= 2:
            sample_period_s = float(np.median(np.diff(ts)))
        else:
            sample_period_s = 0.02

        for role, windows in [
            ("overall", None),
            ("speaking", prep.participant_turns),
            ("listening", prep.robot_turns),
        ]:
            sub = s if windows is None else s[mask_in_windows(ts, windows)]
            total = len(sub)
            if total == 0:
                continue
            counts = sub["subaoi"].value_counts()
            row = {
                "participant": pid,
                "role": role,
                "n_samples": total,
                "total_time_s": total * sample_period_s,
            }
            for lab in labels:
                c = int(counts.get(lab, 0))
                row[f"{lab}_pct"] = c / total
                row[f"{lab}_sec"] = c * sample_period_s
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = ROOT / "results" / "subaoi_dwell_per_participant.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    pcts = [f"{l}_pct" for l in labels]
    secs = [f"{l}_sec" for l in labels]
    print(f"Sub-AOI dwell (N=5 after excluding P5)\n")
    for role in ["overall", "speaking", "listening"]:
        sub = df[df["role"] == role].set_index("participant")
        print(f"--- {role} ---")
        print("% of valid samples:")
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
            print(sub[pcts].to_string())
        print(f"{'mean':<12}" + "  ".join(f"{sub[c].mean():.3f}" for c in pcts))
        print("\nTotal time in seconds:")
        with pd.option_context("display.float_format", "{:.1f}".format, "display.width", 160):
            print(sub[secs + ["total_time_s"]].to_string())
        print(f"{'mean':<12}" + "  ".join(f"{sub[c].mean():.1f}" for c in secs + ["total_time_s"]))
        print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
