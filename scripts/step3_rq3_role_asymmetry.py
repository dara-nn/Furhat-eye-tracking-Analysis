"""Step 3 — RQ2: listening vs speaking on-robot percentages.

Filename retains the legacy ``rq3`` token; the research question itself is
RQ2 in the current report. Internal column names (``rq3_speaking_pct``,
``rq3_listening_pct``) are kept for CSV compatibility.

Computes, per participant, the fraction of listening vs speaking time spent
looking at the robot's face.

Definition (pooled across turns, wall-clock denominator):

- **Listening windows** = each collapsed robot turn ``[r_start, r_end]``.
- **Speaking windows**  = each collapsed participant turn ``[p_start, p_end]``.
  Silences between a participant turn ending and the next robot turn
  starting are excluded — they belong to neither role.
- **State per sample** (same as RQ1): ``robot`` if the Face AOI was hit,
  otherwise ``outside``. Face is a nested AOI, so Eyes / Mouth / Nose hits
  also register as Face. Invalid-tracking samples (all AOI columns blank)
  are dropped.
- **Robot-state groups** = contiguous runs of same-state samples. No
  200 ms filter (unlike RQ1) — we want total time, not event detection.
- **Metric** = (sum of robot-state group durations overlapping the role's
  windows) / (sum of window wall-clock durations).

Tracking gaps (dropped invalid samples) implicitly count against the robot
proportion: they add to the wall-clock denominator but not to the numerator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from step1_preprocess import PreprocessedRecording


# --- Public output -----------------------------------------------------------


@dataclass
class Rq3Result:
    pid: str
    listen_at_robot: float    # seconds summed across listening windows
    listen_total: float       # seconds (wall-clock) of listening windows
    speak_at_robot: float
    speak_total: float
    n_listen_turns: int
    n_speak_turns: int

    @property
    def listen_pct(self) -> float:
        return self.listen_at_robot / self.listen_total if self.listen_total > 0 else float("nan")

    @property
    def speak_pct(self) -> float:
        return self.speak_at_robot / self.speak_total if self.speak_total > 0 else float("nan")


# --- Helpers -----------------------------------------------------------------


def build_robot_groups(samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Collapse valid samples into contiguous robot/outside groups, return
    (t_start, t_end) arrays for the robot-state groups only."""
    s = samples[samples["valid"]].sort_values("timestamp_s").reset_index(drop=True)
    if len(s) == 0:
        return np.array([]), np.array([])
    state = np.where(s["on_robot"], "robot", "outside")
    run_id = (pd.Series(state) != pd.Series(state).shift()).cumsum()
    g = (
        s.groupby(run_id)
        .agg(
            state=("on_robot", lambda c: "robot" if c.iloc[0] else "outside"),
            t_start=("timestamp_s", "min"),
            t_end=("timestamp_s", "max"),
        )
        .reset_index(drop=True)
    )
    robot = g[g["state"] == "robot"]
    return robot["t_start"].to_numpy(), robot["t_end"].to_numpy()


def sum_overlap_with_windows(
    rg_start: np.ndarray, rg_end: np.ndarray, windows: list[tuple[float, float]]
) -> float:
    """Sum the overlap between each robot group and each window."""
    total = 0.0
    for w_start, w_end in windows:
        if len(rg_start) == 0:
            continue
        ov = np.clip(np.minimum(rg_end, w_end) - np.maximum(rg_start, w_start), 0.0, None)
        total += float(ov.sum())
    return total


# --- Step entry point --------------------------------------------------------


def compute_rq3(prep: PreprocessedRecording) -> Rq3Result:
    rg_start, rg_end = build_robot_groups(prep.samples)

    listen_total = sum(e - s for s, e in prep.robot_turns)
    speak_total  = sum(e - s for s, e in prep.participant_turns)

    listen_at_robot = sum_overlap_with_windows(rg_start, rg_end, prep.robot_turns)
    speak_at_robot  = sum_overlap_with_windows(rg_start, rg_end, prep.participant_turns)

    return Rq3Result(
        pid=prep.pid,
        listen_at_robot=listen_at_robot,
        listen_total=listen_total,
        speak_at_robot=speak_at_robot,
        speak_total=speak_total,
        n_listen_turns=len(prep.robot_turns),
        n_speak_turns=len(prep.participant_turns),
    )
