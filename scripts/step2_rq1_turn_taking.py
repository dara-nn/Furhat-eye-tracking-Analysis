"""Step 2 — RQ1: turn-taking regulation.

Canonical terms in the report: RQ1(a) is *look-away at turn start*,
RQ1(b) is *look-back at turn end*. Internal column names in this module
and in the results CSV use ``is_claim`` (RQ1(a)) and ``is_yield`` (RQ1(b))
for historical reasons. The two naming schemes refer to the same metrics.

Computes the two RQ1 gaze proportions for one preprocessed recording:

- **``is_claim`` (look-away at turn start)** — did a ``robot → outside``
  gaze transition happen in a window around ``participant_turn_start``?
- **``is_yield`` (look-back at turn end)** — did an ``outside → robot``
  gaze transition happen in a window around ``participant_turn_end``?

Both metrics are *event-based* and *binary per turn*. A transition counts only
if the two flanking state groups each have duration ≥ ``MIN_GROUP_S`` (this
filters out micro-saccades and noise). A participant's score is the mean of
these per-turn 0/1 values.

The primary window is ±1 s; ±0.5 s is kept as a sensitivity check.

AOI state per sample (after dropping invalid-tracking samples):

- ``robot`` = Face AOI hit (covers Eyes/Mouth/Nose by AOI nesting)
- ``outside`` = everything else (Body, Outside, no AOI hit)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from step1_preprocess import PreprocessedRecording

# --- Parameters --------------------------------------------------------------

MIN_GROUP_S = 0.2     # groups < 200 ms are discarded as noise
W_PRIMARY = 1.0       # primary window half-width (±1 s)
W_SECONDARY = 0.5     # secondary window half-width (±0.5 s), sensitivity check


# --- Public output -----------------------------------------------------------


@dataclass
class Rq1Result:
    pid: str
    # primary ±1 s
    claim_pcts: list[int] = field(default_factory=list)
    yield_pcts: list[int] = field(default_factory=list)
    # secondary ±0.5 s (kept for sensitivity reporting)
    claim_pcts_w500: list[int] = field(default_factory=list)
    yield_pcts_w500: list[int] = field(default_factory=list)


# --- State-group construction -----------------------------------------------


def build_state_groups(samples: pd.DataFrame, min_group_s: float) -> pd.DataFrame:
    """Collapse valid samples into contiguous (robot | outside) groups.

    Invalid-tracking samples are dropped before grouping. Groups with duration
    < ``min_group_s`` are discarded.
    """
    s = samples[samples["valid"]].sort_values("timestamp_s").reset_index(drop=True)
    state = np.where(s["on_robot"], "robot", "outside")
    run_id = (pd.Series(state) != pd.Series(state).shift()).cumsum()
    g = (
        s.groupby(run_id)
        .agg(
            state=("on_robot", lambda col: "robot" if col.iloc[0] else "outside"),
            t_start=("timestamp_s", "min"),
            t_end=("timestamp_s", "max"),
        )
        .reset_index(drop=True)
    )
    g["duration"] = g["t_end"] - g["t_start"]
    if min_group_s > 0:
        g = g[g["duration"] >= min_group_s].reset_index(drop=True)
    return g


def find_transition(
    groups: pd.DataFrame, t_anchor: float, W: float, from_state: str, to_state: str
) -> int:
    """Return 1 if an adjacent (from_state → to_state) pair has its transition
    timestamp (midpoint of t_end and next t_start) inside [t_anchor-W, t_anchor+W]."""
    lo, hi = t_anchor - W, t_anchor + W
    for i in range(len(groups) - 1):
        a, b = groups.iloc[i], groups.iloc[i + 1]
        if a["state"] == from_state and b["state"] == to_state:
            transition_t = (a["t_end"] + b["t_start"]) / 2
            if lo <= transition_t <= hi:
                return 1
    return 0


# --- Step entry point --------------------------------------------------------


def compute_rq1(prep: PreprocessedRecording) -> Rq1Result:
    groups = build_state_groups(prep.samples, MIN_GROUP_S)

    res = Rq1Result(pid=prep.pid)

    for t_start, t_end in prep.participant_turns:
        # Every turn is scored. Denominator = participant's total turn starts
        # (for RQ1(a)) and total turn ends (for RQ1(b)). If the ±W window
        # extends past the recording edge, the clipped portion contains no
        # transitions and the turn scores 0.
        res.claim_pcts.append(find_transition(groups, t_start, W_PRIMARY, "robot", "outside"))
        res.claim_pcts_w500.append(find_transition(groups, t_start, W_SECONDARY, "robot", "outside"))
        res.yield_pcts.append(find_transition(groups, t_end, W_PRIMARY, "outside", "robot"))
        res.yield_pcts_w500.append(find_transition(groups, t_end, W_SECONDARY, "outside", "robot"))

    return res
