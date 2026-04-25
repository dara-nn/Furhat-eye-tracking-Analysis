"""Permutation baseline test — random 2 s windows vs turn-anchored RQ1.

For each participant, draw N random 2 s windows from their recording (uniform
over the recording's interior) and compute the same binary event metrics as
RQ1 on those random windows:

- random-claim = % of random windows with an adjacent ``robot → outside``
  state-group pair whose transition timestamp falls inside the window.
- random-yield = same, but for ``outside → robot``.

Compares to the turn-anchored RQ1(a) / RQ1(b) numbers in
``results_per_participant.csv``. If the turn-anchored number is meaningfully
above the random baseline, the effect is specific to turn boundaries; if it
is indistinguishable, the metric is picking up baseline gaze activity.

Reproduces the numbers quoted in the discussion of ``research-report.md``.
Uses the same parameters as the main RQ1 pipeline (filter 200 ms, ±1 s
window). Set ``SEED`` for determinism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from step1_preprocess import load_events, preprocess_recording, EVENTS_CSV, REC_TO_PID, ROOT
from step2_rq1_turn_taking import build_state_groups, find_transition, MIN_GROUP_S, W_PRIMARY

N_SAMPLES = 100
SEED = 42


def main() -> None:
    events_by_rec = load_events(EVENTS_CSV)
    results = pd.read_csv(ROOT / "results" / "results_per_participant.csv").set_index("participant")

    rng = np.random.default_rng(SEED)

    pids = [p for p in REC_TO_PID.values() if p != "P5"]
    rows = []

    for rec, pid in REC_TO_PID.items():
        if pid not in pids:
            continue
        prep = preprocess_recording(rec, events_by_rec[rec])
        groups = build_state_groups(prep.samples, MIN_GROUP_S)

        lo_t = prep.recording_start_s + W_PRIMARY
        hi_t = prep.recording_end_s - W_PRIMARY
        anchors = rng.uniform(lo_t, hi_t, size=N_SAMPLES)

        rand_claim = np.mean(
            [find_transition(groups, a, W_PRIMARY, "robot", "outside") for a in anchors]
        )
        rand_yield = np.mean(
            [find_transition(groups, a, W_PRIMARY, "outside", "robot") for a in anchors]
        )
        real_claim = float(results.loc[pid, "rq1_claim_pct"])
        real_yield = float(results.loc[pid, "rq1_yield_pct"])

        rows.append(
            {
                "pid": pid,
                "rq1_claim_real": real_claim,
                "rq1_claim_random": rand_claim,
                "claim_delta": real_claim - rand_claim,
                "rq1_yield_real": real_yield,
                "rq1_yield_random": rand_yield,
                "yield_delta": real_yield - rand_yield,
            }
        )

    df = pd.DataFrame(rows)
    print(f"Random 2 s baseline vs turn-anchored RQ1 (N_random={N_SAMPLES}, seed={SEED})\n")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(df.to_string(index=False))
    print("\nMeans:")
    for col in ["rq1_claim_real", "rq1_claim_random", "claim_delta",
                "rq1_yield_real", "rq1_yield_random", "yield_delta"]:
        print(f"  {col:24s} mean={df[col].mean():+.3f}  sd={df[col].std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
