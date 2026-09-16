"""Proposal-rule tests built on synthetic pair features.

The case that matters most is the pass-by: the set contains 7 labeled
near-misses and the whole R2/R3 design exists to keep them from firing. A test
suite that only checked positives would pass while the system was useless.
"""
import numpy as np
import pytest

from pvi.propose.features import PairFeatures, Track
from pvi.propose.rules import (R1_DWELL, R2_DEATH_NEAR, R3_BIRTH_NEAR,
                               R4_DOOR_CHANGE, primary_rule, propose_pair)
from pvi.schema import ClipMeta

FPS = 10.0
N = 100
META = ClipMeta(clip_id="t", width=640, height=480, fps=FPS, n_frames=N)


def pf(d_norm, door=None, person_id=1, static=True):
    return PairFeatures(
        person_id=person_id, vehicle_id=99, vehicle_cls=2,
        d_norm=np.array(d_norm, dtype=float),
        iou=np.zeros(N), speed_bl_s=np.zeros(N),
        visible=~np.isnan(np.array(d_norm, dtype=float)),
        door_delta=np.full(N, np.nan) if door is None else np.array(door, float),
        vehicle_is_static=static,
    )


def track(birth, death, tid=1):
    frames = list(range(birth, death + 1))
    return Track(id=tid, cls=0, frames=frames, boxes=[(0, 0, 10, 20)] * len(frames))


def far_series():
    return [5.0] * N


# --- R1 dwell ---

def test_r1_fires_on_sustained_contact():
    d = far_series()
    d[30:60] = [0.05] * 30     # 3.0 s of contact at 10 fps
    out = propose_pair(pf(d), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert len(out) == 1
    assert R1_DWELL in out[0].rules


def test_r1_does_not_fire_below_min_dwell():
    d = far_series()
    d[30:33] = [0.05] * 3      # 0.3 s, under a 0.5 s dwell
    out = propose_pair(pf(d), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert out == []


def test_r1_fires_on_the_shortest_gt_positive_duration():
    """0.75 s is the shortest positive in the frozen GT; the default dwell of
    0.5 s must let it through."""
    d = far_series()
    d[30:38] = [0.05] * 8      # 0.8 s
    out = propose_pair(pf(d), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert out and R1_DWELL in out[0].rules


# --- R2 / R3, the pass-by discriminators ---

def test_r2_fires_when_the_person_track_ends_at_the_vehicle():
    """Enter: the person stops being visible while in contact."""
    d = far_series()
    d[40:61] = [0.05] * 21
    for i in range(61, N):
        d[i] = np.nan          # track gone
    out = propose_pair(pf(d), track(0, 60), META, 0.15, 0.30, 0.5, 0.12)
    assert out and R2_DEATH_NEAR in out[0].rules


def test_r3_fires_when_the_person_track_starts_at_the_vehicle():
    """Exit: the person appears already in contact."""
    d = [np.nan] * 40 + [0.05] * 21 + [5.0] * (N - 61)
    out = propose_pair(pf(d), track(40, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert out and R3_BIRTH_NEAR in out[0].rules


def test_pass_by_does_not_fire_r2_or_r3():
    """The critical negative. A person walks past: briefly near, but born and
    dead far from the vehicle, and never in contact long enough for R1."""
    d = far_series()
    d[48:52] = [0.10] * 4      # 0.4 s brush past, under the 0.5 s dwell
    out = propose_pair(pf(d), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert out == []


def test_long_pass_by_fires_r1_but_not_r2_r3():
    """A slow loiterer still yields a candidate -- the proposer is recall-first
    and the judge is what rejects it -- but it must not be typed enter/exit."""
    d = far_series()
    d[30:60] = [0.10] * 30
    out = propose_pair(pf(d), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert out
    assert out[0].rules == [R1_DWELL]
    assert primary_rule(out[0]) == R1_DWELL


def test_track_born_in_contact_at_frame_zero_still_fires_r3():
    """Reversal of an earlier clip-edge guard, forced by the ground truth: 3 of
    the 7 exit_vehicle positives start at frame 0. Suppressing R3 there removed
    43% of the exit class. A track born already in contact with the vehicle is
    what an exit looks like."""
    d = [0.05] * 30 + far_series()[30:]
    out = propose_pair(pf(d), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert out and R3_BIRTH_NEAR in out[0].rules
    assert out[0].evidence["birth_at_clip_start"] is True


def test_track_dying_in_contact_at_the_last_frame_still_fires_r2():
    d = far_series()
    d[70:] = [0.05] * (N - 70)
    out = propose_pair(pf(d), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert out and R2_DEATH_NEAR in out[0].rules
    assert out[0].evidence["death_at_clip_end"] is True


def test_clip_edge_is_recorded_in_evidence_for_the_judge():
    """The proposer no longer decides this, so the fact must stay visible."""
    d = far_series()
    d[40:61] = [0.05] * 21
    out = propose_pair(pf(d), track(35, 60), META, 0.15, 0.30, 0.5, 0.12)
    assert out[0].evidence["birth_at_clip_start"] is False
    assert out[0].evidence["death_at_clip_end"] is False


def test_pass_by_still_does_not_fire_r2_r3_despite_no_edge_guard():
    """Removing the guard must not weaken the pass-by discrimination, which is
    what R2/R3 exist for."""
    d = far_series()
    d[48:52] = [0.10] * 4
    out = propose_pair(pf(d), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert out == []


# --- R4 door ---

def test_r4_fires_on_a_door_delta_spike_while_near():
    d = far_series()
    d[30:60] = [0.05] * 30
    door = np.full(N, 0.01)
    door[40:45] = 0.25
    out = propose_pair(pf(d, door), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert out and R4_DOOR_CHANGE in out[0].rules


def test_r4_is_skipped_when_door_delta_is_nan():
    """NaN is the moving-camera / moving-vehicle case. The rule must be skipped,
    not read as zero change."""
    d = far_series()
    d[30:60] = [0.05] * 30
    out = propose_pair(pf(d, None), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert out and R4_DOOR_CHANGE not in out[0].rules


def test_r4_does_not_fire_below_threshold():
    d = far_series()
    d[30:60] = [0.05] * 30
    door = np.full(N, 0.02)
    out = propose_pair(pf(d, door), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert out and R4_DOOR_CHANGE not in out[0].rules


# --- merging and priority ---

def test_hysteresis_yields_one_candidate_not_many():
    d = far_series()
    for i in range(30, 60):
        d[i] = 0.10 if i % 2 else 0.20   # oscillating inside the band
    out = propose_pair(pf(d), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    assert len(out) == 1


def test_primary_rule_prefers_enter_over_dwell():
    """R1 fires on nearly every true interaction, so if it won the priority the
    system would report attend_vehicle for everything."""
    d = far_series()
    d[40:61] = [0.05] * 21
    out = propose_pair(pf(d), track(0, 60), META, 0.15, 0.30, 0.5, 0.12)
    assert R1_DWELL in out[0].rules and R2_DEATH_NEAR in out[0].rules
    assert primary_rule(out[0]) == R2_DEATH_NEAR


def test_evidence_records_min_d_norm_and_dwell():
    d = far_series()
    d[30:60] = [0.05] * 30
    out = propose_pair(pf(d), track(0, N - 1), META, 0.15, 0.30, 0.5, 0.12)
    ev = out[0].evidence
    assert ev["min_d_norm"] == pytest.approx(0.05, abs=0.02)
    assert ev["dwell_s"] > 0.5


def test_no_contact_yields_no_candidates():
    assert propose_pair(pf(far_series()), track(0, N - 1), META,
                        0.15, 0.30, 0.5, 0.12) == []
