"""Output-contract tests.

The load-bearing one is that `pass_by` can never reach the deliverable. It is a
ground-truth and debug-artifact label only; emitting it in outputs/<clip>.json
would mean the system reported a non-interaction as an interaction.
"""
import json

import pytest

from pvi.schema import (GT_TYPES, INTERACTION_TYPES, ClipMeta, Interaction,
                        PersonRef, VehicleRef, load_ground_truth, to_json,
                        write_output)


def make(type_="enter_vehicle", start=10, end=20):
    return Interaction(
        interaction_id="c__i001", event_group_id="c__g001", type=type_,
        frame_start=start, frame_end=end, time_start_s=1.0, time_end_s=2.0,
        person=PersonRef(track_id=1, description="adult, red shirt"),
        vehicle=VehicleRef(track_id=2, cls="car", description="blue sedan"),
    )


def test_pass_by_cannot_be_constructed_as_an_interaction():
    with pytest.raises(ValueError, match="not a reportable interaction type"):
        make("pass_by")


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError):
        make("teleport_into_vehicle")


def test_inverted_span_is_rejected():
    with pytest.raises(ValueError, match="frame_end"):
        make(start=30, end=10)


def test_single_frame_span_is_allowed():
    assert make(start=10, end=10).frame_end == 10


def test_pass_by_is_a_gt_type_but_not_an_interaction_type():
    assert "pass_by" in GT_TYPES
    assert "pass_by" not in INTERACTION_TYPES


def test_json_renames_cls_to_class():
    out = to_json(ClipMeta("c", 640, 480, 30.0, 100), [make()], "sha256:abc")
    assert out["interactions"][0]["vehicle"]["class"] == "car"
    assert "cls" not in out["interactions"][0]["vehicle"]


def test_json_carries_the_four_required_fields():
    """The task requires clip id, time span, person description and vehicle
    description on every record."""
    out = to_json(ClipMeta("c", 640, 480, 30.0, 100), [make()], "sha256:abc")
    rec = out["interactions"][0]
    assert out["clip_id"] == "c"
    assert {"frame_start", "frame_end", "time_start_s", "time_end_s"} <= set(rec)
    assert rec["person"]["description"] and rec["vehicle"]["description"]


def test_write_output_roundtrips(tmp_path):
    p = tmp_path / "out.json"
    write_output(p, ClipMeta("c", 640, 480, 30.0, 100), [make()], "sha256:abc")
    assert json.loads(p.read_text())["config_hash"] == "sha256:abc"


# --- ClipMeta conversions ---

def test_seconds_frames_roundtrip_at_low_fps():
    m = ClipMeta("c", 1280, 720, 6.0, 108)
    assert m.to_frames(1.0) == 6
    assert m.to_s(6) == pytest.approx(1.0)


def test_to_frames_rounds_rather_than_truncates():
    """0.4 s at 6.667 fps is 2.67 frames. Truncating loses a third of the
    window, which matters at the 5-frame shortest event."""
    m = ClipMeta("c", 960, 720, 6.667, 182)
    assert m.to_frames(0.4) == 3


def test_duration_matches_the_inventory():
    m = ClipMeta("c", 1280, 720, 6.0, 108)
    assert m.duration_s == pytest.approx(18.0)


# --- ground truth ---

def test_frozen_ground_truth_loads_and_has_the_documented_counts():
    """Pins the frozen GT: 25 events, 18 positives, 7 pass_by. If this fails,
    the evaluation substrate changed and every reported number is stale."""
    gts = load_ground_truth("data/ground_truth.json")
    assert len(gts) == 25
    assert sum(1 for g in gts if g.is_positive) == 18
    assert sum(1 for g in gts if not g.is_positive) == 7


def test_frozen_ground_truth_type_counts():
    gts = load_ground_truth("data/ground_truth.json")
    counts = {t: sum(1 for g in gts if g.type == t) for t in GT_TYPES}
    assert counts == {"exit_vehicle": 7, "enter_vehicle": 6, "attend_vehicle": 2,
                      "open_close_door": 2, "load_unload": 1, "pass_by": 7}


def test_shortest_positive_is_the_documented_075_seconds():
    """config.SHORTEST_POSITIVE_S drives the min_dwell_s constraint; if the GT
    disagrees with it the constraint is guarding the wrong number."""
    from pvi.config import SHORTEST_POSITIVE_S
    gts = [g for g in load_ground_truth("data/ground_truth.json") if g.is_positive]
    shortest = min(g.time_end_s - g.time_start_s for g in gts)
    assert shortest == pytest.approx(SHORTEST_POSITIVE_S, abs=0.05)


def test_every_gt_event_spans_at_least_one_frame():
    for g in load_ground_truth("data/ground_truth.json"):
        assert g.frame_end >= g.frame_start, g.event_id


def test_gt_event_ids_are_unique():
    """A 2026-09-14 audit found 14 duplicate ids from a bug in label_gt.py.
    Pinned so a regression cannot silently return."""
    gts = load_ground_truth("data/ground_truth.json")
    assert len({g.event_id for g in gts}) == len(gts)
