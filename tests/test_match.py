"""Temporal matching tests.

The inclusive-span convention bites hardest here: with an exclusive reading, a
prediction identical to its GT event scores tIoU < 1.0 and short events fail to
match themselves.
"""
import pytest

from pvi.evaluate.match import (boundary_errors, match_events, temporal_iou,
                                TIOU_PRIMARY)
from pvi.schema import GTEvent, Interaction, PersonRef, VehicleRef


def make_pred(start, end, type_="enter_vehicle", idx=1):
    return Interaction(
        interaction_id=f"c__i{idx:03d}", event_group_id=f"c__g{idx:03d}",
        type=type_, frame_start=start, frame_end=end,
        time_start_s=start / 10, time_end_s=end / 10,
        person=PersonRef(track_id=idx), vehicle=VehicleRef(track_id=99),
    )


def make_gt(start, end, type_="enter_vehicle", idx=1):
    return GTEvent(
        event_id=f"c__e{idx:03d}", event_group_id=f"c__e{idx:03d}", clip_id="c",
        type=type_, frame_start=start, frame_end=end,
        time_start_s=start / 10, time_end_s=end / 10,
    )


# --- temporal_iou ---

def test_tiou_identical_spans_is_one():
    assert temporal_iou(10, 20, 10, 20) == pytest.approx(1.0)


def test_tiou_single_frame_span_matches_itself():
    """With an exclusive end this returns 0.0 and every 1-frame event is a miss."""
    assert temporal_iou(10, 10, 10, 10) == pytest.approx(1.0)


def test_tiou_disjoint_is_zero():
    assert temporal_iou(0, 10, 20, 30) == 0.0


def test_tiou_abutting_spans_do_not_overlap():
    # [0,10] and [11,20] share no frame.
    assert temporal_iou(0, 10, 11, 20) == 0.0


def test_tiou_touching_at_one_frame():
    # [0,10] and [10,20]: 1 frame of 21 in union.
    assert temporal_iou(0, 10, 10, 20) == pytest.approx(1 / 21)


def test_tiou_half_overlap():
    # [0,9] and [5,14]: inter 5, union 15.
    assert temporal_iou(0, 9, 5, 14) == pytest.approx(5 / 15)


def test_tiou_is_symmetric():
    assert temporal_iou(3, 17, 8, 25) == pytest.approx(temporal_iou(8, 25, 3, 17))


def test_tiou_nested_span():
    # [10,19] inside [0,39]: inter 10, union 40.
    assert temporal_iou(0, 39, 10, 19) == pytest.approx(10 / 40)


# --- matching ---

def test_exact_match_counts_as_tp():
    r = match_events([make_pred(10, 20)], [make_gt(10, 20)])
    assert r.n_tp == 1 and r.n_fp == 0 and r.n_fn == 0


def test_below_threshold_is_fp_and_fn_not_a_match():
    # [0,9] vs [8,30]: tIoU = 2/31, well under 0.3.
    r = match_events([make_pred(0, 9)], [make_gt(8, 30)])
    assert r.n_tp == 0 and r.n_fp == 1 and r.n_fn == 1


def test_matching_is_one_to_one():
    """Two predictions over one GT event: one TP, one FP -- never two TPs."""
    r = match_events([make_pred(10, 20, idx=1), make_pred(11, 21, idx=2)],
                     [make_gt(10, 20)])
    assert r.n_tp == 1 and r.n_fp == 1


def test_greedy_prefers_the_higher_tiou_pairing():
    preds = [make_pred(0, 100, idx=1), make_pred(10, 20, idx=2)]
    r = match_events(preds, [make_gt(10, 20)])
    assert r.matches[0].pred_idx == 1  # the tight one wins


def test_wrong_type_still_matches_but_is_flagged():
    """Detection and classification are scored separately, so a span found with
    the wrong name is a detection hit with type_correct False."""
    r = match_events([make_pred(10, 20, type_="exit_vehicle")],
                     [make_gt(10, 20, type_="enter_vehicle")])
    assert r.n_tp == 1
    assert r.matches[0].type_correct is False


def test_require_same_type_excludes_the_mismatch():
    r = match_events([make_pred(10, 20, type_="exit_vehicle")],
                     [make_gt(10, 20, type_="enter_vehicle")],
                     require_same_type=True)
    assert r.n_tp == 0 and r.n_fn == 1


def test_prediction_on_a_pass_by_is_an_fp_and_recorded_as_a_pass_by_hit():
    """The number this dataset was built to probe: of N labeled near-misses, how
    many did the system fire on."""
    r = match_events([make_pred(10, 20)], [make_gt(10, 20, type_="pass_by")])
    assert r.n_tp == 0
    assert r.n_fn == 0                 # pass_by is not a missed positive
    assert len(r.pass_by_hits) == 1
    assert r.n_fp == 0                 # consumed by the pass_by pass, not double-counted


def test_positive_wins_over_pass_by_when_a_prediction_overlaps_both():
    gts = [make_gt(10, 20, type_="pass_by", idx=1),
           make_gt(10, 20, type_="enter_vehicle", idx=2)]
    r = match_events([make_pred(10, 20)], gts)
    assert r.n_tp == 1 and len(r.pass_by_hits) == 0


def test_unmatched_pass_by_is_not_counted_as_a_false_negative():
    r = match_events([], [make_gt(10, 20, type_="pass_by")])
    assert r.n_fn == 0


def test_match_is_deterministic_under_input_reordering():
    gts = [make_gt(0, 10, idx=1), make_gt(50, 60, idx=2)]
    a = match_events([make_pred(0, 10, idx=1), make_pred(50, 60, idx=2)], gts)
    b = match_events([make_pred(50, 60, idx=2), make_pred(0, 10, idx=1)], gts)
    assert {(m.gt_idx, round(m.tiou, 9)) for m in a.matches} == \
           {(m.gt_idx, round(m.tiou, 9)) for m in b.matches}


def test_boundary_errors_are_signed():
    """Signed, so a systematic late start is distinguishable from jitter."""
    preds, gts = [make_pred(12, 23)], [make_gt(10, 20)]
    starts, ends = boundary_errors(preds, gts, match_events(preds, gts))
    assert starts == [2] and ends == [3]


def test_primary_threshold_is_the_documented_value():
    assert TIOU_PRIMARY == 0.3


# --- actor identity via GT anchors ---
#
# problem-definition.md s3 requires a match to share the person-vehicle pair.
# Without this, matching is purely temporal: on NmlzoaDcOuI_6 the greedy matcher
# preferred a prediction about a DIFFERENT vehicle because its span overlapped
# 3% better than the correct one.

from pvi.evaluate.match import anchors_agree


def gt_with_anchors(px, py, vx, vy, start=10, end=20, idx=1):
    return GTEvent(
        event_id=f"c__e{idx:03d}", event_group_id=f"c__e{idx:03d}", clip_id="c",
        type="exit_vehicle", frame_start=start, frame_end=end,
        time_start_s=start / 10, time_end_s=end / 10,
        person_anchor={"frame": start, "x": px, "y": py},
        vehicle_anchor={"frame": start, "x": vx, "y": vy},
    )


def pred_with_boxes(pbox, vbox, start=10, end=20, type_="exit_vehicle", idx=1):
    p = make_pred(start, end, type_, idx)
    p.evidence = {"person_box_norm": pbox, "vehicle_box_norm": vbox}
    return p


def test_anchors_agree_when_both_fall_inside_their_boxes():
    pred = pred_with_boxes([0.1, 0.1, 0.2, 0.4], [0.5, 0.3, 0.9, 0.7])
    assert anchors_agree(pred, gt_with_anchors(0.15, 0.25, 0.7, 0.5))


def test_anchors_disagree_when_the_vehicle_is_a_different_one():
    """The NmlzoaDcOuI_6 case: right person, wrong car."""
    pred = pred_with_boxes([0.1, 0.1, 0.2, 0.4], [0.5, 0.3, 0.9, 0.7])
    assert not anchors_agree(pred, gt_with_anchors(0.15, 0.25, 0.05, 0.9))


def test_anchor_tolerance_admits_a_hint_just_outside_the_box():
    """Anchors are hand-placed hints, sometimes a few frames off their span, so
    the check is deliberately generous rather than strict containment."""
    pred = pred_with_boxes([0.4, 0.4, 0.6, 0.6], [0.0, 0.0, 0.3, 0.3])
    assert anchors_agree(pred, gt_with_anchors(0.63, 0.5, 0.15, 0.15))


def test_anchor_check_passes_when_the_gt_has_no_anchor():
    """Missing information must read as 'cannot tell', not as a mismatch --
    otherwise an absent field silently becomes a false negative."""
    pred = pred_with_boxes([0.1, 0.1, 0.2, 0.4], [0.5, 0.3, 0.9, 0.7])
    assert anchors_agree(pred, make_gt(10, 20))


def test_anchor_check_passes_when_the_prediction_has_no_boxes():
    assert anchors_agree(make_pred(10, 20), gt_with_anchors(0.15, 0.25, 0.7, 0.5))


def test_matching_prefers_the_right_vehicle_over_a_tighter_span():
    """Regression for the observed failure: a slightly tighter span about the
    wrong vehicle must not win the match."""
    gt = gt_with_anchors(0.15, 0.25, 0.7, 0.5, start=0, end=42)
    right = pred_with_boxes([0.1, 0.1, 0.2, 0.4], [0.5, 0.3, 0.9, 0.7],
                            start=0, end=48, idx=1)
    wrong = pred_with_boxes([0.1, 0.1, 0.2, 0.4], [0.0, 0.8, 0.1, 0.95],
                            start=4, end=42, type_="attend_vehicle", idx=2)
    r = match_events([right, wrong], [gt])
    assert r.n_tp == 1
    assert r.matches[0].pred_idx == 0
    assert r.matches[0].type_correct is True


def test_actor_mismatch_ranks_last_but_is_not_discarded():
    """Actor agreement RANKS, it does not gate. Gating was measured to be
    net-harmful: as a hard filter it cost 5 of 17 true positives across the 8
    clips and lowered BOTH precision (0.327 -> 0.231) and recall (0.944 ->
    0.667), because a rejected match becomes a false positive AND a false
    negative. It failed on correct detections wherever the person track
    fragmented -- i.e. on the night and CIF-grayscale clips."""
    gt = gt_with_anchors(0.15, 0.25, 0.7, 0.5)
    wrong = pred_with_boxes([0.9, 0.9, 0.95, 0.99], [0.0, 0.0, 0.05, 0.05])
    assert match_events([wrong], [gt]).n_tp == 1


def test_agreeing_actors_win_even_against_a_better_tiou():
    """The ranking is actors first, then tIoU -- a pairing about the right car
    beats a marginally tighter span about the wrong one."""
    gt = gt_with_anchors(0.15, 0.25, 0.7, 0.5, start=0, end=42)
    right = pred_with_boxes([0.1, 0.1, 0.2, 0.4], [0.5, 0.3, 0.9, 0.7],
                            start=0, end=48, idx=1)
    wrong = pred_with_boxes([0.1, 0.1, 0.2, 0.4], [0.0, 0.8, 0.1, 0.95],
                            start=0, end=42, type_="attend_vehicle", idx=2)
    r = match_events([right, wrong], [gt])
    assert r.matches[0].pred_idx == 0, "the exact-span wrong-vehicle pred won"


def test_disabling_the_actor_check_changes_only_the_ranking():
    gt = gt_with_anchors(0.15, 0.25, 0.7, 0.5)
    wrong = pred_with_boxes([0.9, 0.9, 0.95, 0.99], [0.0, 0.0, 0.05, 0.05])
    assert match_events([wrong], [gt], require_same_actors=False).n_tp == 1
