"""Open-vocabulary door cue attribution.

R4's feature changed from a pixel-change statistic to a text-prompted detection
confidence (experiments/2026-09-16_door-cue-ground-level/findings.md). The
attribution step -- deciding which vehicle a door detection belongs to -- is the
part that can be silently wrong, since a door box sits inside a vehicle box and
their IoU is necessarily tiny.
"""
import numpy as np
import pytest

from pvi.propose.features import Track, door_cue_series

N = 50


def vehicle(box=(100.0, 100.0, 300.0, 200.0), birth=0, death=N - 1, tid=1):
    frames = list(range(birth, death + 1))
    return Track(id=tid, cls=2, frames=frames, boxes=[box] * len(frames))


def test_door_inside_the_vehicle_box_is_attributed_to_it():
    # A door occupying a small part of the vehicle: IoU would be ~0.05.
    dets = {10: [((110.0, 110.0, 150.0, 190.0), 0.42)]}
    s = door_cue_series(dets, vehicle(), N)
    assert s[10] == pytest.approx(0.42)


def test_iou_would_have_missed_that_door():
    """Why attribution uses containment, not IoU."""
    from pvi.geometry import iou
    assert iou((100, 100, 300, 200), (110, 110, 150, 190)) < 0.2


def test_door_on_a_different_vehicle_is_not_attributed():
    dets = {10: [((900.0, 900.0, 950.0, 980.0), 0.5)]}
    assert door_cue_series(dets, vehicle(), N)[10] == 0.0


def test_partially_overlapping_door_below_min_ios_is_rejected():
    # Half in, half out -- ambiguous, so not attributed at the 0.5 floor.
    dets = {10: [((60.0, 110.0, 140.0, 190.0), 0.5)]}
    s = door_cue_series(dets, vehicle(), N, min_ios=0.9)
    assert s[10] == 0.0


def test_strongest_detection_wins_when_several_overlap():
    dets = {10: [((110.0, 110.0, 150.0, 190.0), 0.31),
                 ((160.0, 110.0, 200.0, 190.0), 0.55)]}
    assert door_cue_series(dets, vehicle(), N)[10] == pytest.approx(0.55)


def test_frames_without_a_detection_are_zero_not_nan():
    """The detector ran and found nothing -- evidence of absence. NaN is
    reserved for 'the cue never ran', which R4 skips instead of reading as 0."""
    s = door_cue_series({10: [((110.0, 110.0, 150.0, 190.0), 0.4)]}, vehicle(), N)
    assert s[0] == 0.0 and not np.isnan(s[0])


def test_detection_on_a_frame_the_vehicle_is_absent_is_ignored():
    dets = {5: [((110.0, 110.0, 150.0, 190.0), 0.9)]}
    assert door_cue_series(dets, vehicle(birth=20), N)[5] == 0.0


def test_out_of_range_frames_do_not_raise():
    dets = {N + 10: [((110.0, 110.0, 150.0, 190.0), 0.9)]}
    assert len(door_cue_series(dets, vehicle(), N)) == N


def test_series_length_matches_the_clip():
    assert door_cue_series({}, vehicle(), N).shape == (N,)
