"""Geometry tests. These target the normalization invariant from design-plan s4:
a silent error here is not a crash, it is a threshold that means different things
on different clips.
"""
import math

import pytest

from pvi.geometry import (d_norm, diag, dilate, gap_px, iou, speed_bl_s,
                          union_box, width, height)


def test_gap_is_zero_when_boxes_overlap():
    assert gap_px((0, 0, 10, 10), (5, 5, 15, 15)) == 0.0


def test_gap_is_zero_when_boxes_touch():
    assert gap_px((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_gap_single_axis():
    # Separated in x only, aligned in y: the gap is the x distance.
    assert gap_px((0, 0, 10, 10), (14, 0, 20, 10)) == pytest.approx(4.0)


def test_gap_is_hypotenuse_not_sum_on_diagonal_separation():
    # 3 in x, 4 in y -> 5, not 7 and not 4. This is the case a naive
    # implementation gets wrong.
    assert gap_px((0, 0, 10, 10), (13, 14, 20, 20)) == pytest.approx(5.0)


def test_d_norm_is_scale_invariant():
    """The whole point of normalizing: the same scene at 4x resolution must
    produce the same d_norm. If this fails, no threshold transfers between the
    352x288 and 3840x2160 clips."""
    person, vehicle = (100, 0, 120, 50), (0, 0, 60, 80)
    big_p = tuple(c * 4 for c in person)
    big_v = tuple(c * 4 for c in vehicle)
    assert d_norm(person, vehicle) == pytest.approx(d_norm(big_p, big_v))


def test_d_norm_normalizes_by_vehicle_diagonal():
    # Vehicle 30x40 -> diagonal 50. Person 10 px to its right.
    assert d_norm((40, 0, 50, 20), (0, 0, 30, 40)) == pytest.approx(10.0 / 50.0)


def test_d_norm_zero_on_contact():
    assert d_norm((10, 10, 20, 20), (0, 0, 30, 30)) == 0.0


def test_d_norm_degenerate_vehicle_is_inf_not_crash():
    """A detector can emit a zero-area box; the caller should read 'no usable
    measurement', not catch a ZeroDivisionError."""
    assert math.isinf(d_norm((10, 10, 20, 20), (5, 5, 5, 5)))


def test_iou_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # 10x10 and 10x10 sharing a 5x10 strip: inter 50, union 150.
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_speed_is_body_length_normalized():
    """A 10 px aerial person and a 400 px CCTV person moving one body length per
    second must report the same speed."""
    small = speed_bl_s((0, 0, 4, 10), (10, 0, 14, 10), 1.0)
    big = speed_bl_s((0, 0, 160, 400), (400, 0, 560, 400), 1.0)
    assert small == pytest.approx(big)
    assert small == pytest.approx(1.0)


def test_speed_zero_dt_does_not_divide_by_zero():
    assert speed_bl_s((0, 0, 10, 20), (5, 0, 15, 20), 0.0) == 0.0


def test_union_box():
    assert union_box((0, 5, 10, 15), (3, 0, 20, 8)) == (0, 0, 20, 15)


def test_dilate_is_fractional_so_it_scales():
    small = dilate((0, 0, 10, 10), 0.5)
    assert width(small) == pytest.approx(20.0)
    big = dilate((0, 0, 100, 100), 0.5)
    assert width(big) == pytest.approx(200.0)


def test_dilate_clips_to_bounds():
    out = dilate((0, 0, 10, 10), 1.0, bounds=(15, 15))
    assert out == (0.0, 0.0, 15.0, 15.0)
