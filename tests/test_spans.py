"""Hysteresis, smoothing and span arithmetic.

The inclusive-span convention is the recurring hazard here: the ground truth's
frame_end is the last frame of contact, so a span [10, 10] is one frame long.
Several of these tests exist only to pin that down.
"""
import math

import pytest

from pvi.propose.spans import (dilate_span, hysteresis_spans, merge_spans,
                               moving_average, sample_frames, span_length_s)

NAN = math.nan


def test_hysteresis_single_crossing():
    # Enters below 0.15, leaves above 0.30.
    series = [0.9, 0.5, 0.10, 0.10, 0.20, 0.25, 0.9]
    # Opens at index 2, held through the 0.20/0.25 band, closes when 0.9 > 0.30.
    assert hysteresis_spans(series, enter=0.15, leave=0.30) == [(2, 5)]


def test_hysteresis_suppresses_fragmentation_in_the_band():
    """A person hovering between the thresholds must yield one span, not many.
    This is the entire reason hysteresis is in the design."""
    series = [0.9, 0.1, 0.2, 0.1, 0.25, 0.1, 0.2, 0.9]
    assert hysteresis_spans(series, enter=0.15, leave=0.30) == [(1, 6)]


def test_single_threshold_would_fragment_the_same_series():
    """Contrast case: with enter == leave the trigger degenerates, which is why
    config validation forbids tau_far <= tau_near."""
    series = [0.9, 0.1, 0.2, 0.1, 0.25, 0.1, 0.2, 0.9]
    spans = hysteresis_spans(series, enter=0.15, leave=0.1500001)
    assert len(spans) > 1


def test_hysteresis_rejects_degenerate_thresholds():
    with pytest.raises(ValueError, match="must exceed"):
        hysteresis_spans([0.1, 0.2], enter=0.3, leave=0.3)


def test_hysteresis_span_open_at_end_of_series_is_closed():
    series = [0.9, 0.1, 0.1]
    assert hysteresis_spans(series, enter=0.15, leave=0.30) == [(1, 2)]


def test_hysteresis_nan_does_not_open_a_span():
    assert hysteresis_spans([NAN, NAN], enter=0.15, leave=0.30) == []


def test_hysteresis_nan_holds_an_open_span_across_occlusion():
    """Occlusion must not split one true interaction into two spans."""
    series = [0.9, 0.1, NAN, NAN, 0.1, 0.9]
    assert hysteresis_spans(series, enter=0.15, leave=0.30) == [(1, 4)]


def test_hysteresis_trailing_nan_closes_at_the_last_measurement():
    """A person who enters a vehicle stops being tracked, so the series ends in
    NaN. Closing at the end of the series instead would stretch the event to the
    end of the clip -- wrecking its boundary error and hiding the track death
    that identifies it as an entry in the first place."""
    series = [0.9, 0.1, 0.1, NAN, NAN, NAN]
    assert hysteresis_spans(series, enter=0.15, leave=0.30) == [(1, 2)]


def test_hysteresis_distinguishes_interior_gap_from_trailing_gap():
    interior = [0.9, 0.1, NAN, 0.1, 0.9]
    trailing = [0.9, 0.1, NAN, 0.1, NAN]
    assert hysteresis_spans(interior, enter=0.15, leave=0.30) == [(1, 3)]
    assert hysteresis_spans(trailing, enter=0.15, leave=0.30) == [(1, 3)]


def test_moving_average_preserves_length():
    assert len(moving_average([1.0] * 7, 3)) == 7


def test_moving_average_window_one_is_identity():
    x = [1.0, 5.0, 2.0]
    assert moving_average(x, 1) == x


def test_moving_average_clamps_edges_rather_than_zero_padding():
    """Zero-padding a d_norm series would fabricate contact at the clip
    boundary; the first sample must stay near its neighbours."""
    x = [10.0, 10.0, 10.0, 10.0, 10.0]
    assert moving_average(x, 3) == pytest.approx([10.0] * 5)


def test_moving_average_ignores_nan_but_keeps_all_nan_windows_nan():
    out = moving_average([1.0, NAN, 3.0], 3)
    assert out[1] == pytest.approx(2.0)
    assert all(math.isnan(v) for v in moving_average([NAN, NAN], 3))


def test_span_length_is_inclusive():
    # One frame at 6 fps is 1/6 s, not 0.
    assert span_length_s((10, 10), 6.0) == pytest.approx(1 / 6)
    assert span_length_s((0, 5), 6.0) == pytest.approx(1.0)


def test_shortest_gt_positive_is_five_frames_at_imgr_fps():
    """design-plan s4's binding constraint, pinned as a test: the 0.75 s shortest
    positive is 5 frames at iMGR's 6.667 fps."""
    assert span_length_s((0, 4), 6.667) == pytest.approx(0.75, abs=0.01)


def test_merge_joins_overlapping_and_abutting():
    assert merge_spans([(0, 5), (3, 8)]) == [(0, 8)]
    assert merge_spans([(0, 5), (6, 8)]) == [(0, 8)]


def test_merge_keeps_separated_spans_apart():
    assert merge_spans([(0, 5), (8, 10)]) == [(0, 5), (8, 10)]


def test_merge_honours_gap_tolerance():
    assert merge_spans([(0, 5), (8, 10)], gap_tolerance=2) == [(0, 10)]


def test_merge_is_order_independent():
    assert merge_spans([(8, 10), (0, 5), (3, 6)]) == merge_spans([(0, 5), (3, 6), (8, 10)])


def test_merge_absorbs_nested_span():
    assert merge_spans([(0, 20), (5, 8)]) == [(0, 20)]


def test_dilate_span_clips_at_clip_boundaries():
    assert dilate_span((2, 8), 5, n_frames=10) == (0, 9)


def test_sample_frames_includes_both_endpoints():
    out = sample_frames((0, 100), 5)
    assert out[0] == 0 and out[-1] == 100 and len(out) == 5


def test_sample_frames_short_span_returns_every_frame_without_duplicates():
    assert sample_frames((10, 12), 12) == [10, 11, 12]


def test_sample_frames_single():
    assert sample_frames((10, 20), 1) == [15]
