"""Span extraction from feature time series: hysteresis, smoothing, merging.

These are the deterministic numerical primitives the four proposal rules are
built from. They live in their own module because they are the part of the
pipeline where a silent off-by-one is both easy to write and invisible
downstream -- a span one frame short still looks like a plausible detection.

All spans are INCLUSIVE [start, end] frame indices, matching the ground-truth
convention where frame_end is the last frame of contact.
"""
from __future__ import annotations

import math
from typing import Sequence

Span = tuple[int, int]


def moving_average(x: Sequence[float], window: int) -> list[float]:
    """Centred moving average with edge clamping, NaN-aware.

    Edge clamping (rather than shrinking the window or zero-padding) keeps the
    output the same length as the input and avoids pulling the first and last
    samples toward zero -- which for a d_norm series would fabricate contact at
    the clip boundary.

    NaN inputs are ignored within a window; a window containing only NaN yields
    NaN. This matters because `door_delta` is NaN by design wherever the vehicle
    or camera is moving, and smoothing must not turn that into a number.
    """
    n = len(x)
    if n == 0:
        return []
    if window <= 1:
        return list(x)
    half = window // 2
    out: list[float] = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vals = [v for v in x[lo:hi] if not math.isnan(v)]
        out.append(sum(vals) / len(vals) if vals else math.nan)
    return out


def hysteresis_spans(series: Sequence[float], enter: float, leave: float) -> list[Span]:
    """Spans where a series is 'near', with Schmitt-trigger hysteresis.

    A span opens when the value drops BELOW `enter` and stays open until the
    value rises ABOVE `leave`, where `leave > enter`. Between the two the state
    is held. Without this, a person hovering at the threshold produces a burst
    of one-frame fragments instead of a single span, and every downstream rule
    sees the wrong thing.

    NaN is treated as 'no measurement': it never opens a span, and an interior
    NaN run does not close an open one (the state is held across the gap).
    Occlusion would otherwise split one true interaction into two.

    A span still open when the measurements run out closes at the LAST
    non-NaN sample, not at the end of the series. The distinction matters: a
    person who enters a vehicle stops being tracked, so their d_norm series
    ends in NaN, and closing at the end of the series would stretch the event
    to the end of the clip and destroy both its boundary error and the
    track-death test that identifies it as an entry.

    Raises ValueError if leave <= enter, since the trigger degenerates.
    """
    if leave <= enter:
        raise ValueError(f"leave ({leave}) must exceed enter ({enter}); "
                         "equal or inverted thresholds degenerate the hysteresis")
    spans: list[Span] = []
    start: int | None = None
    last_valid = -1
    for i, v in enumerate(series):
        if math.isnan(v):
            continue
        last_valid = i
        if start is None:
            if v < enter:
                start = i
        elif v > leave:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, last_valid))
    return spans


def span_length_s(span: Span, fps: float) -> float:
    """Inclusive span duration in seconds.

    Inclusive, so a single-frame span is one frame long (1/fps), not zero. The
    ground truth's shortest positive is 0.75 s and thresholds are compared
    against this, so the off-by-one is not cosmetic.
    """
    return (span[1] - span[0] + 1) / fps


def merge_spans(spans: Sequence[Span], gap_tolerance: int = 0) -> list[Span]:
    """Merge overlapping or near-adjacent spans.

    `gap_tolerance` is in frames and is the caller's responsibility to derive
    from seconds. Spans separated by <= gap_tolerance frames are joined, so
    abutting spans (end, end+1) merge at the default of 0.
    """
    if not spans:
        return []
    ordered = sorted(spans)
    out = [ordered[0]]
    for s, e in ordered[1:]:
        ps, pe = out[-1]
        if s <= pe + 1 + gap_tolerance:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def dilate_span(span: Span, pad_frames: int, n_frames: int) -> Span:
    """Pad a span on both sides, clipped to [0, n_frames - 1]."""
    return (max(0, span[0] - pad_frames), min(n_frames - 1, span[1] + pad_frames))


def sample_frames(span: Span, n: int) -> list[int]:
    """`n` frame indices spread uniformly over an inclusive span.

    Endpoints are always included when n >= 2 -- the first and last frames of a
    candidate are the most informative ones for enter/exit, since that is where
    the person appears or vanishes. Returns fewer than `n` indices only when the
    span is shorter than `n` frames, and never returns duplicates.
    """
    lo, hi = span
    length = hi - lo + 1
    if n <= 0 or length <= 0:
        return []
    if n == 1:
        return [lo + length // 2]
    if length <= n:
        return list(range(lo, hi + 1))
    step = (length - 1) / (n - 1)
    return sorted({lo + int(round(i * step)) for i in range(n)})
