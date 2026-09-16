"""Tiered metric reporting.

The tiers are not a stylistic choice -- they are forced by the realized ground
truth counts (problem-definition.md s3). With n=1 for `load_unload`, a per-type
recall is 0% or 100% and carries no information. Quoting one would be the single
easiest way to make this evaluation look more rigorous than it is, so the tier-3
path here returns a confusion matrix and refuses to compute a rate.

Every number is emitted with its denominator attached, because a single event is
5.6% of aggregate recall over 18 positives and a bare percentage hides that.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

from ..schema import (GTEvent, Interaction, INTERACTION_TYPES, TIER2_TYPES,
                      TIER3_TYPES)
from .match import MatchResult, TIOU_PRIMARY, TIOU_STRICT, boundary_errors, match_events


@dataclass(frozen=True)
class Counted:
    """A rate that always carries its denominator."""
    value: float
    n: int

    def __str__(self) -> str:
        return f"{self.value:.3f} (n={self.n})"


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


@dataclass
class PRF:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return _safe_div(2 * p * r, p + r)

    def as_dict(self) -> dict[str, Any]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
                "n_gt": self.tp + self.fn, "n_pred": self.tp + self.fp}


def detection_prf(result: MatchResult) -> PRF:
    """Aggregate detection PRF.

    Predictions landing on a labeled `pass_by` count as false positives here --
    they are wrong. They are additionally broken out in `pass_by_report`, which
    is the diagnostic the 7 labeled near-misses exist to support.
    """
    return PRF(tp=result.n_tp, fp=result.n_fp + len(result.pass_by_hits),
               fn=result.n_fn)


def pass_by_report(gts: Sequence[GTEvent], result: MatchResult) -> dict[str, Any]:
    """Of N labeled near-misses, how many did the system falsely fire on.

    This is the number the dataset was constructed to probe, and it converts a
    bare false-positive count into something diagnostic.
    """
    n_pass_by = sum(1 for g in gts if not g.is_positive)
    hit = len(result.pass_by_hits)
    return {
        "n_pass_by_labeled": n_pass_by,
        "n_pass_by_falsely_fired": hit,
        "false_fire_rate": round(_safe_div(hit, n_pass_by), 4),
        "hit_event_ids": [gts[m.gt_idx].event_id for m in result.pass_by_hits],
    }


def type_confusion(preds: Sequence[Interaction], gts: Sequence[GTEvent],
                   result: MatchResult) -> dict[str, dict[str, int]]:
    """Confusion matrix over matched pairs: gt_type -> pred_type -> count."""
    mat: dict[str, dict[str, int]] = {
        g: {p: 0 for p in INTERACTION_TYPES} for g in INTERACTION_TYPES
    }
    for m in result.matches:
        gt_t, pred_t = gts[m.gt_idx].type, preds[m.pred_idx].type
        if gt_t in mat and pred_t in mat[gt_t]:
            mat[gt_t][pred_t] += 1
    return mat


def per_type_prf(preds: Sequence[Interaction], gts: Sequence[GTEvent],
                 result: MatchResult, types: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Per-type PRF, requiring the type to be correct for a TP.

    Only called for tier-2 types. Calling it for a tier-3 type is prevented at
    the report level, not here, so the function stays reusable if the GT ever
    grows.
    """
    out: dict[str, dict[str, Any]] = {}
    for t in types:
        tp = sum(1 for m in result.matches
                 if gts[m.gt_idx].type == t and preds[m.pred_idx].type == t)
        n_gt = sum(1 for g in gts if g.type == t)
        n_pred = sum(1 for p in preds if p.type == t)
        out[t] = PRF(tp=tp, fp=n_pred - tp, fn=n_gt - tp).as_dict()
    return out


def localization(preds: Sequence[Interaction], gts: Sequence[GTEvent],
                 result: MatchResult) -> dict[str, Any]:
    starts, ends = boundary_errors(preds, gts, result)
    if not starts:
        return {"n_matched": 0}
    return {
        "n_matched": len(starts),
        "median_abs_start_err_frames": statistics.median(abs(v) for v in starts),
        "median_abs_end_err_frames": statistics.median(abs(v) for v in ends),
        # Signed medians diagnose a convention mismatch against the contact-based
        # GT boundary, which absolute values would hide.
        "median_signed_start_err_frames": statistics.median(starts),
        "median_signed_end_err_frames": statistics.median(ends),
    }


def report(preds: Sequence[Interaction], gts: Sequence[GTEvent],
           tiou: float = TIOU_PRIMARY, confident_only: bool = False) -> dict[str, Any]:
    """Full tiered report for one clip or for the pooled set.

    `confident_only` drops events flagged `ambiguous` in the GT. The protocol
    asks for every metric twice -- over all events and over confident ones --
    because the gap between them states how much residual error is definitional
    rather than algorithmic.
    """
    if confident_only:
        gts = [g for g in gts if not g.ambiguous]
    result = match_events(preds, gts, tiou_thresh=tiou)
    positives = [g for g in gts if g.is_positive]

    tier3_counts = {t: sum(1 for g in positives if g.type == t) for t in TIER3_TYPES}

    return {
        "tiou_threshold": tiou,
        "confident_only": confident_only,
        "n_gt_positives": len(positives),
        "n_predictions": len(preds),
        # Tier 1
        "tier1_detection": detection_prf(result).as_dict(),
        "tier1_pass_by": pass_by_report(gts, result),
        # Tier 2 -- rates are defensible at n=6 and n=7, with counts inline.
        "tier2_per_type": per_type_prf(preds, gts, result, TIER2_TYPES),
        # Tier 3 -- counts and confusion only. No rate is computed for these,
        # by design: n = 2, 2, 1.
        "tier3_counts": tier3_counts,
        "tier3_note": ("n<=2 per class; reported as confusion-matrix entries only, "
                       "never as a rate (problem-definition.md s3)"),
        "type_confusion": type_confusion(preds, gts, result),
        "localization": localization(preds, gts, result),
    }


def full_report(preds: Sequence[Interaction], gts: Sequence[GTEvent]) -> dict[str, Any]:
    """Primary and strict tIoU, each over all events and confident events only."""
    return {
        "primary": report(preds, gts, TIOU_PRIMARY, confident_only=False),
        "primary_confident_only": report(preds, gts, TIOU_PRIMARY, confident_only=True),
        "strict": report(preds, gts, TIOU_STRICT, confident_only=False),
        "strict_confident_only": report(preds, gts, TIOU_STRICT, confident_only=True),
    }
