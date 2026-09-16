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


Group = tuple[Sequence[Interaction], Sequence[GTEvent]]


def report(preds: Sequence[Interaction], gts: Sequence[GTEvent],
           tiou: float = TIOU_PRIMARY, confident_only: bool = False) -> dict[str, Any]:
    """Full tiered report for ONE clip."""
    return report_groups([(preds, gts)], tiou, confident_only)


def report_groups(groups: Sequence[Group], tiou: float = TIOU_PRIMARY,
                  confident_only: bool = False) -> dict[str, Any]:
    """Tiered report over one or more clips, **matched within each clip**.

    Pooling predictions and ground truth into one flat matching is wrong, and
    was wrong here for a while: frame spans are clip-local, so a prediction from
    one clip can satisfy an event in another purely because the numbers overlap.
    Demonstrated on the real set -- per-clip true positives summed to 11 while
    the flat pooled matching reported 17, a 55% inflation of every headline
    number.

    So matching happens per group and only the COUNTS are aggregated.

    `confident_only` drops events flagged `ambiguous`. The protocol asks for
    every metric twice -- all events and confident only -- because the gap
    states how much residual error is definitional rather than algorithmic.
    """
    tp = fp = fn = 0
    n_pass_by = n_pass_by_hit = 0
    hit_ids: list[str] = []
    all_positives: list[GTEvent] = []
    n_preds = 0
    confusion = {g: {p: 0 for p in INTERACTION_TYPES} for g in INTERACTION_TYPES}
    per_type_acc = {t: {"tp": 0, "n_gt": 0, "n_pred": 0} for t in TIER2_TYPES}
    starts: list[int] = []
    ends: list[int] = []
    tious: list[float] = []

    for preds, gts in groups:
        if confident_only:
            gts = [g for g in gts if not g.ambiguous]
        result = match_events(preds, gts, tiou_thresh=tiou)
        d = detection_prf(result)
        tp += d.tp
        fp += d.fp
        fn += d.fn
        n_preds += len(preds)
        all_positives.extend(g for g in gts if g.is_positive)

        pb = pass_by_report(gts, result)
        n_pass_by += pb["n_pass_by_labeled"]
        n_pass_by_hit += pb["n_pass_by_falsely_fired"]
        hit_ids.extend(pb["hit_event_ids"])

        for gt_t, row in type_confusion(preds, gts, result).items():
            for pred_t, c in row.items():
                confusion[gt_t][pred_t] += c

        for t, stats in per_type_prf(preds, gts, result, TIER2_TYPES).items():
            per_type_acc[t]["tp"] += stats["tp"]
            per_type_acc[t]["n_gt"] += stats["n_gt"]
            per_type_acc[t]["n_pred"] += stats["n_pred"]

        s_err, e_err = boundary_errors(preds, gts, result)
        starts.extend(s_err)
        ends.extend(e_err)
        tious.extend(m.tiou for m in result.matches)

    tier2 = {t: PRF(tp=a["tp"], fp=a["n_pred"] - a["tp"],
                    fn=a["n_gt"] - a["tp"]).as_dict()
             for t, a in per_type_acc.items()}

    loc: dict[str, Any] = {"n_matched": len(starts)}
    if starts:
        loc.update({
            "median_abs_start_err_frames": statistics.median(abs(v) for v in starts),
            "median_abs_end_err_frames": statistics.median(abs(v) for v in ends),
            "median_signed_start_err_frames": statistics.median(starts),
            "median_signed_end_err_frames": statistics.median(ends),
            "median_tiou": round(statistics.median(tious), 4),
        })

    return {
        "tiou_threshold": tiou,
        "confident_only": confident_only,
        "n_clips": len(groups),
        "n_gt_positives": len(all_positives),
        "n_predictions": n_preds,
        "tier1_detection": PRF(tp=tp, fp=fp, fn=fn).as_dict(),
        "tier1_pass_by": {
            "n_pass_by_labeled": n_pass_by,
            "n_pass_by_falsely_fired": n_pass_by_hit,
            "false_fire_rate": round(_safe_div(n_pass_by_hit, n_pass_by), 4),
            "hit_event_ids": hit_ids,
        },
        "tier2_per_type": tier2,
        "tier3_counts": {t: sum(1 for g in all_positives if g.type == t)
                         for t in TIER3_TYPES},
        "tier3_note": ("n<=2 per class; reported as confusion-matrix entries only, "
                       "never as a rate (problem-definition.md s3)"),
        "type_confusion": confusion,
        "localization": loc,
    }


def full_report(preds: Sequence[Interaction], gts: Sequence[GTEvent]) -> dict[str, Any]:
    """Primary and strict tIoU for ONE clip."""
    return full_report_groups([(preds, gts)])


def full_report_groups(groups: Sequence[Group]) -> dict[str, Any]:
    """Primary and strict tIoU, each over all events and confident events only.

    Takes per-clip groups: matching is clip-scoped, only counts aggregate.
    """
    return {
        "primary": report_groups(groups, TIOU_PRIMARY, confident_only=False),
        "primary_confident_only": report_groups(groups, TIOU_PRIMARY,
                                                confident_only=True),
        "strict": report_groups(groups, TIOU_STRICT, confident_only=False),
        "strict_confident_only": report_groups(groups, TIOU_STRICT,
                                               confident_only=True),
    }
