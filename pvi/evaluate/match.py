"""Temporal matching of predicted interactions to ground-truth events.

Matching is greedy by descending temporal IoU with one-to-one assignment. The
alternative -- optimal (Hungarian) assignment -- is defensible but overkill at
this scale, and greedy has the property that a reviewer can reproduce the
pairing by hand from the tIoU table, which matters more here than the last
fraction of a percent.

Primary threshold is tIoU >= 0.3, deliberately loose because the GT boundaries
are themselves fuzzy (+-2 frames by protocol). tIoU >= 0.5 is reported as a
stricter secondary number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..schema import GTEvent, Interaction

TIOU_PRIMARY = 0.3
TIOU_STRICT = 0.5


def temporal_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """IoU of two INCLUSIVE frame spans.

    Inclusive because the ground truth's frame_end is the last frame of contact,
    not one past it. A span [10, 10] is therefore one frame long, not zero --
    getting this wrong makes every single-frame event score 0.0 against itself.
    """
    a_len = a_end - a_start + 1
    b_len = b_end - b_start + 1
    if a_len <= 0 or b_len <= 0:
        return 0.0
    inter = min(a_end, b_end) - max(a_start, b_start) + 1
    if inter <= 0:
        return 0.0
    union = a_len + b_len - inter
    return inter / union


@dataclass(frozen=True)
class Match:
    pred_idx: int
    gt_idx: int
    tiou: float
    type_correct: bool


@dataclass(frozen=True)
class MatchResult:
    matches: tuple[Match, ...]
    unmatched_pred: tuple[int, ...]   # false positives
    unmatched_gt: tuple[int, ...]     # false negatives (positives only)
    # Predictions that landed on a labeled pass_by. These are false positives,
    # but diagnostically the interesting kind: the set was built to probe them.
    pass_by_hits: tuple[Match, ...]

    @property
    def n_tp(self) -> int:
        return len(self.matches)

    @property
    def n_fp(self) -> int:
        return len(self.unmatched_pred)

    @property
    def n_fn(self) -> int:
        return len(self.unmatched_gt)


# How far outside a predicted box a GT anchor may fall and still be taken to
# refer to the same actor, as a fraction of the box's own size. Generous,
# because the anchors are spatial hints placed by hand, sometimes on a frame a
# few frames outside the event span (see the labeling-protocol audit log).
ANCHOR_TOL = 0.5


def anchors_agree(pred: Interaction, gt: GTEvent, tol: float = ANCHOR_TOL) -> bool:
    """Whether a prediction refers to the same person and vehicle as a GT event.

    `problem-definition.md` §3 requires a match to share the person-vehicle
    pair, and the GT's normalized anchors exist for exactly this. Without the
    check, matching is purely temporal: on `NmlzoaDcOuI_6` the greedy matcher
    picked a prediction about a *different vehicle* over the correct one,
    because its span happened to overlap 3% better.

    Degrades to True when either side lacks the information -- an absent anchor
    or a prediction carrying no box means "cannot tell", and refusing every such
    match would silently convert a missing field into a false negative.
    """
    for kind, anchor in (("person", gt.person_anchor), ("vehicle", gt.vehicle_anchor)):
        if not anchor:
            continue
        box = pred.evidence.get(f"{kind}_box_norm")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        dw, dh = (x2 - x1) * tol, (y2 - y1) * tol
        if not (x1 - dw <= anchor["x"] <= x2 + dw
                and y1 - dh <= anchor["y"] <= y2 + dh):
            return False
    return True


def match_events(preds: Sequence[Interaction], gts: Sequence[GTEvent],
                 tiou_thresh: float = TIOU_PRIMARY,
                 require_same_type: bool = False,
                 require_same_actors: bool = True) -> MatchResult:
    """Greedy one-to-one matching of predictions to GT events.

    `require_same_type=False` by default: detection and classification are
    scored separately, so a span found with the wrong type counts as a detection
    hit and shows up in the type confusion matrix. Conflating them would hide
    whether a miss was a localization failure or a naming one.

    `require_same_actors=True` by default: a prediction must also refer to the
    same person and vehicle, tested against the GT anchors. Temporal overlap
    alone is not a match in the multi-actor clips.

    Positives and `pass_by` negatives are matched in separate passes, positives
    first, so a prediction that overlaps both is credited to the positive.
    """
    pos_idx = [i for i, g in enumerate(gts) if g.is_positive]
    neg_idx = [i for i, g in enumerate(gts) if not g.is_positive]

    taken_pred: set[int] = set()
    matches = _greedy(preds, gts, pos_idx, taken_pred, tiou_thresh,
                      require_same_type, require_same_actors)
    pass_by = _greedy(preds, gts, neg_idx, taken_pred, tiou_thresh, False,
                      require_same_actors)

    matched_gt = {m.gt_idx for m in matches}
    return MatchResult(
        matches=tuple(matches),
        unmatched_pred=tuple(i for i in range(len(preds)) if i not in taken_pred),
        unmatched_gt=tuple(i for i in pos_idx if i not in matched_gt),
        pass_by_hits=tuple(pass_by),
    )


def _greedy(preds: Sequence[Interaction], gts: Sequence[GTEvent],
            gt_pool: list[int], taken_pred: set[int],
            tiou_thresh: float, require_same_type: bool,
            require_same_actors: bool = True) -> list[Match]:
    cands: list[tuple[float, int, int]] = []
    for pi, p in enumerate(preds):
        if pi in taken_pred:
            continue
        for gi in gt_pool:
            g = gts[gi]
            if require_same_type and p.type != g.type:
                continue
            if require_same_actors and not anchors_agree(p, g):
                continue
            t = temporal_iou(p.frame_start, p.frame_end, g.frame_start, g.frame_end)
            if t >= tiou_thresh:
                cands.append((t, pi, gi))

    # Sort by descending tIoU; ties broken by index so the result is
    # deterministic regardless of input ordering.
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))

    used_gt: set[int] = set()
    out: list[Match] = []
    for t, pi, gi in cands:
        if pi in taken_pred or gi in used_gt:
            continue
        taken_pred.add(pi)
        used_gt.add(gi)
        out.append(Match(pred_idx=pi, gt_idx=gi, tiou=t,
                         type_correct=preds[pi].type == gts[gi].type))
    return out


def boundary_errors(preds: Sequence[Interaction], gts: Sequence[GTEvent],
                    result: MatchResult) -> tuple[list[int], list[int]]:
    """Signed start/end frame errors over matched pairs (pred - gt).

    Signed, not absolute, so a systematic bias (the system consistently starting
    late) is distinguishable from symmetric jitter. The metric layer reports the
    median of the absolute values, but the sign is what diagnoses a convention
    mismatch against the contact-based GT boundary.
    """
    starts, ends = [], []
    for m in result.matches:
        starts.append(preds[m.pred_idx].frame_start - gts[m.gt_idx].frame_start)
        ends.append(preds[m.pred_idx].frame_end - gts[m.gt_idx].frame_end)
    return starts, ends
