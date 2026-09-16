"""Candidate proposal: four recall-oriented rules over pair feature series.

This stage is tuned for RECALL. It is allowed to be wrong often; the VLM judge
is what buys precision back. A candidate the proposer never emits, though, is
unrecoverable -- so when in doubt, propose.

The rules (design-plan s5):

  R1 dwell        contiguous `near` for >= min_dwell_s        -> attend, load_unload
  R2 death-near   person track ENDS while near                -> enter_vehicle
  R3 birth-near   person track STARTS while near              -> exit_vehicle
  R4 door-change  door_delta spike while any person is near   -> open_close_door

R2 and R3 are what structurally separate enter/exit from pass-by: a person who
walks past is born far from the vehicle and dies far from it, whereas someone
who gets in stops being visible *at* the vehicle. That asymmetry is geometric,
not semantic, which is why it belongs here rather than in the judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import math
import numpy as np

from ..schema import ClipMeta
from .features import PairFeatures, Track
from .spans import Span, hysteresis_spans, merge_spans, span_length_s

R1_DWELL = "R1_dwell"
R2_DEATH_NEAR = "R2_death_near"
R3_BIRTH_NEAR = "R3_birth_near"
R4_DOOR_CHANGE = "R4_door_change"

# Rule -> the type the geometric (ablation) judge assigns. The VLM judge is free
# to disagree; this mapping is the control arm's entire classification logic.
RULE_TYPE = {
    R1_DWELL: "attend_vehicle",
    R2_DEATH_NEAR: "enter_vehicle",
    R3_BIRTH_NEAR: "exit_vehicle",
    R4_DOOR_CHANGE: "open_close_door",
}

# Tolerance, in seconds, for "the track's birth/death coincides with this
# near-span". Not a clip-edge guard -- see below.
BIRTH_DEATH_TOL_S = 0.35

# There is deliberately NO clip-edge suppression, and that is a reversal.
#
# An earlier version refused to let R2/R3 fire when the track was born at the
# clip's first frame or died at its last, reasoning that "a person visible in
# frame 0 did not appear; the clip did". Measured against the frozen ground
# truth, that guard suppressed **3 of the 7 `exit_vehicle` positives** -- every
# one of them starts at frame 0 -- plus 1 of 6 `enter_vehicle`. These clips are
# short segments cut around their events, so events abut the boundaries by
# construction rather than by accident.
#
# The guard also removed the wrong cases. R3 only fires when the track is born
# *while already in contact with the vehicle*, and someone already touching the
# car in frame 0 is exactly what having-just-exited looks like. The guard was
# deleting the signal, not the noise.
#
# The residual risk it addressed -- a pedestrian who merely happens to stand
# beside a car at frame 0 and then walks away -- is a semantic judgement, which
# is the judge's job, not the proposer's. The proposer is recall-first by
# design. `at_clip_edge` is recorded in the evidence so the distinction stays
# visible to the judge, to the metrics, and to a reviewer.


@dataclass
class Candidate:
    person_id: int
    vehicle_id: int
    vehicle_cls: int
    frame_start: int
    frame_end: int
    rules: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> Span:
        return (self.frame_start, self.frame_end)

    def duration_s(self, fps: float) -> float:
        return span_length_s(self.span, fps)


def near_spans(pf: PairFeatures, tau_near: float, tau_far: float) -> list[Span]:
    return hysteresis_spans(list(pf.d_norm), enter=tau_near, leave=tau_far)


def propose_pair(pf: PairFeatures, person: Track, meta: ClipMeta,
                 tau_near: float, tau_far: float, min_dwell_s: float,
                 door_conf_thresh: float) -> list[Candidate]:
    """Apply all four rules to one pair and merge the spans they produce."""
    spans = near_spans(pf, tau_near, tau_far)
    if not spans:
        return []

    tol = max(1, meta.to_frames(BIRTH_DEATH_TOL_S))
    fired: dict[Span, list[str]] = {}

    for sp in spans:
        hits: list[str] = []

        if span_length_s(sp, meta.fps) >= min_dwell_s:
            hits.append(R1_DWELL)

        # R2/R3 ask whether the track's own birth/death coincides with this
        # near-span. No clip-edge suppression -- see BIRTH_DEATH_TOL_S above.
        if abs(person.death_frame - sp[1]) <= tol:
            hits.append(R2_DEATH_NEAR)
        if abs(person.birth_frame - sp[0]) <= tol:
            hits.append(R3_BIRTH_NEAR)

        # R4's feature is now an open-vocabulary "open car door" confidence,
        # not a pixel-change statistic (see the door-cue findings). NaN still
        # means "not measured" -- the cue did not run -- and is skipped rather
        # than read as zero.
        dd = pf.door_delta[sp[0]:sp[1] + 1]
        if dd.size and not np.all(np.isnan(dd)):
            if float(np.nanmax(dd)) >= door_conf_thresh:
                hits.append(R4_DOOR_CHANGE)

        if hits:
            fired[sp] = hits

    if not fired:
        return []

    merged = merge_spans(list(fired))
    out: list[Candidate] = []
    for m in merged:
        rules: list[str] = []
        for sp, hits in fired.items():
            if sp[0] >= m[0] and sp[1] <= m[1]:
                rules.extend(h for h in hits if h not in rules)
        seg = pf.d_norm[m[0]:m[1] + 1]
        valid = seg[~np.isnan(seg)]
        out.append(Candidate(
            person_id=pf.person_id, vehicle_id=pf.vehicle_id,
            vehicle_cls=pf.vehicle_cls,
            frame_start=m[0], frame_end=m[1],
            rules=sorted(rules),
            evidence={
                "min_d_norm": round(float(valid.min()), 4) if valid.size else None,
                "dwell_s": round(span_length_s(m, meta.fps), 3),
                "vehicle_is_static": pf.vehicle_is_static,
                "person_birth": person.birth_frame,
                "person_death": person.death_frame,
                # Kept visible because R2/R3 no longer suppress these: a track
                # born at frame 0 may be a real exit or may be the clip simply
                # starting, and that call belongs to the judge.
                "birth_at_clip_start": person.birth_frame <= tol,
                "death_at_clip_end": person.death_frame >= meta.n_frames - 1 - tol,
                "max_iou": round(float(np.nanmax(pf.iou[m[0]:m[1] + 1])), 4)
                if not np.all(np.isnan(pf.iou[m[0]:m[1] + 1])) else None,
            },
        ))
    return out


def propose(pairs: Sequence[tuple[PairFeatures, Track]], meta: ClipMeta,
            tau_near: float, tau_far: float, min_dwell_s: float,
            door_conf_thresh: float) -> list[Candidate]:
    out: list[Candidate] = []
    for pf, person in pairs:
        out.extend(propose_pair(pf, person, meta, tau_near, tau_far,
                                min_dwell_s, door_conf_thresh))
    # Deterministic ordering: the pipeline's output order must not depend on
    # dict iteration or track-creation order.
    out.sort(key=lambda c: (c.frame_start, c.frame_end, c.person_id, c.vehicle_id))
    return out


def primary_rule(cand: Candidate) -> str:
    """The rule that decides the type when several fired.

    Priority is enter/exit first, then the door cue, then dwell. Rationale: R2
    and R3 encode a structural observation (the track began or ended at the
    vehicle) that R1 cannot -- R1 fires on essentially every true interaction as
    well, so letting it win would collapse every type to `attend_vehicle`.
    """
    for r in (R2_DEATH_NEAR, R3_BIRTH_NEAR, R4_DOOR_CHANGE, R1_DWELL):
        if r in cand.rules:
            return r
    return R1_DWELL
