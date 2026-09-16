"""Rule-only judge -- the ablation control arm.

This is NOT a stub. It is the baseline the VLM arm is measured against, and it
is deliberately crude: it accepts every candidate the proposer emitted and types
it from whichever rule fired. That makes it exactly "the geometric pipeline with
no semantic layer", which is the comparison the design plan asks for.

Its expected failure is precision. The proposer is tuned for recall and this
judge rejects nothing, so every labeled `pass_by` that produced a candidate
becomes a false positive. That number *is* the result: it quantifies how much
the semantic layer has to buy back.
"""
from __future__ import annotations

from ..propose.features import PairFeatures
from ..propose.rules import Candidate, RULE_TYPE, primary_rule
from ..schema import ClipMeta
from .base import Judge, Verdict

VEHICLE_NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class GeometricJudge(Judge):
    """Accepts everything; types from the triggering rule."""

    def __init__(self, features: dict[tuple[int, int], PairFeatures] | None = None):
        # Optional, only to enrich the descriptions. The judge works without it.
        self.features = features or {}

    @property
    def name(self) -> str:
        return "geometric"

    def judge(self, cand: Candidate, meta: ClipMeta) -> Verdict | None:
        rule = primary_rule(cand)
        vname = VEHICLE_NAMES.get(cand.vehicle_cls, "vehicle")
        pf = self.features.get((cand.person_id, cand.vehicle_id))
        state = "parked" if (pf.vehicle_is_static if pf else
                             cand.evidence.get("vehicle_is_static", True)) else "moving"

        return Verdict(
            is_interaction=True,
            type=RULE_TYPE[rule],
            # No appearance model here by design -- the track id is the only
            # identity this arm has, and claiming clothing colours it never
            # computed would misrepresent the ablation.
            person_desc=f"person track {cand.person_id}",
            vehicle_desc=f"{vname} track {cand.vehicle_id}, {state}",
            note=f"accepted by {rule} (geometric ablation arm; no semantic check)",
            # Fixed, not a real confidence: this arm has no calibrated score, and
            # emitting a varying number would invite reading it as one.
            confidence=0.5,
        )
