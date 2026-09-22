"""Slot-level accuracy of the person and vehicle descriptions.

The task requires a description of the person(s) and of the vehicle for every
interaction; detection metrics say nothing about whether those are right. This
scores them against the ground truth's `person_desc` / `vehicle_desc`.

Which pairs are scored: true positives under the PRIMARY matching (tIoU >= 0.3,
within-clip), and of those only the ones whose actors agree with the GT anchors
(match.anchors_agree). A temporal match about a different person would grade a
description of the wrong individual, so it is counted as `actor_mismatch` and
excluded rather than scored as wrong.

The GT strings are free text in a fixed labeling order, "<age> <sex>, <upper>,
<lower>[, <carried>...]" and "<colour>, <body>, <state>". They are parsed here,
at evaluation time, rather than rewritten: the frozen file is not touched, and
the parse is pinned by a test over every GT string.

Scoring assumptions -- these ARE the metric, so they are stated, not buried:

- **Hue vs lightness.** A colour is scored twice. `hue` asks whether the named
  colour is right; it is only scored when the GT names one (the labeler wrote
  plain "dark"/"light" where only luminance was visible). `lightness` asks
  whether dark/light is right, and is scored for every colour.
- **grey == silver** for hue. On a car they are the same paint seen differently.
- **Lightness classes** below. grey, red and orange are compatible with both,
  because at night or in compressed video they land on either side.
- **`unknown` is an abstention**: counted, excluded from accuracy, and reported
  as lost coverage. A model that abstains on grayscale footage should not score
  as wrong, and one that abstains on everything should not score as right.
- **Vehicle state**: parked and stopped merge into `static`; the GT's
  "driving" / "starting to drive" are `moving`.
- **Carrying** is yes/no on both sides (the model is not asked to name the
  object; see judge/describe.py). Scored only when the GT description is in full
  form (it names clothing), since a terse GT like "young boy" says nothing
  about what was carried.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..judge.describe import UNKNOWN
from ..schema import GTEvent, Interaction
from .match import TIOU_PRIMARY, anchors_agree, match_events

DARK = frozenset({"black", "dark", "brown", "blue", "green", "purple", "navy"})
LIGHT = frozenset({"white", "light", "beige", "yellow", "pink", "silver", "cream"})
EITHER = frozenset({"grey", "red", "orange"})
LUMINANCE_ONLY = frozenset({"dark", "light"})
HUE_ALIASES = {"silver": "grey", "gray": "grey"}
COLOUR_WORDS = DARK | LIGHT | EITHER | {"gray"}

UPPER_GARMENTS = frozenset({"shirt", "top", "t-shirt", "jacket", "coat",
                            "hoodie", "sweater", "blouse", "dress"})
LOWER_GARMENTS = frozenset({"pants", "shorts", "trousers", "jeans", "skirt",
                            "leggings"})
BODY_WORDS = {"sedan": "sedan", "hatchback": "hatchback", "suv": "suv",
              "van": "van", "minivan": "van", "pickup": "pickup",
              "truck": "truck", "bus": "bus"}
STATE_PHRASES = (("starting to drive", "moving"), ("driving", "moving"),
                 ("moving", "moving"), ("parked", "static"),
                 ("stopped", "static"))

# Outcomes of one slot comparison.
CORRECT, WRONG, ABSTAIN = "correct", "wrong", "abstain"


# --- GT parsing ---

def _colours_in(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z]+", text.lower())
                     if w in COLOUR_WORDS)


def parse_gt_person(desc: str) -> dict[str, Any]:
    """"adult man, dark shirt, light pants, black bag" -> slots.

    Colour slots are SETS (a GT colour may name several). Missing = UNKNOWN,
    which the scorer treats as "not scorable", never as a label.
    """
    toks = [t.strip().lower() for t in desc.split(",") if t.strip()]
    out: dict[str, Any] = {"sex": UNKNOWN, "age": UNKNOWN,
                           "upper_color": UNKNOWN, "lower_color": UNKNOWN,
                           "carrying": UNKNOWN}
    if not toks:
        return out
    who = set(toks[0].split())
    if who & {"adult"}:
        out["age"] = "adult"
    elif who & {"young", "child", "boy", "girl", "baby"}:
        out["age"] = "child"
    if who & {"man", "male", "boy"}:
        out["sex"] = "male"
    elif who & {"woman", "female", "girl"}:
        out["sex"] = "female"

    rest = toks[1:]
    n_clothing = 0
    for t in rest[:2]:
        words = set(t.split())
        if words & {"clothes", "clothing"}:
            out["upper_color"] = out["lower_color"] = _colours_in(t) or UNKNOWN
            n_clothing = 1
            break
        if words & UPPER_GARMENTS and out["upper_color"] == UNKNOWN:
            out["upper_color"] = _colours_in(t) or UNKNOWN
            n_clothing += 1
        elif words & LOWER_GARMENTS and out["lower_color"] == UNKNOWN:
            out["lower_color"] = _colours_in(t) or UNKNOWN
            n_clothing += 1
        else:
            break
    if n_clothing:
        out["carrying"] = len(rest) > n_clothing
    return out


def parse_gt_vehicle(desc: str) -> dict[str, Any]:
    """"white and black, hatchback, parked" -> slots."""
    low = desc.lower()
    body = next((BODY_WORDS[w] for w in re.findall(r"[a-z]+", low)
                 if w in BODY_WORDS), UNKNOWN)
    state = next((s for phrase, s in STATE_PHRASES if phrase in low), UNKNOWN)
    return {"color": _colours_in(low) or UNKNOWN, "body_type": body,
            "state": state}


# --- slot comparison ---

def _hue(c: str) -> str:
    return HUE_ALIASES.get(c, c)


def _lightness(c: str) -> frozenset[str]:
    if c in DARK:
        return frozenset({"dark"})
    if c in LIGHT:
        return frozenset({"light"})
    if c in EITHER:
        return frozenset({"dark", "light"})
    return frozenset()


def compare_colour(gt: frozenset[str] | str, pred: str
                   ) -> tuple[str | None, str | None]:
    """(hue outcome, lightness outcome); None = not scorable for this GT."""
    if gt == UNKNOWN or not gt:
        return None, None
    gt_hues = {_hue(c) for c in gt if c not in LUMINANCE_ONLY}
    hue_scorable = bool(gt_hues)
    if pred == UNKNOWN:
        return (ABSTAIN if hue_scorable else None), ABSTAIN

    if not hue_scorable:
        hue = None
    elif pred in LUMINANCE_ONLY:
        hue = ABSTAIN            # the model gave luminance only; no hue claimed
    else:
        hue = CORRECT if _hue(pred) in gt_hues else WRONG

    gt_light = frozenset().union(*(_lightness(c) for c in gt))
    light = CORRECT if _lightness(pred) & gt_light else WRONG
    return hue, light


def _compare_exact(gt: str, pred: str) -> str | None:
    if gt == UNKNOWN:
        return None
    if pred == UNKNOWN:
        return ABSTAIN
    return CORRECT if gt == pred else WRONG


def _pred_state(s: str) -> str:
    return {"parked": "static", "stopped": "static"}.get(s, s)


def score_pair(gt: GTEvent, pred: Interaction) -> dict[str, str]:
    """Slot -> outcome for one matched pair. Unscorable slots are omitted."""
    pa, va = pred.person.attributes or {}, pred.vehicle.attributes or {}
    gp, gv = parse_gt_person(gt.person_desc), parse_gt_vehicle(gt.vehicle_desc)
    out: dict[str, str | None] = {}

    out["person.sex"] = _compare_exact(gp["sex"], pa.get("sex", UNKNOWN))
    out["person.age"] = _compare_exact(gp["age"], pa.get("age", UNKNOWN))
    for slot in ("upper_color", "lower_color"):
        hue, light = compare_colour(gp[slot], pa.get(slot, UNKNOWN))
        out[f"person.{slot}.hue"], out[f"person.{slot}.lightness"] = hue, light
    if gp["carrying"] != UNKNOWN:
        pc = pa.get("carrying", UNKNOWN)
        out["person.carrying"] = (ABSTAIN if pc == UNKNOWN
                                  else CORRECT if (pc == "yes") == gp["carrying"]
                                  else WRONG)

    hue, light = compare_colour(gv["color"], va.get("color", UNKNOWN))
    out["vehicle.color.hue"], out["vehicle.color.lightness"] = hue, light
    out["vehicle.body_type"] = _compare_exact(gv["body_type"],
                                              va.get("body_type", UNKNOWN))
    out["vehicle.state"] = _compare_exact(
        gv["state"], _pred_state(va.get("state", UNKNOWN)))
    return {k: v for k, v in out.items() if v is not None}


# --- aggregation ---

@dataclass
class _Tally:
    correct: int = 0
    wrong: int = 0
    abstain: int = 0

    def add(self, outcome: str) -> None:
        setattr(self, outcome, getattr(self, outcome) + 1)

    def as_dict(self) -> dict[str, Any]:
        answered = self.correct + self.wrong
        total = answered + self.abstain
        return {"correct": self.correct, "wrong": self.wrong,
                "abstain": self.abstain,
                "accuracy": round(self.correct / answered, 4) if answered else None,
                "coverage": round(answered / total, 4) if total else None}


@dataclass
class DescriptionReport:
    n_matched: int = 0
    n_scored: int = 0
    n_actor_mismatch: int = 0
    n_without_attributes: int = 0
    slots: dict[str, _Tally] = field(default_factory=dict)
    pairs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        person = _Tally()
        vehicle = _Tally()
        for k, t in self.slots.items():
            agg = person if k.startswith("person.") else vehicle
            for o in (CORRECT, WRONG, ABSTAIN):
                for _ in range(getattr(t, o)):
                    agg.add(o)
        return {
            "tiou_threshold": TIOU_PRIMARY,
            "n_matched": self.n_matched,
            "n_scored": self.n_scored,
            "n_actor_mismatch": self.n_actor_mismatch,
            "n_without_attributes": self.n_without_attributes,
            "person_all_slots": person.as_dict(),
            "vehicle_all_slots": vehicle.as_dict(),
            "per_slot": {k: self.slots[k].as_dict() for k in sorted(self.slots)},
            "pairs": self.pairs,
        }


def description_report(groups: Sequence[tuple[Sequence[Interaction],
                                               Sequence[GTEvent]]]
                       ) -> dict[str, Any]:
    """Score descriptions over per-clip (preds, gts) groups; matched within clip."""
    rep = DescriptionReport()
    for preds, gts in groups:
        result = match_events(preds, gts, tiou_thresh=TIOU_PRIMARY)
        for m in result.matches:
            p, g = preds[m.pred_idx], gts[m.gt_idx]
            rep.n_matched += 1
            if not anchors_agree(p, g):
                rep.n_actor_mismatch += 1
                continue
            if p.person.attributes is None and p.vehicle.attributes is None:
                rep.n_without_attributes += 1
                continue
            outcomes = score_pair(g, p)
            rep.n_scored += 1
            for k, o in outcomes.items():
                rep.slots.setdefault(k, _Tally()).add(o)
            rep.pairs.append({
                "gt_event_id": g.event_id, "interaction_id": p.interaction_id,
                "gt_person": g.person_desc, "pred_person": p.person.description,
                "gt_vehicle": g.vehicle_desc,
                "pred_vehicle": p.vehicle.description,
                "outcomes": outcomes,
            })
    return rep.as_dict()
