"""Structured appearance description of an accepted interaction's actors.

A SEPARATE VLM call from the adjudication, not extra fields on it. The judge
prompt's text is part of its cache key, so any edit there re-rolls every cached
verdict -- and the detection numbers in RESULT_ANALYSIS.md with them -- to change a field
the verdict does not depend on. Keeping description out of the judge prompt
means a description change can never move detection F1, and vice versa.

It runs only on ACCEPTED candidates, over the same rendered frames the judge saw
(RED person, BLUE vehicle), so it describes exactly the actors the verdict is
about.

Slots are closed-vocabulary wherever the ground truth can score them (colours,
sex, age, carrying, body type, state) and free text only for garment names.

`carrying` is yes/no, deliberately not an object name. As free text the model
named objects the ground truth does not contain -- "umbrella", "suitcase", and
on `mKzCQKTHizw_0` "gun" -- and a hallucinated weapon in a deliverable is the
kind of loaded claim ambiguity A6 rules out. Presence is all the GT can score
anyway. `unknown` is always allowed: a grayscale or night clip that
does not show a hue should abstain, not guess. `dark`/`light` exist for the same
reason -- they are what the labeler wrote when only luminance was visible.
"""
from __future__ import annotations

import json

from .prompt import PromptBundle

UNKNOWN = "unknown"

COLOURS: tuple[str, ...] = (
    "black", "white", "grey", "silver", "red", "blue", "green", "yellow",
    "orange", "brown", "beige", "pink", "purple", "dark", "light", UNKNOWN,
)
SEXES: tuple[str, ...] = ("male", "female", UNKNOWN)
AGES: tuple[str, ...] = ("adult", "child", UNKNOWN)
BODY_TYPES: tuple[str, ...] = (
    "sedan", "hatchback", "suv", "van", "pickup", "truck", "bus",
    "motorcycle", "other", UNKNOWN,
)
# `stopped` = stationary with a driver aboard / in traffic; `parked` = left.
# The scorer merges the two (evaluate/descriptions.py): the difference is often
# not visible within one short span.
STATES: tuple[str, ...] = ("parked", "stopped", "moving", UNKNOWN)
YES_NO: tuple[str, ...] = ("yes", "no", UNKNOWN)

PERSON_SLOTS = {"sex": SEXES, "age": AGES, "upper_color": COLOURS,
                "lower_color": COLOURS, "carrying": YES_NO}
PERSON_FREE = ("upper_garment", "lower_garment")
VEHICLE_SLOTS = {"color": COLOURS, "body_type": BODY_TYPES, "state": STATES}

SYSTEM_PROMPT = """\
You describe people and vehicles in surveillance footage. You report only what \
is visible and answer only with JSON."""

USER_TEMPLATE = """\
These {n} images are frames from one clip, in time order. One person is \
outlined in RED and one vehicle is outlined in BLUE. Describe ONLY those two; \
ignore everyone and everything else.

Fill every field. Use exactly one of the listed values where a list is given. \
Answer "unknown" whenever the frames do not show it clearly - for example a \
colour in grayscale or very dark footage. If only brightness is visible, use \
"dark" or "light" rather than guessing a hue.

{{"person": {{
   "sex": "<{sexes}>",
   "age": "<{ages}>",
   "upper_color": "<{colours}>",
   "upper_garment": "<a few words, e.g. t-shirt, jacket, coat, or unknown>",
   "lower_color": "<{colours}>",
   "lower_garment": "<a few words, e.g. trousers, jeans, shorts, skirt, or unknown>",
   "carrying": "<yes|no|unknown - is the person holding or carrying anything>"}},
 "vehicle": {{
   "color": "<{colours}>",
   "body_type": "<{bodies}>",
   "state": "<{states}>"}}}}"""


def user_prompt(n_images: int) -> str:
    return USER_TEMPLATE.format(
        n=n_images, sexes="|".join(SEXES), ages="|".join(AGES),
        colours="|".join(COLOURS), bodies="|".join(BODY_TYPES),
        states="|".join(STATES))


def rebundle(judge_bundle: PromptBundle) -> PromptBundle:
    """Same images and geometry as the judge saw; description text instead.

    The cache key covers the text, so this can never collide with a verdict.
    """
    return PromptBundle(images=judge_bundle.images,
                        frame_indices=judge_bundle.frame_indices,
                        system=SYSTEM_PROMPT,
                        user=user_prompt(len(judge_bundle.images)),
                        geometry=judge_bundle.geometry)


def _closed(value, vocab: tuple[str, ...]) -> str:
    """Normalize to the vocabulary; anything off-list becomes `unknown`.

    Off-list is not silently mapped to a near neighbour ("navy" -> "blue"):
    that mapping would be an unrecorded scoring decision made at parse time.
    """
    v = str(value or "").strip().lower().replace("gray", "grey")
    return v if v in vocab else UNKNOWN


def _free(value) -> str:
    v = str(value or "").strip()
    return v if v else UNKNOWN


def parse_response(text: str) -> dict | None:
    """Parse a description response into {"person": {...}, "vehicle": {...}}.

    Returns None when there is no usable JSON object at all; individual bad
    slots degrade to `unknown` instead, so one malformed field does not discard
    the rest of the description.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    p = obj.get("person") if isinstance(obj.get("person"), dict) else {}
    v = obj.get("vehicle") if isinstance(obj.get("vehicle"), dict) else {}
    if not p and not v:
        return None
    person = {k: _closed(p.get(k), vocab) for k, vocab in PERSON_SLOTS.items()}
    person.update({k: _free(p.get(k)) for k in PERSON_FREE})
    vehicle = {k: _closed(v.get(k), vocab) for k, vocab in VEHICLE_SLOTS.items()}
    return {"person": person, "vehicle": vehicle}


def _cloth(colour: str, garment: str) -> str:
    parts = [x for x in (colour, garment) if x != UNKNOWN]
    return " ".join(parts)


def render_person(a: dict) -> str:
    """Human-readable description, derived from the slots so the two can never
    disagree."""
    who = " ".join(x for x in (a["age"], a["sex"]) if x != UNKNOWN) or "person"
    parts = [who]
    for c, g in (("upper_color", "upper_garment"), ("lower_color", "lower_garment")):
        s = _cloth(a[c], a[g])
        if s:
            parts.append(s)
    if a["carrying"] == "yes":
        parts.append("carrying an object")
    return ", ".join(parts)


def render_vehicle(a: dict) -> str:
    parts = [x for x in (a["color"], a["body_type"], a["state"]) if x != UNKNOWN]
    return ", ".join(parts) or "vehicle"
