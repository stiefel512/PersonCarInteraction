"""Prompt construction for the VLM judge: frame selection, cropping, visual
prompting, and the text itself.

Visual prompting is not decoration. In `iMGR_0AG3a8_2_3` there are several
pedestrians and multiple vehicles in frame at once; a crop alone cannot tell the
model *which* person and *which* vehicle the question is about. Drawing the
person in red and the vehicle in blue, and naming those colours in the prompt,
is what makes the question well-posed.

The type definitions in the prompt are load-bearing rather than explanatory.
`attend_vehicle` (n=2) and `pass_by` are the hard boundary and there are not
enough instances to tune a threshold for it, so the distinction has to come from
the wording here (design-plan s6.5). That is a known, accepted weakness.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..geometry import Box, dilate, union_box
from ..propose.rules import Candidate
from ..schema import INTERACTION_TYPES, ClipMeta
from ..propose.spans import dilate_span, sample_frames

PERSON_COLOUR = (255, 40, 40)      # red
VEHICLE_COLOUR = (40, 120, 255)    # blue

TYPE_DEFINITIONS = """\
- enter_vehicle: the person moves from outside the vehicle to inside it. They \
stop being visible because they are now in the vehicle.
- exit_vehicle: the person moves from inside the vehicle to outside it. They \
first become visible at the vehicle.
- open_close_door: the person opens and/or closes a door, boot or tailgate \
without a full entry or exit.
- load_unload: the person moves an object into or out of the vehicle.
- attend_vehicle: sustained deliberate physical engagement that is none of the \
above - leaning on it, inspecting it, cleaning it, reaching through a window.
- pass_by: NOT an interaction. The person merely walks or runs near the \
vehicle, or stands near it, without deliberately engaging with it. Proximity \
alone is never an interaction. Choose this whenever the person does not touch \
the vehicle or act on it."""

SYSTEM_PROMPT = """\
You analyse short surveillance clips and decide whether a specific person is \
interacting with a specific vehicle. You answer only with JSON."""

USER_TEMPLATE = """\
These {n} images are frames from one clip, in time order, spanning {dur:.1f} \
seconds.

The person in question is outlined in RED. The vehicle in question is outlined \
in BLUE. Ignore every other person and vehicle.

Decide which one of these best describes what the RED person does with the BLUE \
vehicle across these frames:

{types}

Two rules that override everything else:
1. If the person only passes by, walks near, runs near or stands near the \
vehicle without deliberately engaging with it, the answer is pass_by.
2. Judge only what is visible. Do not infer intent, ownership, or whether the \
action is authorised.

Answer with JSON and nothing else:
{{"type": "<one of: {type_list}>",
 "confidence": <0.0 to 1.0>,
 "person": "<build/approx age, upper garment + colour, lower garment + colour, \
carried object or 'none'>",
 "vehicle": "<colour, body type, state (parked/moving), position in frame>",
 "note": "<one sentence describing what happens>"}}"""

ALLOWED_TYPES = tuple(INTERACTION_TYPES) + ("pass_by",)


@dataclass
class PromptBundle:
    images: list[Image.Image]
    frame_indices: list[int]
    system: str
    user: str

    def cache_key(self) -> str:
        """Hash of the exact image bytes plus the prompt text.

        Keyed on the rendered images rather than on frame indices and box
        coordinates, because a change in cropping or box drawing must invalidate
        the entry -- otherwise a cached verdict would be replayed against
        different pixels. This is what lets the committed cache reproduce
        results without a GPU (design-plan s7).
        """
        h = hashlib.sha256()
        for im in self.images:
            h.update(im.tobytes())
            h.update(f"{im.size}".encode())
        h.update(self.system.encode())
        h.update(self.user.encode())
        return h.hexdigest()


def crop_box(person_boxes: Sequence[Box], vehicle_boxes: Sequence[Box],
             margin: float, meta: ClipMeta) -> Box:
    """One fixed crop for the whole candidate, not a per-frame crop.

    Fixed because a crop that follows the person makes the vehicle appear to
    move, which is precisely the cue the model needs to read correctly. The
    union over all frames keeps both actors in view for the whole span.
    """
    boxes = list(person_boxes) + list(vehicle_boxes)
    u = boxes[0]
    for b in boxes[1:]:
        u = union_box(u, b)
    return dilate(u, margin, bounds=(meta.width, meta.height))


def draw_boxes(frame: np.ndarray, person: Box | None, vehicle: Box | None,
               crop: Box) -> Image.Image:
    """Draw the two boxes, then crop. Order matters: drawing after cropping
    would need the boxes re-expressed in crop coordinates, which is an easy
    off-by-one; drawing first keeps everything in frame coordinates."""
    img = Image.fromarray(frame).convert("RGB")
    dr = ImageDraw.Draw(img)
    # Line width scales with frame size so the box is visible at 352x288 and
    # not overwhelming at 4K.
    w = max(2, int(round(min(img.width, img.height) / 180)))
    if vehicle is not None:
        dr.rectangle(vehicle, outline=VEHICLE_COLOUR, width=w)
    if person is not None:
        dr.rectangle(person, outline=PERSON_COLOUR, width=w)
    return img.crop(tuple(int(round(v)) for v in crop))


def build(cand: Candidate, meta: ClipMeta, frames: dict[int, np.ndarray],
          person_boxes: dict[int, Box], vehicle_boxes: dict[int, Box],
          n_frames: int, context_pad_s: float, crop_margin: float) -> PromptBundle:
    span = dilate_span(cand.span, meta.to_frames(context_pad_s), meta.n_frames)
    indices = [i for i in sample_frames(span, n_frames) if i in frames]

    pb = [person_boxes[i] for i in indices if i in person_boxes]
    vb = [vehicle_boxes[i] for i in indices if i in vehicle_boxes]
    crop = crop_box(pb or [(0, 0, meta.width, meta.height)],
                    vb or [(0, 0, meta.width, meta.height)], crop_margin, meta)

    images = [draw_boxes(frames[i], person_boxes.get(i), vehicle_boxes.get(i), crop)
              for i in indices]

    user = USER_TEMPLATE.format(
        n=len(images),
        dur=(span[1] - span[0] + 1) / meta.fps,
        types=TYPE_DEFINITIONS,
        type_list=", ".join(ALLOWED_TYPES),
    )
    return PromptBundle(images=images, frame_indices=indices,
                        system=SYSTEM_PROMPT, user=user)


def parse_response(text: str) -> dict | None:
    """Extract the JSON object from a model response.

    Tolerant of a fenced block or surrounding prose, because a 7B model will
    sometimes add both despite the instruction. Returns None rather than raising
    on unparseable output, so one bad response degrades that single candidate
    instead of aborting the clip.
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
    if not isinstance(obj, dict) or obj.get("type") not in ALLOWED_TYPES:
        return None
    try:
        obj["confidence"] = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
    except (TypeError, ValueError):
        obj["confidence"] = 0.0
    for k in ("person", "vehicle", "note"):
        obj[k] = str(obj.get(k, "") or "")
    return obj
