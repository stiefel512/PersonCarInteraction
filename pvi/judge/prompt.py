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

# Bump when draw_boxes() or crop_box() changes what the model is shown. The
# cache key derives from box GEOMETRY rather than rendered pixels (see
# PromptBundle.cache_key), so a change to the rendering itself would otherwise
# be invisible to the key and replay stale verdicts against different images.
RENDER_VERSION = 1

TYPE_DEFINITIONS = """\
- enter_vehicle: the person ends up inside the vehicle. They may be seen \
getting in, or they may simply stop being visible AT the vehicle because they \
are now inside it. Walking away afterwards does not happen - they do not \
reappear.
- exit_vehicle: the person starts out inside the vehicle. They may be seen \
getting out, or their FIRST appearance may already be at the vehicle, after \
which they move away. Somebody whose first appearance is at the vehicle - never \
seen approaching it from elsewhere - came out of it.
- open_close_door: the person opens and/or closes a door, boot or tailgate \
without a full entry or exit.
- load_unload: the person moves an object into or out of the vehicle.
- attend_vehicle: sustained deliberate physical engagement that is none of the \
above - leaning on it, inspecting it, cleaning it, reaching through a window.
- pass_by: NOT an interaction. The person walks or runs past the vehicle, or \
stands near it, without engaging with it - they were already walking before \
they reached it and they keep going afterwards. Proximity alone is never an \
interaction."""

SYSTEM_PROMPT = """\
You analyse short surveillance clips and decide whether a specific person is \
interacting with a specific vehicle. You answer only with JSON."""

USER_TEMPLATE = """\
These {n} images are frames from one clip, in time order, spanning {dur:.1f} \
seconds.

The person in question is outlined in RED. The vehicle in question is outlined \
in BLUE. Ignore every other person and vehicle.
{tracking_evidence}
Decide which one of these best describes what the RED person does with the BLUE \
vehicle across these frames:

{types}

Three rules that override everything else:
1. Proximity is not interaction. If the person walks or runs past the vehicle, \
or stands near it, without touching it or acting on it, the answer is pass_by - \
however close they get. When in doubt, answer pass_by.
2. Rule 1 does NOT apply when the notes above say the person appeared at the \
vehicle in open view, or was last seen at it in open view. Somebody who is \
simply there, not having walked in from off-camera, came out of the vehicle; \
somebody who stops being visible at it, without walking off-camera, got into it.
3. Judge only what is visible. Do not infer intent, ownership, or whether the \
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
    # Integer-quantized geometry the images were rendered from. See cache_key.
    geometry: tuple = ()

    def cache_key(self, salt: str = "") -> str:
        """Hash of the QUANTIZED GEOMETRY, the prompt text, and `salt`.

        Not the rendered pixels, and that is a deliberate reversal. Hashing
        image bytes made the cache correct but useless: any floating-point
        noise anywhere upstream changes a box in the fifth decimal, shifts a
        drawn rectangle by a fraction of a pixel, and invalidates the entry.

        Measured -- reinstalling the environment resolved torch 2.14.0+cu130
        instead of +cu132, which moved detection boxes by ~0.05 px. Every
        interaction's type, span, ids and confidence were unchanged, but the
        cache missed and the VLM re-ran, returning a different free-text note.
        A cache that only survives a byte-identical GPU stack cannot deliver
        what design-plan s7 promises: reproduction by a reviewer without one.

        So the key is what the images are DERIVED from -- frame indices and
        integer box coordinates -- plus RENDER_VERSION, which stands in for the
        drawing logic that the pixels used to capture implicitly. Bump that
        constant when the rendering changes.

        `salt` carries what is outside the prompt entirely: model id, revision,
        decode and pixel budgets. Without it a 32B run would replay 7B verdicts,
        which is precisely the comparison design-plan s6.4 asks for.
        """
        h = hashlib.sha256()
        h.update(f"v{RENDER_VERSION}".encode())
        h.update(repr(self.geometry).encode())
        h.update(repr(list(self.frame_indices)).encode())
        h.update(self.system.encode())
        h.update(self.user.encode())
        h.update(salt.encode())
        return h.hexdigest()


def quantize(box: Box) -> tuple[int, int, int, int]:
    """Round a box to integer pixels.

    The unit of quantization is one pixel because that is the unit the image is
    rasterized in anyway; anything finer is noise the renderer discards.
    """
    return tuple(int(round(v)) for v in box)


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
    # Quantize before drawing: sub-pixel jitter from a different GPU build must
    # not change the rasterized rectangle, or the rendering is as fragile as the
    # byte-hash cache key used to be.
    if vehicle is not None:
        dr.rectangle(quantize(vehicle), outline=VEHICLE_COLOUR, width=w)
    if person is not None:
        dr.rectangle(quantize(person), outline=PERSON_COLOUR, width=w)
    return img.crop(quantize(crop))


def tracking_evidence(cand: Candidate, meta: ClipMeta) -> str:
    """State what the tracker observed that the sampled frames cannot show.

    This is the hybrid design doing its job. Whether a person was *first seen*
    at the vehicle, or *last seen* at it, is a fact about the whole clip; a
    handful of crops cannot convey it, and it is precisely what separates an
    exit or an entry from someone walking past.

    Without it the VLM rejected `exit_vehicle` events as `pass_by` at 0.9
    confidence -- including one at tIoU 0.97 where all four proposal rules
    fired -- because a person already beside a car who then walks away looks
    exactly like a pass-by in isolation.

    Only structural observations are passed. No ground truth, and no hint about
    what the answer should be: the model is told what was seen, not what to
    conclude.
    """
    from ..propose.rules import R2_DEATH_NEAR, R3_BIRTH_NEAR, R4_DOOR_CHANGE

    lines = []
    ev = cand.evidence
    if R3_BIRTH_NEAR in cand.rules:
        if ev.get("born_at_frame_edge"):
            lines.append("- The RED person first became visible at the EDGE of "
                         "the picture, so they walked in from off-camera. They "
                         "did not come out of the BLUE vehicle.")
        else:
            lines.append("- The RED person's FIRST appearance anywhere in the "
                         "clip is here, at the BLUE vehicle, in open view and "
                         "not at the edge of the picture. They did not walk in "
                         "from off-camera.")
    if R2_DEATH_NEAR in cand.rules:
        if ev.get("died_at_frame_edge"):
            lines.append("- The RED person was last seen at the EDGE of the "
                         "picture, so they walked off-camera. They did not get "
                         "into the BLUE vehicle.")
        else:
            lines.append("- The RED person's LAST appearance anywhere in the "
                         "clip is here, at the BLUE vehicle, in open view and "
                         "not at the edge of the picture. They did not walk "
                         "off-camera.")
    if R4_DOOR_CHANGE in cand.rules:
        lines.append("- A door of the BLUE vehicle was detected open during "
                     "this span.")
    if cand.evidence.get("birth_at_clip_start"):
        lines.append("- The clip itself begins at this moment, so anything "
                     "before it is simply not recorded.")
    if not lines:
        return ""
    return ("\nWhat the tracker observed across the WHOLE clip, which these "
            "frames alone cannot show:\n" + "\n".join(lines) + "\n")


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

    # Exactly what the images were rendered from, quantized. Anything affecting
    # the pixels must appear here or in RENDER_VERSION.
    geometry = (quantize(crop),
                tuple(quantize(person_boxes[i]) if i in person_boxes else None
                      for i in indices),
                tuple(quantize(vehicle_boxes[i]) if i in vehicle_boxes else None
                      for i in indices))

    user = USER_TEMPLATE.format(
        n=len(images),
        dur=(span[1] - span[0] + 1) / meta.fps,
        tracking_evidence=tracking_evidence(cand, meta),
        types=TYPE_DEFINITIONS,
        type_list=", ".join(ALLOWED_TYPES),
    )
    return PromptBundle(images=images, frame_indices=indices,
                        system=SYSTEM_PROMPT, user=user, geometry=geometry)


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
