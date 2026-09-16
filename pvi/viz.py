"""Annotated-frame rendering for reported interactions.

Optional deliverable (design-plan s2). Its purpose is review, not evidence: the
`evidence` block in each output record is what justifies a detection. This makes
a reviewer able to check a verdict in a second rather than a minute.

Colour convention matches the VLM's visual prompt (`judge/prompt.py`) on purpose
-- red person, blue vehicle -- so what a reviewer sees is what the model saw.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .geometry import Box
from .judge.prompt import PERSON_COLOUR, VEHICLE_COLOUR
from .propose.spans import sample_frames
from .schema import ClipMeta, Interaction
from .video import read_frames

ACCEPT_COLOUR = (60, 200, 90)


def _line_width(img: Image.Image) -> int:
    return max(2, int(round(min(img.width, img.height) / 180)))


def render_interaction(frame: np.ndarray, inter: Interaction,
                       person_box: Box | None, vehicle_box: Box | None,
                       frame_idx: int, meta: ClipMeta) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB")
    dr = ImageDraw.Draw(img)
    w = _line_width(img)

    if vehicle_box is not None:
        dr.rectangle(vehicle_box, outline=VEHICLE_COLOUR, width=w)
    if person_box is not None:
        dr.rectangle(person_box, outline=PERSON_COLOUR, width=w)

    caption = (f"{inter.type}  conf={inter.confidence:.2f}  "
               f"f{inter.frame_start}-{inter.frame_end}  "
               f"({meta.to_s(frame_idx):.1f}s)")
    pad = 4 * w
    dr.rectangle((0, 0, img.width, pad * 3), fill=(0, 0, 0))
    dr.text((pad, pad), caption, fill=ACCEPT_COLOUR)
    return img


def render_clip(clip_path: str | Path, meta: ClipMeta,
                interactions: Sequence[Interaction],
                person_boxes: dict[int, dict[int, Box]],
                vehicle_boxes: dict[int, dict[int, Box]],
                outdir: str | Path, n_per_event: int = 4,
                max_width: int = 1280) -> list[Path]:
    """Write `n_per_event` annotated frames per reported interaction.

    Frames are decoded once for the whole clip rather than per event, since
    linear decode is the expensive part and several events usually overlap.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    wanted: dict[int, list[tuple[Interaction, int]]] = {}
    for n, inter in enumerate(interactions, start=1):
        for f in sample_frames((inter.frame_start, inter.frame_end), n_per_event):
            wanted.setdefault(f, []).append((inter, n))
    if not wanted:
        return []

    frames = read_frames(clip_path, sorted(wanted), meta)
    written: list[Path] = []
    for f in sorted(wanted):
        for inter, n in wanted[f]:
            img = render_interaction(
                frames[f], inter,
                person_boxes.get(inter.person.track_id, {}).get(f),
                vehicle_boxes.get(inter.vehicle.track_id, {}).get(f),
                f, meta)
            if img.width > max_width:
                scale = max_width / img.width
                img = img.resize((max_width, int(img.height * scale)), Image.LANCZOS)
            p = out / f"{meta.clip_id}__i{n:03d}_f{f:06d}.jpg"
            img.save(p, quality=90)
            written.append(p)
    return written
