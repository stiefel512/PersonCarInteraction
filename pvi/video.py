"""Deterministic frame access, straight from the MP4 at native resolution.

Two things this module exists to guarantee:

1. **Native resolution.** `data/frames/` is downscaled to max width 1600 for the
   labeling UI, which makes `gt1125_06` 1600x900 instead of 3840x2160. Reading
   those JPEGs would silently answer the small-object question on 2.4x-downscaled
   pixels. Nothing here touches that directory.

2. **Frame indices mean one thing.** Indices are 0-based and count decoded
   frames in presentation order, with no seeking and no frame dropping -- the
   same convention `tools/extract_frames.py` used, so the frozen ground truth
   lines up. (Those JPEGs are named from 000001.jpg, so GT frame 0 is
   000001.jpg; the offset is in the filename, not in the index.)

Seeking is avoided deliberately. Seeking in a long-GOP H.264 stream lands on a
keyframe and the decoder's notion of "frame N" after a seek is codec- and
version-dependent -- exactly the kind of thing that makes two runs disagree.
Linear decode is slower and answers the same question every time.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .schema import ClipMeta


def probe(path: str | Path) -> ClipMeta:
    """Read clip metadata with ffprobe, counting frames rather than trusting
    the container's nb_frames (which is often absent or wrong)."""
    p = Path(path)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
         "-of", "json", str(p)],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return ClipMeta(clip_id=p.stem, width=int(s["width"]), height=int(s["height"]),
                    fps=num / den, n_frames=int(s["nb_read_frames"]))


def iter_frames(path: str | Path, meta: ClipMeta | None = None
                ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(index, frame_rgb)` for every frame, in order, at native size.

    Streams through a pipe rather than materializing the clip: 600 frames of 4K
    RGB is ~15 GB, which does not fit anywhere sensible.
    """
    m = meta or probe(path)
    frame_bytes = m.width * m.height * 3
    # `-fps_mode passthrough` (not the removed `-vsync 0`): emit exactly the
    # decoded frames, no duplication or dropping to hit a target rate. ffmpeg 9
    # removed `-vsync` outright, so the old spelling fails with
    # "Unrecognized option", producing zero frames rather than an obvious error.
    # stderr goes to a temp file rather than a pipe: an undrained stderr pipe
    # deadlocks ffmpeg once it fills, and a 600-frame 4K decode has plenty of
    # time to get there.
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(path),
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-"],
            stdout=subprocess.PIPE, stderr=errf,
        )
        assert proc.stdout is not None
        try:
            idx = 0
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                yield idx, np.frombuffer(buf, np.uint8).reshape(m.height, m.width, 3)
                idx += 1
        finally:
            # Close the pipe before waiting, or ffmpeg blocks writing into a full
            # buffer when the caller abandons the iterator early (`read_frames`
            # does exactly that).
            proc.stdout.close()
            rc = proc.wait()
            if idx == 0 and rc != 0:
                errf.seek(0)
                raise RuntimeError(
                    f"ffmpeg produced no frames for {Path(path).name} "
                    f"(exit {rc}): {errf.read().decode(errors='replace').strip()}"
                )


def read_frames(path: str | Path, indices: Sequence[int],
                meta: ClipMeta | None = None) -> dict[int, np.ndarray]:
    """Decode just the requested frames, by linear scan, stopping at the last one.

    Returns a dict rather than a list so a caller cannot silently mistake
    position in the result for frame index -- the single most likely way to
    misalign a VLM crop with the span it is supposed to show.
    """
    want = set(indices)
    if not want:
        return {}
    last = max(want)
    out: dict[int, np.ndarray] = {}
    for i, frame in iter_frames(path, meta):
        if i in want:
            out[i] = frame.copy()   # the buffer is reused; copy before keeping
        if i >= last:
            break
    missing = want - set(out)
    if missing:
        raise IndexError(f"{Path(path).name}: frames {sorted(missing)} not decoded "
                         f"(clip has {(meta or probe(path)).n_frames} frames)")
    return out


def clip_path(videos_dir: str | Path, clip_id: str) -> Path:
    p = Path(videos_dir) / f"{clip_id}.mp4"
    if not p.exists():
        raise FileNotFoundError(p)
    return p
