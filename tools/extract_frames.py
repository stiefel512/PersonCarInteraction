"""Extract every frame of each clip to data/frames/<clip_id>/NNNNNN.jpg.

Deterministic: no seeking, no frame dropping -- ffmpeg decodes linearly so the
Nth written JPEG is the Nth decoded frame, which is the index the labeler and
the pipeline both report.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

MAX_WIDTH = 1600  # 4K frames are downscaled for labeling only; indices are unaffected


def probe(clip: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
         "-show_entries", "format=duration",
         "-of", "json", str(clip)],
        capture_output=True, text=True, check=True,
    )
    d = json.loads(out.stdout)
    s, f = d["streams"][0], d["format"]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return {
        "clip_id": clip.stem,
        "width": s["width"],
        "height": s["height"],
        "fps": num / den,
        "n_frames": int(s["nb_read_frames"]),
        "duration_s": float(f["duration"]),
    }


def extract(clip: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip),
         "-vf", f"scale='min({MAX_WIDTH},iw)':-2:flags=bicubic",
         "-fps_mode", "passthrough", "-q:v", "3",
         str(out_dir / "%06d.jpg")],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=Path("Videos"))
    ap.add_argument("--out", type=Path, default=Path("data/frames"))
    args = ap.parse_args()

    meta = {}
    for clip in sorted(args.videos.glob("*.mp4")):
        info = probe(clip)
        extract(clip, args.out / info["clip_id"])
        written = len(list((args.out / info["clip_id"]).glob("*.jpg")))
        if written != info["n_frames"]:
            print(f"  ! {info['clip_id']}: probed {info['n_frames']}, wrote {written}"
                  f" -- using {written}")
            info["n_frames"] = written
        meta[info["clip_id"]] = info
        print(f"  {info['clip_id']}: {written} frames @ {info['fps']:.2f} fps")

    meta_path = args.out.parent / "clips.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {meta_path}")


if __name__ == "__main__":
    main()
