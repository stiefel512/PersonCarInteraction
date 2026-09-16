"""Minimal frame-accurate span labeler for person-vehicle interactions.

Stdlib + Pillow only, so it runs on the system interpreter without the
torch venv. Reads data/clips.json + data/frames/, writes data/ground_truth.json.

Keys
  space        play / pause
  <- ->        step +-1 frame        Shift+<- ->   step +-10
  PgUp PgDn    step +-25             Home End      first / last frame
  [ ]          mark span start / end at current frame
  t            cycle interaction type
  a            toggle 'ambiguous'
  Return       commit the staged event
  x            delete selected event from the list
  n b          next / previous clip
  s            save
Mouse
  left-click   set person anchor (normalized, at current frame)
  right-click  set vehicle anchor
"""
from __future__ import annotations

import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

TYPES = [
    "enter_vehicle",
    "exit_vehicle",
    "open_close_door",
    "load_unload",
    "attend_vehicle",
    "pass_by",          # explicit negative: near-miss, NOT an interaction
]

CANVAS_W, CANVAS_H = 1120, 700
ROOT = Path(__file__).resolve().parent.parent
CLIPS_JSON = ROOT / "data" / "clips.json"
FRAMES_DIR = ROOT / "data" / "frames"
GT_JSON = ROOT / "data" / "ground_truth.json"


class Labeler:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.clips: dict = json.loads(CLIPS_JSON.read_text())
        self.clip_ids = sorted(self.clips)
        self.events: list[dict] = []
        if GT_JSON.exists():
            self.events = json.loads(GT_JSON.read_text())["events"]

        self.ci = 0
        self.fi = 0
        self.playing = False
        self.span: list[int | None] = [None, None]
        self.p_anchor: dict | None = None
        self.v_anchor: dict | None = None
        self._photo = None
        self._scale = (1.0, 0, 0)  # scale, offset_x, offset_y

        self._build_ui()
        self._load_clip(0)

    # ---------- ui ----------

    def _build_ui(self) -> None:
        self.root.title("GT labeler")
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main, width=CANVAS_W, height=CANVAS_H,
                                bg="#1a1a1a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", lambda e: self._set_anchor(e, "person"))
        self.canvas.bind("<Button-3>", lambda e: self._set_anchor(e, "vehicle"))

        side = ttk.Frame(main, padding=8)
        side.grid(row=0, column=1, sticky="ns")

        self.hdr = ttk.Label(side, text="", font=("TkDefaultFont", 11, "bold"))
        self.hdr.pack(anchor="w")
        self.sub = ttk.Label(side, text="", foreground="#555")
        self.sub.pack(anchor="w", pady=(0, 8))

        self.scrub = ttk.Scale(side, from_=0, to=1, orient="horizontal",
                               length=300, command=self._on_scrub)
        self.scrub.pack(fill="x", pady=(0, 10))

        ttk.Label(side, text="type  (t)").pack(anchor="w")
        self.type_var = tk.StringVar(value=TYPES[0])
        ttk.OptionMenu(side, self.type_var, TYPES[0], *TYPES).pack(fill="x")

        self.span_lbl = ttk.Label(side, text="span: - .. -")
        self.span_lbl.pack(anchor="w", pady=(8, 0))
        self.anchor_lbl = ttk.Label(side, text="anchors: none", foreground="#555")
        self.anchor_lbl.pack(anchor="w")

        self.fields: dict[str, tk.Entry] = {}
        for key, label in [("person_desc", "person  (build, upper+colour, lower+colour, carrying)"),
                           ("vehicle_desc", "vehicle  (colour, body type, state, position)"),
                           ("note", "note  (free text)"),
                           ("event_group_id", "group id  (blank = own)")]:
            ttk.Label(side, text=label, wraplength=300).pack(anchor="w", pady=(8, 0))
            e = ttk.Entry(side, width=42)
            e.pack(fill="x")
            self.fields[key] = e

        self.amb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="ambiguous (a)", variable=self.amb_var).pack(anchor="w", pady=6)
        ttk.Button(side, text="Add event  (Return)", command=self._commit).pack(fill="x")

        ttk.Label(side, text="events in this clip").pack(anchor="w", pady=(12, 0))
        self.listbox = tk.Listbox(side, height=11, width=46)
        self.listbox.pack(fill="both", expand=True)
        ttk.Button(side, text="Delete selected  (x)", command=self._delete).pack(fill="x", pady=(4, 0))
        ttk.Button(side, text="Save  (s)", command=self._save).pack(fill="x", pady=(4, 0))

        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        for seq, fn in [
            ("<Left>", lambda e: self._step(-1)), ("<Right>", lambda e: self._step(1)),
            ("<Shift-Left>", lambda e: self._step(-10)), ("<Shift-Right>", lambda e: self._step(10)),
            ("<Prior>", lambda e: self._step(-25)), ("<Next>", lambda e: self._step(25)),
            ("<Home>", lambda e: self._goto(0)), ("<End>", lambda e: self._goto(self.n - 1)),
            ("<space>", lambda e: self._toggle_play()),
            ("<bracketleft>", lambda e: self._mark(0)), ("<bracketright>", lambda e: self._mark(1)),
            ("t", lambda e: self._cycle_type()), ("a", lambda e: self.amb_var.set(not self.amb_var.get())),
            ("<Return>", lambda e: self._commit()), ("x", lambda e: self._delete()),
            ("n", lambda e: self._load_clip(self.ci + 1)), ("b", lambda e: self._load_clip(self.ci - 1)),
            ("s", lambda e: self._save()),
        ]:
            self.root.bind(seq, self._guard(fn))

    def _guard(self, fn):
        """Ignore hotkeys while a text field has focus."""
        def wrapped(event):
            if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry)):
                return None
            return fn(event)
        return wrapped

    # ---------- state ----------

    @property
    def clip_id(self) -> str:
        return self.clip_ids[self.ci]

    @property
    def meta(self) -> dict:
        return self.clips[self.clip_id]

    @property
    def n(self) -> int:
        return len(self.frames)

    def _load_clip(self, idx: int) -> None:
        self.ci = idx % len(self.clip_ids)
        self.frames = sorted((FRAMES_DIR / self.clip_id).glob("*.jpg"))
        if not self.frames:
            messagebox.showerror("missing frames",
                                 f"No frames for {self.clip_id}. Run tools/extract_frames.py")
            return
        self.playing = False
        self.span = [None, None]
        self.p_anchor = self.v_anchor = None
        self.scrub.configure(to=self.n - 1)
        self._refresh_list()
        self._goto(0)

    def _goto(self, i: int) -> None:
        self.fi = max(0, min(self.n - 1, i))
        self._draw()

    def _step(self, d: int) -> None:
        self.playing = False
        self._goto(self.fi + d)

    def _on_scrub(self, val: str) -> None:
        i = int(float(val))
        if i != self.fi:
            self.playing = False
            self._goto(i)

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self._tick()

    def _tick(self) -> None:
        if not self.playing:
            return
        if self.fi >= self.n - 1:
            self.playing = False
            return
        self._goto(self.fi + 1)
        self.root.after(int(1000 / max(1.0, self.meta["fps"])), self._tick)

    # ---------- drawing ----------

    def _draw(self) -> None:
        img = Image.open(self.frames[self.fi])
        s = min(CANVAS_W / img.width, CANVAS_H / img.height)
        w, h = int(img.width * s), int(img.height * s)
        ox, oy = (CANVAS_W - w) // 2, (CANVAS_H - h) // 2
        self._scale = (s, ox, oy)
        self._photo = ImageTk.PhotoImage(img.resize((w, h), Image.BILINEAR))

        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)
        for anchor, colour, tag in [(self.p_anchor, "#ff4d4d", "P"), (self.v_anchor, "#4da6ff", "V")]:
            if anchor:
                x, y = ox + anchor["x"] * w, oy + anchor["y"] * h
                self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, outline=colour, width=2)
                self.canvas.create_text(x, y - 16, text=tag, fill=colour,
                                        font=("TkDefaultFont", 10, "bold"))

        t = self.fi / self.meta["fps"]
        self.hdr.config(text=f"[{self.ci + 1}/{len(self.clip_ids)}]  {self.clip_id}")
        self.sub.config(text=f"frame {self.fi}/{self.n - 1}    t = {t:.3f} s    "
                             f"{self.meta['width']}x{self.meta['height']} @ {self.meta['fps']:.2f} fps")
        self.scrub.set(self.fi)
        self._update_span_lbl()

    def _update_span_lbl(self) -> None:
        a, b = self.span
        fps = self.meta["fps"]
        fmt = lambda v: "-" if v is None else f"{v} ({v / fps:.2f}s)"
        self.span_lbl.config(text=f"span: {fmt(a)} .. {fmt(b)}")
        bits = []
        if self.p_anchor:
            bits.append(f"P@{self.p_anchor['frame']}")
        if self.v_anchor:
            bits.append(f"V@{self.v_anchor['frame']}")
        self.anchor_lbl.config(text="anchors: " + (", ".join(bits) if bits else "none"))

    # ---------- editing ----------

    def _set_anchor(self, event, which: str) -> None:
        s, ox, oy = self._scale
        img = Image.open(self.frames[self.fi])
        w, h = int(img.width * s), int(img.height * s)
        x, y = (event.x - ox) / w, (event.y - oy) / h
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return
        a = {"frame": self.fi, "x": round(x, 4), "y": round(y, 4)}
        if which == "person":
            self.p_anchor = a
        else:
            self.v_anchor = a
        self._draw()

    def _mark(self, end: int) -> None:
        self.span[end] = self.fi
        self._update_span_lbl()

    def _cycle_type(self) -> None:
        i = TYPES.index(self.type_var.get())
        self.type_var.set(TYPES[(i + 1) % len(TYPES)])

    def _commit(self) -> None:
        a, b = self.span
        if a is None or b is None:
            messagebox.showwarning("incomplete", "Mark both ends with [ and ].")
            return
        if b < a:
            a, b = b, a
        fps = self.meta["fps"]
        # max existing index + 1, never count+1: deleting then adding must not reuse an id
        used = [int(m.group(1)) for e in self.events if e["clip_id"] == self.clip_id
                for m in [re.fullmatch(rf"{re.escape(self.clip_id)}__e(\d+)", e["event_id"])] if m]
        eid = f"{self.clip_id}__e{max(used, default=0) + 1:03d}"
        group = self.fields["event_group_id"].get().strip()
        self.events.append({
            "event_id": eid,
            "event_group_id": f"{self.clip_id}__{group}" if group else eid,
            "clip_id": self.clip_id,
            "type": self.type_var.get(),
            "frame_start": a,
            "frame_end": b,
            "time_start_s": round(a / fps, 3),
            "time_end_s": round(b / fps, 3),
            "person_desc": self.fields["person_desc"].get().strip(),
            "vehicle_desc": self.fields["vehicle_desc"].get().strip(),
            "person_anchor": self.p_anchor,
            "vehicle_anchor": self.v_anchor,
            "ambiguous": self.amb_var.get(),
            "note": self.fields["note"].get().strip(),
        })
        self.span = [None, None]
        self.p_anchor = self.v_anchor = None
        for k, e in self.fields.items():
            if k != "vehicle_desc":   # vehicle usually repeats across events
                e.delete(0, "end")
        self.amb_var.set(False)
        self._refresh_list()
        self._draw()

    def _delete(self) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        mine = [e for e in self.events if e["clip_id"] == self.clip_id]
        self.events.remove(mine[sel[0]])
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.listbox.delete(0, "end")
        for e in [x for x in self.events if x["clip_id"] == self.clip_id]:
            flag = "?" if e["ambiguous"] else " "
            self.listbox.insert(
                "end", f"{flag} {e['frame_start']:>4}-{e['frame_end']:<4} {e['type']:<16} "
                       f"{e['person_desc'][:18]}")

    def _save(self) -> None:
        GT_JSON.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "boundary_convention": "contact-based; see docs/labeling-protocol.md",
            "types": TYPES,
            "events": sorted(self.events, key=lambda e: (e["clip_id"], e["frame_start"])),
        }
        GT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
        self.root.title(f"GT labeler  --  saved {len(self.events)} events")


if __name__ == "__main__":
    if not CLIPS_JSON.exists():
        raise SystemExit("run tools/extract_frames.py first")
    tk_root = tk.Tk()
    Labeler(tk_root)
    tk_root.mainloop()
