#!/usr/bin/env python3
"""Style C: text-only lyric video (words build up on black/white flicker screens).

Usage: python3 lyric_video.py spec.json out.mp4

Spec (JSON):
{
  "audio": "path/to/audio.wav or video.mp4",
  "start": 0.0,            # where to start in the audio (s)
  "end": 15.3,             # where to stop in the audio (s)
  "size": [1080, 1920],
  "fps": 30,
  "snap": true,            # nudge each event onto the nearest audio onset
  "events": [              # each event holds until the next one
    {"t": 0.0, "text": "...", "bg": "white"},
    {"t": 3.75, "text": "tu", "bg": "white"},
    {"t": 4.0, "text": "tu veux\nla plus", "fx": "blur"}
  ]
}
"bg" carries over from the previous event when omitted. "fx": "blur" smears
the text in over a few frames (the ghosted look in the references).
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "TikTokSans_500Medium.ttf")
# Font size as a fraction of frame width, by line count: fewer lines = bigger.
SIZE_BY_LINES = {1: 0.125, 2: 0.092, 3: 0.074, 4: 0.062}
MAX_BLOCK_W = 0.78
WHITE = (226, 250)  # background flicker range
BLACK = (26, 36)
INK = {"white": 18, "black": 236}


def load_onsets(audio, start, end):
    import librosa

    sr = 22050
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(end - start), "-i", audio,
         "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    y = np.frombuffer(pcm, np.float32)
    return librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)


def snap(t, onsets, window=0.12):
    if len(onsets) == 0:
        return t
    i = int(np.argmin(np.abs(onsets - t)))
    return float(onsets[i]) if abs(onsets[i] - t) <= window else t


def text_layer(text, W, H):
    """Return an alpha mask (float 0..1) with the text block centred, lines left-aligned."""
    lines = text.split("\n")
    size = int(W * SIZE_BY_LINES.get(len(lines), 0.04))
    while True:
        font = ImageFont.truetype(FONT, size)
        widths = [font.getlength(line) for line in lines]
        if max(widths) <= W * MAX_BLOCK_W or size < 20:
            break
        size -= 2
    asc, desc = font.getmetrics()
    lh = int((asc + desc) * 1.02)
    bw, bh = int(max(widths)), lh * len(lines)
    x0, y0 = (W - bw) // 2, (H - bh) // 2
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((x0, y0 + i * lh), line, font=font, fill=255)
    return img


def main(spec_path, out_path):
    spec = json.load(open(spec_path))
    W, H = spec.get("size", [1080, 1920])
    fps = spec.get("fps", 30)
    start, end = spec.get("start", 0.0), spec["end"]
    dur = end - start
    events = sorted(spec["events"], key=lambda e: e["t"])

    if spec.get("snap"):
        onsets = load_onsets(spec["audio"], start, end)
        for e in events[1:]:
            e["t"] = snap(e["t"], onsets)

    bg = "white"
    for e in events:
        bg = e.setdefault("bg", bg)

    rng = np.random.default_rng(7)
    # Grain: a few pre-rendered half-res noise tiles, cycled.
    grain = [
        np.asarray(Image.fromarray(rng.normal(128, 40, (H // 2, W // 2)).clip(0, 255).astype(np.uint8)).resize((W, H)),
                   dtype=np.float32) - 128
        for _ in range(6)
    ]
    masks = {}

    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-ss", str(start), "-t", str(dur), "-i", spec["audio"],
         "-map", "0:v", "-map", "1:a",
         "-af", f"loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=out:st={max(dur - 0.3, 0)}:d=0.3",
         "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-tune", "grain", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-shortest", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)

    level = None
    n_frames = int(round(dur * fps))
    ei = 0
    for f in range(n_frames):
        t = f / fps
        while ei + 1 < len(events) and events[ei + 1]["t"] <= t:
            ei += 1
        ev = events[ei]
        lo, hi = WHITE if ev["bg"] == "white" else BLACK
        # Flicker: hold a brightness for ~3 frames, then jump; reset on bg change.
        if level is None or not (lo <= level <= hi) or f % 3 == 0:
            level = rng.uniform(lo, hi)
        frame = np.full((H, W), level, np.float32) + grain[(f // 2) % len(grain)] * 0.07

        key = ev["text"]
        if key not in masks:
            masks[key] = text_layer(key, W, H)
        mask = masks[key]
        since = t - ev["t"]
        if ev.get("fx") == "blur" and since < 0.2:
            radius = 14 * (1 - since / 0.2)
            m = np.asarray(mask.filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255
            m = np.maximum(m * 1.6, np.asarray(mask, np.float32) / 255 * (since / 0.2)).clip(0, 1)
        else:
            m = np.asarray(mask, np.float32) / 255
        frame = frame * (1 - m) + INK[ev["bg"]] * m
        ff.stdin.write(frame.clip(0, 255).astype(np.uint8).tobytes())

    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"wrote {out_path} ({dur:.2f}s, {len(events)} events)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
