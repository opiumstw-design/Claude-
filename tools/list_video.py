#!/usr/bin/env python3
"""Style B: numbered-list video (title + 1-5 list over a beat-cut clip montage).

Usage: python3 list_video.py spec.json out.mp4

Spec (JSON):
{
  "audio": "path/to/audio source",
  "start": 0.0, "end": 11.8,         # section of the audio to use (s)
  "size": [1080, 1920], "fps": 30,
  "cut_every": 2,                     # cut to the next clip every N beats
  "title": "TOP 5 WAYS TO BE NONCHALANT",
  "items": ["use both airpods", "..."],
  "reveal": [1, 2, 3, 4, 5],          # order the items fill in ("ranking" look: e.g. [3, 5, 1, 4, 2])
  "clips": [{"src": "media/clips/a.mov", "in": 2.0}, ...]   # used in order, cycled if short
}
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"


def beats(audio, start, end):
    import librosa

    sr = 22050
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(end - start), "-i", audio,
         "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    y = np.frombuffer(pcm, np.float32)
    _, b = librosa.beat.beat_track(y=y, sr=sr, units="time")
    b = [float(x) for x in b]
    if len(b) < 2:
        return b
    # The tracker sometimes drops out mid-song; fill gaps (and the tail) on the median beat grid.
    step = float(np.median(np.diff(b)))
    filled = [b[0]]
    for t in b[1:] + [len(y) / sr + step]:
        while t - filled[-1] > 1.5 * step:
            filled.append(filled[-1] + step)
        filled.append(t)
    return filled[:-1]


def wrap(text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def draw_outlined(img, xy, text, font, anchor="la"):
    """White text with a thin dark outline and soft shadow (TikTok caption look)."""
    W, H = img.size
    shadow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shadow).text((xy[0] + 3, xy[1] + 4), text, font=font, fill=170, anchor=anchor)
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    img.paste(Image.new("RGBA", (W, H), (0, 0, 0, 255)), (0, 0), shadow)
    ImageDraw.Draw(img).text(xy, text, font=font, fill=(255, 255, 255, 255), anchor=anchor,
                             stroke_width=3, stroke_fill=(0, 0, 0, 255))


def overlay(spec, shown, W, H):
    """RGBA overlay: title plus the numbered list with only items in `shown` filled in."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    tfont = ImageFont.truetype(BOLD, int(W * 0.062))
    y = int(H * 0.155)
    for line in wrap(spec["title"].upper(), tfont, W * 0.84):
        draw_outlined(img, (W // 2, y), line, tfont, anchor="ma")
        y += int(tfont.size * 1.18)

    lfont = ImageFont.truetype(SEMI, int(W * 0.05))
    items = spec["items"]
    rows = [f"{i + 1}. {t}" for i, t in enumerate(items)]
    block_w = max(lfont.getlength(r) for r in rows)
    x = int((W - min(block_w, W * 0.86)) / 2)
    y += int(H * 0.03)
    for i, item in enumerate(items):
        draw_outlined(img, (x, y), f"{i + 1}.", lfont)
        if i + 1 in shown:
            draw_outlined(img, (x + int(lfont.getlength(f"{len(items)}.  ")), y), item, lfont)
        y += int(lfont.size * 1.55)
    return img


def run(cmd):
    subprocess.run(cmd, check=True)


def main(spec_path, out_path):
    spec = json.load(open(spec_path))
    W, H = spec.get("size", [1080, 1920])
    fps = spec.get("fps", 30)
    start, end = spec.get("start", 0.0), spec["end"]
    dur = end - start
    every = spec.get("cut_every", 2)

    b = [t for t in beats(spec["audio"], start, end) if 0.3 < t < dur - 0.3]
    cuts = [0.0] + b[every - 1::every]
    cuts = [c for i, c in enumerate(cuts) if i == 0 or c - cuts[i - 1] > 0.6] + [dur]
    frames = [round(c * fps) for c in cuts]

    tmp = tempfile.mkdtemp(prefix="listvid_")
    clips = spec["clips"]
    seg_paths = []
    for i in range(len(frames) - 1):
        c = clips[i % len(clips)]
        n = frames[i + 1] - frames[i]
        p = os.path.join(tmp, f"seg{i:03d}.mp4")
        src_dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                        "-of", "csv=p=0", c["src"]], capture_output=True, text=True).stdout)
        t_in = max(0.0, min(c.get("in", 0), src_dur - n / fps - 0.1))
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
              f"setsar=1,fps={fps},{c.get('vf', 'null')}")
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t_in:.3f}", "-i", c["src"],
             "-vf", vf, "-frames:v", str(n), "-an", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "16", "-pix_fmt", "yuv420p", p])
        got = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                  "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", p],
                                 capture_output=True, text=True).stdout.strip() or 0)
        if got < n:  # clip ran out: hold its last frame
            run(["ffmpeg", "-v", "error", "-y", "-i", p, "-vf", f"tpad=stop_mode=clone:stop={n - got}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", p + ".mp4"])
            os.replace(p + ".mp4", p)
        seg_paths.append(p)
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in seg_paths))
    montage = os.path.join(tmp, "montage.mp4")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", montage])

    # Item reveal times: one per cut, starting at the first cut; squeeze in if there are few cuts.
    order = spec.get("reveal", list(range(1, len(spec["items"]) + 1)))
    slots = cuts[1:-1]
    if len(slots) < len(order):
        slots = list(np.linspace(dur * 0.12, dur * 0.8, len(order)))
    else:
        step = len(slots) / len(order)
        slots = [slots[int(k * step)] for k in range(len(order))]

    states = [(0.0, set())]
    shown = set()
    for t, idx in zip(slots, order):
        shown = shown | {idx}
        states.append((t, set(shown)))
    inputs, chain, last = [], [], "[0:v]"
    for k, (t0, s) in enumerate(states):
        t1 = states[k + 1][0] if k + 1 < len(states) else dur + 1
        png = os.path.join(tmp, f"ov{k}.png")
        overlay(spec, s, W, H).save(png)
        inputs += ["-i", png]
        out = f"[v{k}]"
        chain.append(f"{last}[{k + 2}:v]overlay=0:0:enable='between(t,{t0:.3f},{t1 - 1e-3:.3f})'{out}")
        last = out

    run(["ffmpeg", "-v", "error", "-y", "-i", montage,
         "-ss", str(start), "-t", str(dur), "-i", spec["audio"], *inputs,
         "-filter_complex", ";".join(chain) +
         f";[1:a]loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=out:st={max(dur - 0.3, 0)}:d=0.3[a]",
         "-map", last, "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
         "-t", str(dur), "-movflags", "+faststart", out_path])
    print(f"wrote {out_path} ({dur:.2f}s, {len(seg_paths)} cuts, reveals at "
          + ", ".join(f"{t:.2f}" for t in slots) + ")")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
