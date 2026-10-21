#!/usr/bin/env python3
"""Beat-cut montage edits with TikTok-style text (styles A and B).

Usage: python3 edit_video.py spec.json out.mp4

Style A (relatable caption) uses "caption"; style B (numbered list / ranking)
uses "title" + "items". Each segment is one clip, a whole number of beats long;
"show" reveals list items at that segment's first frame, so text lands on the beat.

{
  "audio": "media/sources/Download_5.mp4", "start": 0.0, "end": 30.1,
  "size": [1080, 1920], "fps": 30,
  "transition": "white",            # default for every cut: white | mix | cut
  "layout": "window", "window_ratio": 1.0,   # optional: footage in a centred w/h window, black bars
  "caption": "when ur genuinely chilling and ...",            # style A
  "title": "TOP 5 WAYS TO BE NONCHALANT",                     # style B
  "items": ["use both airpods", "..."],
  "segments": [
    {"src": "media/clips/x.mov", "in": 1.0, "beats": 4},
    {"src": "media/clips/y.mov", "in": 2.5, "beats": 4, "show": [1],
     "speed": 0.25, "transition": "mix", "vf": "transpose=2", "x": 0.4, "fit": "cover"}
  ]
}
"speed" < 1 is slow motion (for 120/240 fps clips). "vf" is an extra ffmpeg
filter applied to the clip before it is cropped to the frame. "x" is the horizontal
focus for cropping wide clips (0 = left edge, 1 = right), "y" the vertical focus for
tall clips (0 = top); "fit": "bars" letterboxes instead.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
BOLD = os.path.join(FONTS, "TikTokSans_700Bold.ttf")
SEMI = os.path.join(FONTS, "TikTokSans_600SemiBold.ttf")
XFADE = {"white": ("fadewhite", 6), "mix": ("fade", 8), "cut": ("fade", 1)}  # (xfade name, frames)


def beat_grid(audio, start, end):
    import librosa

    sr = 22050
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(end - start), "-i", audio,
         "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    y = np.frombuffer(pcm, np.float32)
    _, b = librosa.beat.beat_track(y=y, sr=sr, units="time")
    b = [float(x) for x in b]
    # The tracker sometimes drops out mid-song; fill gaps (and the tail) on the median beat grid.
    step = float(np.median(np.diff(b)))
    filled = [b[0]]
    for t in b[1:] + [len(y) / sr + 8 * step]:
        while t - filled[-1] > 1.5 * step:
            filled.append(filled[-1] + step)
        filled.append(t)
    return filled


def wrap(text, font, max_w):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            trial = f"{cur} {word}".strip()
            if font.getlength(trial) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def draw_text(img, xy, text, font, anchor="la"):
    """White text, thin dark outline and soft drop shadow, like TikTok's caption style."""
    W, H = img.size
    shadow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shadow).text((xy[0] + 2, xy[1] + 3), text, font=font, fill=150, anchor=anchor)
    img.paste(Image.new("RGBA", (W, H), (0, 0, 0, 255)), (0, 0), shadow.filter(ImageFilter.GaussianBlur(5)))
    ImageDraw.Draw(img).text(xy, text, font=font, fill=(255, 255, 255, 255), anchor=anchor,
                             stroke_width=2, stroke_fill=(0, 0, 0, 200))


def window(spec, W, H):
    """Footage window for layout "window": full width, `window_ratio` (w/h) tall, centred; else None."""
    if spec.get("layout") != "window":
        return None
    bh = min(H, int(round(W / spec.get("window_ratio", 1.0) / 2)) * 2)
    return W, bh, (H - bh) // 2 // 2 * 2


def overlay(spec, shown, W, H):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    box = window(spec, W, H)
    if "caption" in spec:
        font = ImageFont.truetype(SEMI, int(W * spec.get("caption_size", 0.052)))
        lines = wrap(spec["caption"], font, W * 0.84)
        lh = int(font.size * 1.22)
        y = int(H * spec.get("caption_y", 0.47 if box else 0.40)) - lh * len(lines) // 2
        for line in lines:
            draw_text(img, (W // 2, y), line, font, anchor="ma")
            y += lh
        return img

    tfont = ImageFont.truetype(BOLD, int(W * 0.062))
    tlines = wrap(spec["title"].upper(), tfont, W * 0.84)
    tlh = int(tfont.size * 1.18)
    # Full-frame: title near the top. Window layout: title sits in the black bar above the footage.
    y = int(H * 0.155) if not box else box[2] - int(H * 0.018) - tlh * len(tlines)
    for line in tlines:
        draw_text(img, (W // 2, y), line, tfont, anchor="ma")
        y += tlh
    if box:
        y = box[2] - int(H * 0.012)
    lfont = ImageFont.truetype(SEMI, int(W * 0.05))
    items = spec["items"]
    num_w = lfont.getlength(f"{len(items)}.  ")
    block_w = num_w + max(lfont.getlength(t) for t in items)
    x = int((W - min(block_w, W * 0.88)) / 2)
    y += int(H * 0.03)
    for i, item in enumerate(items):
        draw_text(img, (x, y), f"{i + 1}.", lfont)
        if i + 1 in shown:
            draw_text(img, (x + int(num_w), y), item, lfont)
        y += int(lfont.size * 1.55)
    return img


def fit(s, W, H, box=None):
    """Fill the frame (or the footage window, black around it).

    "cover" crops around focus "x"/"y" (0..1); "bars" letterboxes in black instead.
    """
    bw, bh, top = box or (W, H, 0)
    if s.get("fit") == "bars":
        f = [f"scale={bw}:{bh}:force_original_aspect_ratio=decrease", f"pad={bw}:{bh}:(ow-iw)/2:(oh-ih)/2:black"]
    else:
        fx, fy = s.get("x", 0.5), s.get("y", 0.5)
        f = [f"scale={bw}:{bh}:force_original_aspect_ratio=increase",
             f"crop={bw}:{bh}:'max(0,min(iw-{bw},iw*{fx}-{bw}/2))':'max(0,min(ih-{bh},ih*{fy}-{bh}/2))'"]
    return f + ([f"pad={W}:{H}:0:{top}:black"] if box else [])


def probe_duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "csv=p=0", path], capture_output=True, text=True).stdout)


def main(spec_path, out_path):
    spec = json.load(open(spec_path))
    W, H = spec.get("size", [1080, 1920])
    fps = spec.get("fps", 30)
    start, end = spec.get("start", 0.0), spec["end"]
    dur = end - start
    total = round(dur * fps)
    grid = beat_grid(spec["audio"], start, end)

    # Cut frames on the beat grid; the first segment starts at 0, the last runs to the end.
    segs, cuts, beat_i = [], [0], 0
    for seg in spec["segments"]:
        beat_i += seg["beats"]
        f = round(grid[min(beat_i, len(grid) - 1)] * fps)
        segs.append(seg)
        if f >= total - fps // 3:
            break
        cuts.append(f)
    cuts.append(total)

    # Transition k sits between segment k-1 and k: d frames, split into post (k-1) / pre (k) handles.
    trans = [None] + [XFADE[s.get("transition", spec.get("transition", "white"))] for s in segs[1:]]
    pre = [0] + [d - (d // 2 or 1) for _, d in trans[1:]]
    post = [(trans[k + 1][1] // 2 or 1) if k + 1 < len(segs) else 0 for k in range(len(segs))]

    tmp = tempfile.mkdtemp(prefix="edit_")
    seg_paths = []
    for k, s in enumerate(segs):
        # +2 spare frames: xfade drops what's left of the outgoing clip, and the spare keeps a
        # 1-frame "cut" from ending the chain early when offsets round badly.
        n = cuts[k + 1] - cuts[k] + pre[k] + post[k] + 2
        speed = s.get("speed", 1.0)
        need = n / fps * speed
        t_in = s.get("in", 0) - pre[k] / fps * speed
        t_in = max(0.0, min(t_in, probe_duration(s["src"]) - need - 0.05))
        vf = ",".join(x for x in [
            s.get("vf"), f"setpts=PTS/{speed}" if speed != 1 else None,
            *fit(s, W, H, window(spec, W, H)), "setsar=1",
            f"fps={fps}", f"tpad=stop_mode=clone:stop={n}", "settb=1/" + str(fps)] if x)
        p = os.path.join(tmp, f"seg{k:03d}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t_in:.3f}", "-i", s["src"], "-vf", vf,
                        "-frames:v", str(n), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                        "-pix_fmt", "yuv420p", p], check=True)
        seg_paths.append(p)

    # Text states: list items appear on the first frame of the segment that "show"s them.
    states, shown = [(0, set())], set()
    for k, s in enumerate(segs):
        if s.get("show"):
            shown = shown | set(s["show"])
            states.append((cuts[k], set(shown)))

    inputs = []
    for p in seg_paths:
        inputs += ["-i", p]
    inputs += ["-ss", str(start), "-t", str(dur), "-i", spec["audio"]]
    a_idx = len(seg_paths)
    chain, last = [], "[0:v]"
    for k in range(1, len(segs)):
        name, d = trans[k]
        out = f"[x{k}]"
        chain.append(f"{last}[{k}:v]xfade=transition={name}:duration={d / fps:.4f}:"
                     f"offset={(cuts[k] - pre[k]) / fps:.4f}{out}")
        last = out
    for j, (f0, s) in enumerate(states):
        f1 = states[j + 1][0] if j + 1 < len(states) else total + fps
        png = os.path.join(tmp, f"ov{j}.png")
        overlay(spec, s, W, H).save(png)
        inputs += ["-i", png]
        out = f"[t{j}]"
        chain.append(f"{last}[{a_idx + 1 + j}:v]overlay=0:0:enable='between(n,{f0},{f1 - 1})'{out}")
        last = out
    chain.append(f"[{a_idx}:a]loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=out:st={max(dur - 0.3, 0)}:d=0.3[a]")

    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(chain),
                    "-map", last, "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-r", str(fps), "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
                    "-frames:v", str(total), "-movflags", "+faststart", out_path], check=True)
    print(f"wrote {out_path} ({dur:.2f}s, {len(segs)} clips, text changes at "
          + ", ".join(f"{f / fps:.2f}" for f, _ in states[1:]) + ")")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
