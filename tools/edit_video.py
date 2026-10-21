#!/usr/bin/env python3
"""Beat-cut edits with TikTok "Classic"-style text (relatable captions, lists, rankings).

Usage: python3 edit_video.py spec.json out.mp4

A relatable video uses "caption" (one line of text, centred on the frame); a list or
ranking uses "title" + "items". Each segment is one clip, a whole number of beats long;
"show" reveals list items at that segment's first frame, so text lands on the beat.

{
  "audio": "media/sources/Download_5.mp4", "start": 0.0, "end": 30.1,
  "size": [1080, 1920], "fps": 30,
  "transition": "mix",              # default for every cut: white | mix | cut
  "caption": "when ur genuinely chilling and ...",
  "layout": "window", "window_ratio": 1.0,     # optional "cropped" look: footage cropped into a
                                               # centred w/h window with black bars, text kept central
  "title": "TOP 5 WAYS TO BE NONCHALANT", "items": ["use both airpods", "..."],
  "segments": [
    {"src": "media/clips/x.mov", "in": 1.0, "beats": 4},
    {"src": "media/clips/y.mov", "in": 2.5, "beats": 4, "show": [1], "transition": "white"},
    {"src": "media/clips/z.mov", "in": 0, "beats": 4, "caption": "me:"}
  ]
}
A segment "caption" (skits: "teacher: ..." then "me:") replaces the caption from that
segment on, until another segment sets one.
Clips are never zoomed: one that is already about 9:16 fills the frame, anything wider
(landscape, square) is letterboxed in black at full width. "fit": "cover" | "bars" forces
one or the other, and "x"/"y" (0..1) move the crop of a "cover" clip. "speed" < 1 is slow
motion (for 120/240 fps clips); "vf" is an extra ffmpeg filter run first (e.g. "transpose=2").
iPhone HDR (HLG/PQ) clips are tone-mapped to SDR so their colours match the rest.
"white" is the slow flash from the references: the frame ramps linearly to white,
cuts on the white peak, and ramps back ("flash" seconds per side, default 0.8).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
# Montserrat is the closest free match to TikTok's "Classic" text style (Proxima Nova).
BOLD = os.path.join(FONTS, "Montserrat_700Bold.ttf")
SEMI = os.path.join(FONTS, "Montserrat_600SemiBold.ttf")
TRACKING = -0.02  # letter spacing, in em; the app sets Classic text tight
MIX_FRAMES = 8
TONEMAP = ("zscale=tin={t}:min=bt2020nc:pin=bt2020:rin=tv:t=linear:npl=203,format=gbrpf32le,"
           "zscale=p=bt709,tonemap=tonemap=mobius:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


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
    if len(b) < 2:  # too short to find a tempo (fine for single-clip videos): assume 120 bpm
        b = [0.0, 0.5]
    # The tracker sometimes drops out mid-song; fill gaps (and the tail) on the median beat grid.
    step = float(np.median(np.diff(b)))
    filled = [b[0]]
    for t in b[1:] + [len(y) / sr + 8 * step]:
        while t - filled[-1] > 1.5 * step:
            filled.append(filled[-1] + step)
        filled.append(t)
    return filled


def text_width(text, font):
    return font.getlength(text) + TRACKING * font.size * max(len(text) - 1, 0)


def wrap(text, font, max_w):
    """Greedy wrap, like the TikTok text tool (a short last line is normal there)."""
    lines = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            trial = f"{cur} {word}".strip()
            if text_width(trial, font) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def draw_text(img, xy, text, font, center=False):
    """White text, thin dark outline and soft shadow, with tight tracking (TikTok Classic look)."""
    W, H = img.size
    x0 = xy[0] - text_width(text, font) / 2 if center else xy[0]
    track = TRACKING * font.size
    shadow = Image.new("L", (W, H), 0)
    sd, d = ImageDraw.Draw(shadow), ImageDraw.Draw(img)
    for i, ch in enumerate(text):
        x = x0 + font.getlength(text[:i]) + track * i
        sd.text((x + 2, xy[1] + 3), ch, font=font, fill=140)
    img.paste(Image.new("RGBA", (W, H), (0, 0, 0, 255)), (0, 0), shadow.filter(ImageFilter.GaussianBlur(5)))
    for i, ch in enumerate(text):
        x = x0 + font.getlength(text[:i]) + track * i
        d.text((x, xy[1]), ch, font=font, fill=(255, 255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0, 230))


def overlay(spec, shown, caption, W, H):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if "title" not in spec:
        if not caption:
            return img
        font = ImageFont.truetype(SEMI, int(W * spec.get("caption_size", 0.058)))
        lines = wrap(caption, font, W * 0.86)
        lh = int(font.size * 1.2)
        # Lower third like the account's posts; in the window layout, the lower part of the footage.
        box = window(spec, W, H)
        cy = spec.get("caption_y", (box[2] + box[1] * 0.82) / H if box else 0.75)
        y = int(H * cy) - lh * len(lines) // 2
        for line in lines:
            draw_text(img, (W // 2, y), line, font, center=True)
            y += lh
        return img

    tfont = ImageFont.truetype(BOLD, int(W * 0.062))
    y = int(H * 0.155)
    for line in wrap(spec["title"].upper(), tfont, W * 0.86):
        draw_text(img, (W // 2, y), line, tfont, center=True)
        y += int(tfont.size * 1.18)
    lfont = ImageFont.truetype(SEMI, int(W * 0.05))
    items = spec["items"]
    num_w = text_width(f"{len(items)}.  ", lfont)
    block_w = num_w + max(text_width(t, lfont) for t in items)
    x = int((W - min(block_w, W * 0.9)) / 2)
    y += int(H * 0.03)
    for i, item in enumerate(items):
        draw_text(img, (x, y), f"{i + 1}.", lfont)
        if i + 1 in shown:
            draw_text(img, (x + int(num_w), y), item, lfont)
        y += int(lfont.size * 1.55)
    return img


def probe(path):
    out = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,color_transfer:stream_side_data=rotation:format=duration", "-of", "json", path],
        capture_output=True, text=True).stdout)
    st = out["streams"][0]
    rot = next((abs(int(sd.get("rotation", 0))) for sd in st.get("side_data_list", []) if "rotation" in sd), 0)
    w, h = (st["height"], st["width"]) if rot in (90, 270) else (st["width"], st["height"])
    return {"w": w, "h": h, "transfer": st.get("color_transfer"), "duration": float(out["format"]["duration"])}


def window(spec, W, H):
    """Footage box for the "window" (cropped) layout: full width, window_ratio (w/h), centred."""
    if spec.get("layout") != "window":
        return None
    bh = min(H, int(round(W / spec.get("window_ratio", 1.0) / 2)) * 2)
    return W, bh, (H - bh) // 4 * 2


def fit(s, info, W, H, box=None):
    """Full frame: near-9:16 clips cover, anything wider is letterboxed (never zoomed).
    Window layout: every clip is cropped to fill the window, black bars around it."""
    w, h = info["w"], info["h"]
    if "transpose" in (s.get("vf") or ""):
        w, h = h, w
    bw, bh, top = box or (W, H, 0)
    mode = s.get("fit") or ("cover" if box or abs((w / h) / (W / H) - 1) < 0.2 else "bars")
    if mode == "bars":
        f = [f"scale={bw}:{bh}:force_original_aspect_ratio=decrease", f"pad={bw}:{bh}:(ow-iw)/2:(oh-ih)/2:black"]
    else:
        fx, fy = s.get("x", 0.5), s.get("y", 0.5)
        f = [f"scale={bw}:{bh}:force_original_aspect_ratio=increase",
             f"crop={bw}:{bh}:'max(0,min(iw-{bw},iw*{fx}-{bw}/2))':'max(0,min(ih-{bh},ih*{fy}-{bh}/2))'"]
    return f + ([f"pad={W}:{H}:0:{top}:black"] if box else [])


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
    kinds = [None] + [s.get("transition", spec.get("transition", "mix")) for s in segs[1:]]

    # A mix overlaps the two clips (xfade); a cut or white flash is a 1-frame xfade, with the
    # white ramps baked into the clips on either side of it.
    d = [0] + [MIX_FRAMES if k == "mix" else 1 for k in kinds[1:]]
    pre = [0] + [x - (x // 2 or 1) for x in d[1:]]
    post = [(d[k + 1] // 2 or 1) if k + 1 < len(segs) else 0 for k in range(len(segs))]
    flash = spec.get("flash", 0.8)

    tmp = tempfile.mkdtemp(prefix="edit_")
    seg_paths = []
    for k, s in enumerate(segs):
        visible = cuts[k + 1] - cuts[k]
        # +2 spare frames: xfade drops what's left of the outgoing clip, and the spare keeps a
        # 1-frame cut from ending the chain early when offsets round badly.
        n = visible + pre[k] + post[k] + 2
        info = probe(s["src"])
        speed = s.get("speed", 1.0)
        t_in = s.get("in", 0) - pre[k] / fps * speed
        t_in = max(0.0, min(t_in, info["duration"] - n / fps * speed - 0.05))
        ramp = min(flash, 0.45 * visible / fps)
        fades = []
        if kinds[k] == "white":
            fades.append(f"fade=t=in:st=0:d={ramp:.3f}:color=white")
        if k + 1 < len(segs) and kinds[k + 1] == "white":
            fades.append(f"fade=t=out:st={(visible - ramp * fps) / fps:.3f}:d={ramp:.3f}:color=white")
        hdr = TONEMAP.format(t=info["transfer"]) if info["transfer"] in ("arib-std-b67", "smpte2084") else None
        vf = ",".join(x for x in [
            hdr, s.get("vf"), f"setpts=PTS/{speed}" if speed != 1 else None,
            *fit(s, info, W, H, window(spec, W, H)), "setsar=1", f"fps={fps}", f"tpad=stop_mode=clone:stop={n}",
            "settb=1/" + str(fps), *fades] if x)
        p = os.path.join(tmp, f"seg{k:03d}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t_in:.3f}", "-i", s["src"], "-vf", vf,
                        "-frames:v", str(n), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
                        "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
                        "-color_primaries", "bt709", "-color_trc", "bt709", p], check=True)
        seg_paths.append(p)

    # Text states: list items appear on the first frame of the segment that "show"s them;
    # a segment caption takes over from its first frame.
    caption = spec.get("caption")
    states, shown = [(0, set(), segs[0].get("caption", caption))], set()
    for k, s in enumerate(segs):
        if s.get("show") or (k and "caption" in s):
            shown = shown | set(s.get("show", []))
            caption = s.get("caption", states[-1][2])
            states.append((cuts[k], set(shown), caption))

    inputs = []
    for p in seg_paths:
        inputs += ["-i", p]
    inputs += ["-ss", str(start), "-t", str(dur), "-i", spec["audio"]]
    a_idx = len(seg_paths)
    chain, last = [], "[0:v]"
    for k in range(1, len(segs)):
        out = f"[x{k}]"
        chain.append(f"{last}[{k}:v]xfade=transition=fade:duration={d[k] / fps:.4f}:"
                     f"offset={(cuts[k] - pre[k]) / fps:.4f}{out}")
        last = out
    for j, (f0, s, cap) in enumerate(states):
        f1 = states[j + 1][0] if j + 1 < len(states) else total + fps
        png = os.path.join(tmp, f"ov{j}.png")
        overlay(spec, s, cap, W, H).save(png)
        inputs += ["-i", png]
        out = f"[t{j}]"
        chain.append(f"{last}[{a_idx + 1 + j}:v]overlay=0:0:enable='between(n,{f0},{f1 - 1})'{out}")
        last = out
    chain.append(f"[{a_idx}:a]loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=out:st={max(dur - 0.3, 0)}:d=0.3[a]")

    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(chain),
                    "-map", last, "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries",
                    "bt709", "-color_trc", "bt709", "-r", str(fps), "-c:a", "aac", "-b:a", "192k",
                    "-ar", "44100", "-frames:v", str(total), "-movflags", "+faststart", out_path], check=True)
    print(f"wrote {out_path} ({dur:.2f}s, {len(segs)} clips"
          + (", text changes at " + ", ".join(f"{st[0] / fps:.2f}" for st in states[1:]) if len(states) > 1 else "")
          + ")")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
