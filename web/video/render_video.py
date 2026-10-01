"""Render a captioned presentation from the verified receiver replay, offline."""

from __future__ import annotations

import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/demo-video"
DATA = json.loads((OUT / "measured-run.json").read_text())
W, H, FPS, LENGTH = 1920, 1080, 30, 82
BG, INK, MUTED, LINE = "#f3f5f7", "#182637", "#526276", "#dce3e9"
TEAL, GOLD, WHITE = "#087c6c", "#9a661e", "#ffffff"
FONT_REG = Path("/usr/share/fonts/TTF/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf")
if not FONT_REG.exists():
    FONT_REG = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    FONT_BOLD = FONT_REG.with_name("DejaVuSans-Bold.ttf")
FONTS = {(s, b): ImageFont.truetype(str(FONT_BOLD if b else FONT_REG), s)
         for s in [20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 56, 64, 72, 96] for b in [False, True]}

CAPTIONS = [
    (0, 7, "A receiver can miss a signal even when it scans the correct band.",
     "It also has to listen at the right time."),
    (7, 16, "Our receiver covers eight bands, but listens to only one at a time.",
     "Short signal windows make the timing of each visit matter."),
    (16, 22, "The temporal model learns signal timing from past receiver observations.",
     "Unobserved bands stay masked; the scheduler never receives future truth."),
    (22, 27, "A reinforcement-learning policy uses those forecasts to choose a band",
     "and a listening dwell of 1, 10 or 50 ticks."),
    (27, 35, "Both strategies now face identical simulated signal activity.",
     "The scans and counters below replay actual receiver decisions."),
    (35, 43, "Round robin follows a fixed band order and a 50-tick listening dwell.",
     "It covers the spectrum without adapting visits to predicted signal windows."),
    (43, 51, "Temporal forecasting estimates upcoming activity from observed hits and misses.",
     "The RL policy uses that timing information to allocate listening time."),
    (51, 58, "Each new observation feeds the next forecast and scheduling decision.",
     "Colored dots mark true captures; false alarms do not increase the counter."),
    (58, 65, "In this selected run, temporal forecasting + RL captures 37 signals.",
     "Round robin captures 10. That is exactly 3.7 times as many detections."),
    (65, 72, "Interception ratio rises from 7.5% to 27.8% of the same 133 signals.",
     "Both use the same receiver sensitivity, noise and retuning model."),
    (72, 82, "The advantage comes from scheduling across frequency and time.",
     "Observe, forecast, choose, listen — then update from receiver feedback."),
]


def text(draw, xy, value, size=28, color=INK, bold=False, anchor=None):
    draw.text(xy, str(value), font=FONTS[size, bold], fill=color, anchor=anchor)


def card(draw, box, fill=WHITE, outline=LINE, radius=16):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def arrow(draw, start, end, color=TEAL, width=4):
    draw.line([start, end], fill=color, width=width)
    theta = math.atan2(end[1] - start[1], end[0] - start[0])
    points = [end] + [(end[0] - 14 * math.cos(theta + offset),
                      end[1] - 14 * math.sin(theta + offset)) for offset in [-.45, .45]]
    draw.polygon(points, fill=color)


def ease(v):
    v = max(0, min(1, v))
    return v * v * (3 - 2 * v)


def base(title, subtitle, chapter):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 100), fill=WHITE)
    d.line((0, 100, W, 100), fill=LINE, width=2)
    d.rounded_rectangle((64, 30, 104, 70), radius=9, fill=TEAL)
    for x, y in [(74, 57), (84, 45), (94, 37)]:
        d.line((x, 62, x, y), fill=WHITE, width=4)
    text(d, (122, 30), "Spectra Scheduler", 32, bold=True)
    text(d, (1856, 38), "PASSIVE · RECEIVE ONLY", 22, MUTED, anchor="ra")
    text(d, (64, 139), title, 44, bold=True)
    text(d, (64, 207), subtitle, 26, MUTED)
    text(d, (1856, 158), chapter, 22, MUTED, anchor="ra")
    d.line((64, 884, 1856, 884), fill=LINE, width=2)
    text(d, (64, 1030), "Selected simulation example · spatial scan · reporting seed 16007 · not an aggregate benchmark", 20, MUTED)
    return im


def subtitle(im, t):
    d = ImageDraw.Draw(im)
    cap = next(c for c in CAPTIONS if c[0] <= t < c[1])
    for y, line in [(925, cap[2]), (973, cap[3])]:
        text(d, (W / 2, y), line, 30, INK, anchor="mm")
    # Quiet chapter progress line instead of a distracting playback UI.
    d.rectangle((0, 1074, int(W * t / LENGTH), 1079), fill=TEAL)


PREFIX = {}
for name in ["baseline", "adaptive"]:
    PREFIX[name] = np.cumsum([s["captured"] for s in DATA[name]["steps"]])
DECISION_TICKS = np.array([x["tick"] for x in DATA["adaptive"]["decisions"]])


def diagram(d, y, active=-1):
    entries = [
        (64, "01", "Receiver observations", "Past hits, misses and power", "Unobserved bands are masked"),
        (674, "02", "Temporal forecasting", "Estimate future signal windows", "Causal learned timing model"),
        (1284, "03", "RL scheduling", "Choose band and listening dwell", "Trained from trajectory returns"),
    ]
    for i, (x, index, title, a, b) in enumerate(entries):
        card(d, (x, y, x + 572, y + 240), outline=TEAL if active == i else LINE)
        text(d, (x + 28, y + 24), index, 24, TEAL, True)
        text(d, (x + 28, y + 70), title, 32, bold=True)
        text(d, (x + 28, y + 132), a, 26, MUTED)
        text(d, (x + 28, y + 174), b, 24, MUTED)
        if i < 2:
            arrow(d, (x + 580, y + 120), (x + 602, y + 120))
    d.line((1570, y + 253, 1570, y + 286, 350, y + 286), fill=TEAL, width=3)
    arrow(d, (350, y + 286), (350, y + 250), width=3)
    text(d, (960, y + 301), "Listen → receiver feedback → update the next decision", 26, TEAL, anchor="ma")


def cover(t):
    im = base("Temporal forecasting + reinforcement learning",
              "Scheduling a passive receiver across frequency and time", "01 / 06")
    d = ImageDraw.Draw(im)
    card(d, (64, 292, 1160, 832))
    text(d, (112, 338), "Listen at the right time.", 56, bold=True)
    text(d, (112, 426), "Forecast signal windows from receiver history.", 32, MUTED)
    text(d, (112, 480), "Let an RL policy choose where and how long to listen.", 32, MUTED)
    for y, val, label, color in [(604, 10, "Round robin", GOLD), (705, 37, "Temporal forecasting + RL", TEAL)]:
        text(d, (112, y), label, 28, color, True)
        width = int(750 * val / 37 * ease(t / 2))
        if width:
            d.rounded_rectangle((112, y + 43, 112 + width, y + 66), radius=7, fill=color)
        text(d, (1088, y + 22), val, 40, color, True, anchor="rm")
    card(d, (1200, 292, 1856, 832), fill="#e7f3ef", outline="#cce5dd")
    text(d, (1528, 480), "3.7×", 96, TEAL, True, anchor="mm")
    text(d, (1528, 571), "as many true detections", 32, TEAL, anchor="mm")
    text(d, (1528, 652), "37 vs 10 · verified CUDA replay", 26, MUTED, anchor="mm")
    text(d, (1528, 704), "In this selected simulation run", 24, MUTED, anchor="mm")
    return im


def problem(t):
    im = base("One receiver. Eight bands. Limited listening time.",
              "Spatially scanning emitters create brief opportunities to receive a signal", "02 / 06")
    d = ImageDraw.Draw(im)
    tick = min(511, int((t - 7) / 9 * 512))
    current = DATA["baseline"]["steps"][tick]
    x0, y0, width, row = 100, 301, 1150, 62
    d.rounded_rectangle((64, 274, 1290, 832), radius=16, fill=WHITE, outline=LINE, width=2)
    for band in range(8):
        y = y0 + (7 - band) * row
        fill = "#eaf3ef" if band == current["band"] else "#f4f6f8"
        d.rounded_rectangle((x0, y, x0 + width, y + 48), radius=8, fill=fill)
        text(d, (x0 + 18, y + 10), f"Band {band}", 24, MUTED)
        events = [tx for tx in DATA["truth"] if tx["band"] == band and tick - 65 <= tx["time_step"] <= tick + 65]
        for tx in events:
            x = x0 + 200 + (tx["time_step"] - tick + 65) / 130 * 900
            d.rounded_rectangle((x, y + 12, x + 8, y + 36), radius=3, fill="#93a7b7")
        if band == current["band"]:
            d.rectangle((x0 + 640, y + 3, x0 + 657, y + 45), outline=GOLD, width=4)
    card(d, (1330, 274, 1856, 832))
    text(d, (1364, 322), "The scan decision", 32, bold=True)
    text(d, (1364, 398), "Where to listen?", 32, TEAL, True)
    text(d, (1364, 452), "When to visit?", 32, TEAL, True)
    text(d, (1364, 506), "How long to stay?", 32, TEAL, True)
    text(d, (1364, 617), "Gray marks: signal activity", 24, MUTED)
    text(d, (1364, 664), "Gold outline: listening band", 24, GOLD)
    text(d, (1364, 738), "Signal activity shown for", 24, MUTED)
    text(d, (1364, 776), "explanation and evaluation.", 24, MUTED)
    return im


def method(t):
    im = base("Forecast timing. Learn the scan action. Close the loop.",
              "A temporal forecaster supplies timing evidence to a trained RL action policy", "03 / 06")
    d = ImageDraw.Draw(im)
    diagram(d, 290, int((t - 16) / 2) % 3)
    card(d, (64, 684, 1856, 832))
    text(d, (102, 714), "Action space", 26, MUTED)
    text(d, (102, 760), "8 bands × {1, 10, 50} listening ticks", 32, TEAL, True)
    text(d, (1000, 714), "Causal input boundary", 26, MUTED)
    text(d, (1000, 760), "Receiver history only — no future truth", 30, bold=True)
    return im


PLOTS = {}
PLOT_W, PLOT_H = 756, 336
for name, color in [("baseline", GOLD), ("adaptive", TEAL)]:
    plot = Image.new("RGB", (PLOT_W, PLOT_H), "#f4f7f9")
    d = ImageDraw.Draw(plot)
    for b in range(1, 8):
        d.line((0, b * 42, PLOT_W, b * 42), fill=LINE, width=1)
    for tick in range(0, 512, 64):
        x = int(tick / 511 * (PLOT_W - 1))
        d.line((x, 0, x, PLOT_H), fill=LINE, width=1)
    for tx in DATA["truth"]:
        x = int(tx["time_step"] / 511 * (PLOT_W - 1))
        y = (7 - tx["band"]) * 42
        d.rectangle((x, y + 6, x + 2, y + 36), fill="#bac7d0")
    previous = None
    for s in DATA[name]["steps"]:
        x = int(s["tick"] / 511 * (PLOT_W - 1))
        y = (7 - s["band"]) * 42 + 21
        if s["listening"]:
            d.line((x, y, x + 2, y), fill=color, width=4)
            if previous is not None:
                d.line((previous[0], previous[1], x, y), fill=color, width=3)
            previous = (x, y)
        else:
            previous = None
            d.line((x, y - 10, x, y + 10), fill="#667789", width=2)
        if s["captured"]:
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=WHITE, width=1)
    PLOTS[name] = plot


def replay(t):
    im = base("Measured replay: fixed sweep versus forecast-guided RL",
              "Identical signal activity · same receiver model · 512 physical ticks · 133 transmissions", "04 / 06")
    d = ImageDraw.Draw(im)
    tick = min(511, int((t - 27) / 31 * 512))
    for name, x, label, color in [("baseline", 64, "Round robin", GOLD),
                                  ("adaptive", 980, "Temporal forecasting + RL", TEAL)]:
        card(d, (x, 276, x + 876, 714))
        text(d, (x + 26, 300), label, 30, color, True)
        for band in range(8):
            text(d, (x + 35, 362 + (7 - band) * 42 + 21), band, 22, MUTED, anchor="mm")
        left, top = x + 78, 362
        im.paste(PLOTS[name], (left, top))
        d = ImageDraw.Draw(im)
        cursor = left + int(tick / 511 * (PLOT_W - 1))
        # Future observations are concealed; the truth activity stays visible.
        if cursor < left + PLOT_W - 1:
            d.rectangle((cursor + 2, top, left + PLOT_W, top + PLOT_H - 1), fill="#eef2f5")
            for tx in DATA["truth"]:
                if tx["time_step"] > tick:
                    sx = left + int(tx["time_step"] / 511 * (PLOT_W - 1))
                    sy = top + (7 - tx["band"]) * 42
                    d.rectangle((sx, sy + 6, sx + 2, sy + 36), fill="#c3cdd5")
        d.line((cursor, top, cursor, top + PLOT_H), fill=INK, width=2)
        for v in [0, 128, 256, 384, 511]:
            text(d, (left + v / 511 * (PLOT_W - 1), 702), v, 20, MUTED, anchor="mt")
        card(d, (x, 752, x + 876, 860))
        count = int(PREFIX[name][tick])
        s = DATA[name]["steps"][tick]
        text(d, (x + 26, 774), "True detections", 24, MUTED)
        text(d, (x + 242, 768), count, 40, color, True)
        text(d, (x + 395, 779), f"Band {s['band']} · {'listening' if s['listening'] else 'retuning'}", 24, MUTED)
        if name == "adaptive":
            decision = DATA[name]["decisions"][max(0, np.searchsorted(DECISION_TICKS, tick, side="right") - 1)]
            text(d, (x + 395, 819), f"Selected dwell: {decision['dwell']} ticks", 22, TEAL)
        else:
            text(d, (x + 395, 819), "Fixed dwell: 50 listening ticks", 22, GOLD)
    text(d, (1856, 240), f"Tick {tick} / 511", 22, MUTED, anchor="ra")
    return im


def results(t):
    im = base("3.7× as many detections in this measured replay",
              "37 true captures versus 10 · a 270% increase over the round-robin count", "05 / 06")
    d = ImageDraw.Draw(im)
    card(d, (64, 292, 1200, 832))
    for y, name, label, color in [(344, "baseline", "Round robin · 50-tick dwell", GOLD),
                                  (537, "adaptive", "Temporal forecasting + RL", TEAL)]:
        m = DATA[name]["metrics"]
        text(d, (104, y), label, 32, color, True)
        val = m["detected_transmissions"]
        width = int(750 * val / 37 * ease((t - 58) / 1.5))
        d.rounded_rectangle((104, y + 68, 104 + max(width, 1), y + 118), radius=8, fill=color)
        text(d, (1116, y + 78), f"{val} / 133", 40, color, True, anchor="ra")
        text(d, (104, y + 145), f"Interception ratio: {m['interception_ratio'] * 100:.1f}%", 28, MUTED)
    card(d, (1240, 292, 1856, 832), fill="#e7f3ef", outline="#cce5dd")
    text(d, (1548, 463), "3.7×", 96, TEAL, True, anchor="mm")
    text(d, (1548, 559), "37 ÷ 10 true detections", 28, TEAL, anchor="mm")
    for y, line in [(641, "Same simulated signal activity"), (687, "Same sensitivity and retuning model"), (733, "False alarms excluded from counts")]:
        text(d, (1548, y), line, 24, MUTED, anchor="mm")
    return im


def outro(t):
    im = base("A scan strategy that uses both frequency and time",
              "Temporal forecasting + a trajectory-trained reinforcement-learning policy", "06 / 06")
    d = ImageDraw.Draw(im)
    diagram(d, 290)
    card(d, (64, 684, 1856, 832), fill="#e7f3ef", outline="#cce5dd")
    text(d, (102, 710), "MEASURED EXAMPLE", 22, TEAL, True)
    text(d, (102, 756), "37 vs 10 true detections", 36, TEAL, True)
    text(d, (955, 710), "INTERCEPTION RATIO", 22, TEAL, True)
    text(d, (955, 756), "7.5% → 27.8%", 36, TEAL, True)
    text(d, (1700, 742), "3.7×", 64, TEAL, True, anchor="mm")
    return im


def frame(t):
    render = cover if t < 7 else problem if t < 16 else method if t < 27 else replay if t < 58 else results if t < 72 else outro
    im = render(t)
    subtitle(im, t)
    return im


def soundtrack():
    """An original low-volume instrumental pad; no third-party audio assets."""
    rate = 48000
    chords = [(130.81, 164.81, 196.00), (110.00, 130.81, 164.81),
              (87.31, 130.81, 174.61), (98.00, 146.83, 196.00)]
    with wave.open(str(OUT / "soundtrack.wav"), "wb") as stream:
        stream.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        for sec in range(LENGTH):
            time = np.arange(sec * rate, (sec + 1) * rate, dtype=np.float64) / rate
            current = (sec // 8) % 4
            progress = time % 8
            mix = np.clip(progress / 1.5, 0, 1)
            samples = np.zeros(rate)
            for chord, strength in [(chords[(current - 1) % 4], 1 - mix), (chords[current], mix)]:
                for f in chord:
                    samples += strength * (np.sin(2 * np.pi * f * time) + .22 * np.sin(2 * np.pi * f * 2 * time)) / 3
            envelope = np.minimum(np.clip(time / 3, 0, 1), np.clip((LENGTH - time) / 4, 0, 1))
            samples *= .045 * envelope * (.85 + .15 * np.sin(2 * np.pi * .08 * time))
            stereo = np.column_stack((samples, samples * .98))
            stream.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())


def subtitle_time(seconds):
    return f"00:{seconds // 60:02}:{seconds % 60:02},000"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    soundtrack()
    (OUT / "spectra-demo.srt").write_text("\n\n".join(
        f"{i + 1}\n{subtitle_time(a)} --> {subtitle_time(b)}\n{line1}\n{line2}"
        for i, (a, b, line1, line2) in enumerate(CAPTIONS)) + "\n")
    for t in [3, 12, 20, 38, 52, 65, 77]:
        frame(t).save(OUT / f"preview-{t}.png")
    frame(65).save(OUT / "youtube-thumbnail.png")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{W}x{H}",
               "-framerate", str(FPS), "-i", "pipe:0", "-i", str(OUT / "soundtrack.wav"),
               "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "4",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
               "-movflags", "+faststart", "-shortest", str(OUT / "spectra-temporal-rl-demo.mp4")]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for i in range(LENGTH * FPS):
            process.stdin.write(frame(i / FPS).tobytes())
            if i % (FPS * 10) == 0:
                print(f"Rendered {i // FPS} / {LENGTH} seconds", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Video encoding failed")
    except BaseException:
        process.kill()
        process.wait()
        raise
    print(OUT / "spectra-temporal-rl-demo.mp4", flush=True)


if __name__ == "__main__":
    main()
