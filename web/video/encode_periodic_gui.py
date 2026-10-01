"""Convert the actual GUI recording to an upload-ready H.264/AAC MP4."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/demo-video"


def stamp(t):
    milliseconds = round(t * 1000)
    sec, ms = divmod(milliseconds, 1000)
    minutes, sec = divmod(sec, 60)
    hour, minutes = divmod(minutes, 60)
    return f"{hour:02}:{minutes:02}:{sec:02},{ms:03}"


def main():
    record = json.loads((OUT / "periodic-recording.json").read_text())
    # Discard the recorder's first second while the initial frame settles.
    lead = 1.0
    duration = record["duration_seconds"] - lead
    captions = record["captions"]
    subs = []
    for i, cap in enumerate(captions):
        end = captions[i + 1]["time"] - lead if i + 1 < len(captions) else duration
        subs.append(f"{i + 1}\n{stamp(max(0,cap['time'] - lead))} --> {stamp(end)}\n" + "\n".join(cap["lines"]))
    (OUT / "spectra-periodic-demo.srt").write_text("\n\n".join(subs) + "\n")
    target = OUT / "spectra-periodic-demo.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(record["trim_start_seconds"] + lead), "-i", str(OUT / "actual-periodic-gui-recording.webm"),
        "-i", str(OUT / "soundtrack.wav"), "-t", str(duration),
        "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-threads", "4", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-af", f"afade=t=out:st={max(0, duration - 4)}:d=4",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(target),
    ], check=True)
    for t, name in [(0, "periodic-start.png"), (20, "periodic-playback.png"), (duration - 2, "periodic-final.png")]:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(t),
                        "-i", str(target), "-frames:v", "1", str(OUT / name)], check=True)
    print(target)


if __name__ == "__main__":
    main()
