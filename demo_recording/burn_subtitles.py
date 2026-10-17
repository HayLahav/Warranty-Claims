"""Draw subtitles.srt onto demo.mp4 -> demo_subtitled.mp4 (same style as the original recording).

Run from the project folder after editing subtitles.srt:
    .venv\\Scripts\\python demo_recording\\burn_subtitles.py
Then rebuild the teleprompter:
    .venv\\Scripts\\python demo_recording\\make_teleprompter.py
If a new subtitle ends after the video does, the last frame is held so the video lasts long enough.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FONT = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 40)


def read_srt(path: Path) -> list[tuple[float, float, str]]:
    def secs(s):
        h, m, rest = s.split(":")
        sec, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000
    cues = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = block.strip().splitlines()
        a, b = [x.strip() for x in lines[1].split("-->")]
        cues.append((secs(a), secs(b), " ".join(l.strip() for l in lines[2:])))
    return cues


def video_info(ff: str, path: Path) -> tuple[int, int, float]:
    err = subprocess.run([ff, "-i", str(path)], capture_output=True, text=True).stderr
    w, h = map(int, re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", err).groups())
    hh, mm, ss = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err).groups()
    return w, h, int(hh) * 3600 + int(mm) * 60 + float(ss)


def wrap_px(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    """Wrap by on-screen width, so lines use the full width of the box."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}" if cur else word
        if cur and font.getlength(trial) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur] if cur else lines


LRM = "‎"          # invisible left-to-right mark
HEBREW = re.compile("[֐-׿]")


def visual(line: str) -> str:
    """Reorder one line of a Hebrew subtitle for drawing (PIL here has no right-to-left layout engine).

    The whole line is laid out right-to-left, so a line holding only "0.99." still ends on the left.
    English in brackets, e.g. "(IoU)", is wrapped in left-to-right marks so its brackets stay with it.
    """
    line = re.sub(r"\(([A-Za-z0-9 .\-]+)\)", LRM + r"(\1)" + LRM, line)
    return get_display(line, base_dir="R").replace(LRM, "")


def caption_png(text: str, w: int, h: int, out: Path) -> None:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines = wrap_px(text, FONT, w - 160)
    if HEBREW.search(text):                     # decided per subtitle, not per wrapped line
        lines = [visual(line) for line in lines]
    line_h = FONT.getbbox("Ag")[3] + 12
    text_w = max(FONT.getlength(line) for line in lines)
    box_w, box_h = text_w + 64, line_h * len(lines) + 30
    x0, y0 = (w - box_w) / 2, h - box_h - 54
    d.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=14, fill=(18, 26, 36, 205))
    for i, line in enumerate(lines):
        d.text(((w - FONT.getlength(line)) / 2, y0 + 15 + i * line_h), line, font=FONT, fill=(255, 255, 255, 255))
    img.save(out)


def main() -> None:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", default="subtitles.srt")
    ap.add_argument("--out", default="demo_subtitled.mp4")
    args = ap.parse_args()
    video, out = HERE / "demo.mp4", HERE / args.out
    cues = read_srt(HERE / args.srt)
    w, h, dur = video_info(ff, video)
    hold = max(0.0, cues[-1][1] + 1.0 - dur)
    with tempfile.TemporaryDirectory() as tmp:
        inputs, chain, last = ["-i", str(video)], [], "0:v"
        if hold:
            chain.append(f"[0:v]tpad=stop_mode=clone:stop_duration={hold:.3f}[base]")
            last = "base"
        for n, (a, b, text) in enumerate(cues, 1):
            png = Path(tmp) / f"{n:03d}.png"
            caption_png(text, w, h, png)
            inputs += ["-i", str(png)]
            chain.append(f"[{last}][{n}:v]overlay=0:0:enable='between(t,{a:.3f},{b:.3f})'[v{n}]")
            last = f"v{n}"
        script = Path(tmp) / "filter.txt"
        script.write_text(";\n".join(chain), encoding="utf-8")
        subprocess.run([ff, "-y", "-loglevel", "error", *inputs, "-filter_complex_script", str(script),
                        "-map", f"[{last}]", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    print(f"wrote {out} ({len(cues)} subtitles{f', last frame held {hold:.1f} s longer' if hold else ''})")


if __name__ == "__main__":
    main()
