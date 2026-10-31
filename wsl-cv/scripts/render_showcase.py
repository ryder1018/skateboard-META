"""Render showcase.html to an MP4, frame by frame, in a headless browser.

Every frame is rendered deterministically through the page's renderAt(t),
so the video is smooth regardless of how fast the machine renders WebGL.
The page's synthesized soundtrack (the same one it plays live) is rendered
offline in the browser and muxed in as AAC; --no-audio leaves it out.

Usage:
    python scripts/render_showcase.py <showcase.html> [--out showcase.mp4] [--fps 30] [--size 1920x1080] [--style comic] [--no-audio]

--style picks the Muse Image art style the level settles on after the restyle
step (default: the first style in styles.json; 'original' keeps the photo).

Requires: pip install playwright && playwright install chromium; ffmpeg on PATH.
"""

import argparse
import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def render(
    html_path: str,
    out_path: str,
    fps: int = 30,
    size: str = "1920x1080",
    start: float = 0.0,
    end: float | None = None,
    style: str | None = None,
    audio: bool = True,
) -> str:
    from playwright.sync_api import sync_playwright

    width, height = (int(v) for v in size.lower().split("x"))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH.")

    url = Path(html_path).resolve().as_uri() + "?capture=1" + (f"&style={style}" if style else "")
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(url)
        page.wait_for_function("window.__showcaseReady === true", timeout=120_000)
        if errors:
            raise RuntimeError("Showcase page errors: " + "; ".join(errors))

        duration = page.evaluate("window.__showcase.duration")
        end = duration if end is None else min(end, duration)
        frames = int(round((end - start) * fps))
        print(f"Rendering {frames} frames ({end - start:.1f}s at {fps} fps, {width}x{height})...")

        audio_args: list[str] = []
        wav_path = None
        if audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav:
                wav.write(base64.b64decode(page.evaluate("window.__showcase.audioWav()")))
                wav_path = wav.name
            audio_args = ["-ss", f"{start:.3f}", "-t", f"{frames / fps:.3f}", "-i", wav_path,
                          "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k"]
            print("  soundtrack rendered")

        encoder = subprocess.Popen(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "image2pipe", "-framerate", str(fps), "-c:v", "png", "-i", "-",
                *audio_args,
                "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", out_path,
            ],
            stdin=subprocess.PIPE,
        )
        started = time.perf_counter()
        try:
            for i in range(frames):
                page.evaluate(f"window.__showcase.renderAt({start + i / fps})")
                encoder.stdin.write(page.screenshot(type="png"))
                if (i + 1) % fps == 0 or i + 1 == frames:
                    elapsed = time.perf_counter() - started
                    print(f"  {i + 1}/{frames} frames, {elapsed:.0f}s elapsed", flush=True)
        finally:
            encoder.stdin.close()
            encoder.wait()
            browser.close()
            if wav_path:
                os.unlink(wav_path)
        if errors:
            raise RuntimeError("Showcase page errors: " + "; ".join(errors))
        if encoder.returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {encoder.returncode}.")

    print(f"  ✔ Video saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render the pipeline showcase page to MP4.")
    parser.add_argument("html_path")
    parser.add_argument("--out", default=None, help="Output MP4 (default: next to the HTML)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--style", default=None, help="Art style to settle on (a styles.json name or 'original')")
    parser.add_argument("--no-audio", action="store_true", help="Leave the synthesized soundtrack out")
    args = parser.parse_args()
    out = args.out or os.path.splitext(args.html_path)[0] + ".mp4"
    try:
        render(args.html_path, out, args.fps, args.size, args.start, args.end, args.style, not args.no_audio)
    except Exception as exc:
        print(f"Render failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
