"""Build showcase/index.html: a gallery linking every generated showcase page.

Usage:
    python scripts/showcase_index.py [showcase_dir]
"""

import base64
import html
import json
import os
import sys

import cv2


def _thumbnail(folder: str) -> str | None:
    """Small inline JPEG of the run's level visualization (or its photo)."""
    candidates = [os.path.join(folder, "level_vis.png")] + [
        os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    for path in candidates:
        image = cv2.imread(path) if os.path.isfile(path) else None
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = 520 / max(width, height)
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    return None


def build_index(showcase_dir: str) -> str:
    cards = []
    for name in sorted(os.listdir(showcase_dir)):
        folder = os.path.join(showcase_dir, name)
        if not os.path.isfile(os.path.join(folder, "showcase.html")):
            continue
        level = {}
        track_path = os.path.join(folder, "track_points.json")
        if os.path.exists(track_path):
            with open(track_path, encoding="utf-8") as f:
                level = json.load(f).get("level", {})
        thumb = _thumbnail(folder)
        behind = ", ".join(o["label"] for o in level.get("occluders", [])) or "—"
        cards.append(
            f"""<a class="card" href="{html.escape(name)}/showcase.html">
  {'<img src="' + thumb + '" alt="">' if thumb else '<div class="noimg"></div>'}
  <div class="body">
    <div class="name">{html.escape(name)}</div>
    <div class="mission">{html.escape(level.get("mission", "Ride the line"))}</div>
    <div class="meta"><span>surface</span>{html.escape(", ".join(level.get("surfaces", [])) or "geometric")}</div>
    <div class="meta"><span>behind</span>{html.escape(behind)}</div>
  </div>
</a>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SkateP Levels</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>
  :root {{ --bg: #06080c; --panel: #0e131c; --line: rgba(255,255,255,0.08); --text: #e9eef6; --muted: #8a94a6; --lime: #c4ff3a; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); font-family: "Space Grotesk", system-ui, sans-serif; padding: 40px 16px 60px; }}
  header {{ max-width: 1200px; margin: 0 auto 28px; }}
  h1 {{ margin: 0; font-size: 30px; letter-spacing: -0.02em; }}
  h1 span {{ color: var(--lime); }}
  header p {{ margin: 6px 0 0; color: var(--muted); font: 13px "JetBrains Mono", monospace; }}
  .grid {{ max-width: 1200px; margin: 0 auto; display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 18px; }}
  .card {{ display: block; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; color: inherit; text-decoration: none; transition: border-color 0.15s, transform 0.15s; }}
  .card:hover {{ border-color: rgba(196,255,58,0.5); transform: translateY(-2px); }}
  .card img, .noimg {{ width: 100%; aspect-ratio: 16 / 10; object-fit: cover; display: block; background: #111722; }}
  .body {{ padding: 14px 16px 16px; }}
  .name {{ font: 600 12px "JetBrains Mono", monospace; color: var(--lime); letter-spacing: 0.04em; }}
  .mission {{ font-size: 16px; font-weight: 500; margin: 6px 0 10px; line-height: 1.3; }}
  .meta {{ font: 12px "JetBrains Mono", monospace; color: var(--text); margin-top: 3px; }}
  .meta span {{ color: var(--muted); display: inline-block; width: 64px; }}
  .empty {{ color: var(--muted); font: 13px "JetBrains Mono", monospace; }}
</style>
</head>
<body>
<header>
  <h1>Skate<span>P</span> levels</h1>
  <p>{len(cards)} photo{'' if len(cards) == 1 else 's'} → playable 2.5D levels · open one to replay its pipeline in 3D</p>
</header>
<main class="grid">
{chr(10).join(cards) if cards else '<p class="empty">No showcases yet. Drop photos into photos/ and run ./run_photos.sh</p>'}
</main>
</body>
</html>
"""
    out = os.path.join(showcase_dir, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"  ✔ Gallery saved to {out} ({len(cards)} level(s))")
    return out


if __name__ == "__main__":
    build_index(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "showcase"))
