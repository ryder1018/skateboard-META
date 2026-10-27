"""Build the 3D pipeline showcase (a single self-contained HTML file) for one run.

The page replays the whole photo -> level pipeline in a three.js scene:
photo -> DA3 depth relief -> Muse Spark concepts -> SAM 3.1 objects ->
depth-sorted 2.5D layers -> Muse Spark surface choice + track -> ride, passing
behind the objects that stand in front of the track.

Usage:
    python scripts/showcase.py <image> <inference_dir> <send2unity_dir> [--out showcase.html]
"""

import argparse
import base64
import json
import os
from typing import Any

import cv2
import numpy as np

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "showcase_template.html")
DATA_PLACEHOLDER = "__SHOWCASE_DATA__"

# Per-object overlay colors (BGR). Skateable objects get bright, warm-to-green
# hues; scenery gets cool, muted ones so the rideable things read first.
SKATEABLE_COLORS = [(61, 255, 198), (31, 138, 255), (255, 214, 54), (80, 120, 255), (120, 255, 120), (210, 120, 255)]
SCENERY_COLORS = [(170, 120, 90), (150, 110, 140), (120, 130, 150), (160, 150, 110), (130, 100, 120), (110, 140, 130)]


def _data_url(image: np.ndarray, ext: str, params: list[int] | None = None) -> str:
    ok, encoded = cv2.imencode(ext, image, params or [])
    if not ok:
        raise RuntimeError(f"Failed to encode {ext} image.")
    mime = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[ext]
    return f"data:{mime};base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _jpeg(image: np.ndarray) -> str:
    return _data_url(image, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90])


def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _depth_assets(depth16: np.ndarray, size: tuple[int, int]) -> tuple[str, str]:
    """Colorized depth for display and a smoothed near=white displacement map."""
    depth = cv2.resize(depth16, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
    near = 1.0 - np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    near_u8 = (near * 255).astype(np.uint8)

    color = cv2.applyColorMap(near_u8, cv2.COLORMAP_TURBO)

    disp_w = 384
    disp_h = max(1, round(size[1] * disp_w / size[0]))
    disp = cv2.resize(near_u8, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    disp = cv2.GaussianBlur(disp, (0, 0), 2.0)
    return _jpeg(color), _data_url(disp, ".png")


def _sam_assets(
    image_bgr: np.ndarray, labels: np.ndarray, objects: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Darkened photo with colored masks and outlines, plus a label anchor per object."""
    height, width = labels.shape
    overlay = (image_bgr * 0.42).astype(np.float32)
    outlines = np.zeros((height, width), dtype=np.uint8)
    anchors = []
    skate_i = scenery_i = 0

    for obj in objects:
        if obj.get("label_id") is None:
            continue
        mask = labels == obj["label_id"]
        if not mask.any():
            continue
        if obj["skateable"]:
            color = SKATEABLE_COLORS[skate_i % len(SKATEABLE_COLORS)]
            skate_i += 1
            alpha = 0.55
        else:
            color = SCENERY_COLORS[scenery_i % len(SCENERY_COLORS)]
            scenery_i += 1
            alpha = 0.38
        overlay[mask] = overlay[mask] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha

        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outlines, contours, -1, 255, max(1, width // 500))

        # Anchor the 3D label at the point deepest inside the mask, so it
        # always sits on the object even for thin or concave shapes.
        distance = cv2.distanceTransform(np.pad(mask_u8, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        anchors.append(
            {
                "id": obj["id"],
                "label": obj["label"],
                "skateable": obj["skateable"],
                "score": obj.get("score"),
                "at": [round(x / (width - 1), 4), round(y / (height - 1), 4)],
                "color": "#%02x%02x%02x" % (color[2], color[1], color[0]),
            }
        )

    overlay[outlines > 0] = (240, 240, 240)
    return _jpeg(overlay.astype(np.uint8)), anchors


def _layer_asset(path: str) -> str | None:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        return _data_url(image, ".webp", [cv2.IMWRITE_WEBP_QUALITY, 92])
    return _jpeg(image)


def _mask_overlay(mask: np.ndarray, image_bgr: np.ndarray | None, tint: tuple[int, int, int] | None) -> str | None:
    """RGBA cut-out of a mask: the photo itself (image_bgr) or a flat tint."""
    if not mask.any():
        return None
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = image_bgr if image_bgr is not None else np.array(tint, dtype=np.uint8)
    alpha = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (0, 0), 1.0)
    rgba[..., 3] = alpha if image_bgr is not None else (alpha * 0.5).astype(np.uint8)
    return _data_url(rgba, ".webp", [cv2.IMWRITE_WEBP_QUALITY, 92])


def build_showcase(image_path: str, inference_dir: str, send2unity_dir: str, out_html: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    height, width = image_bgr.shape[:2]

    depth16 = cv2.imread(os.path.join(inference_dir, f"{base}_depth_16bit.png"), cv2.IMREAD_UNCHANGED)
    labels = np.load(os.path.join(inference_dir, f"{base}_depth_mask.npy"))
    if labels.shape != (height, width):
        labels = cv2.resize(labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
    scene = _load_json(os.path.join(inference_dir, f"{base}_sam31_objects.json")) or {}
    objects = scene.get("objects", [])
    track = _load_json(os.path.join(send2unity_dir, "track_points.json")) or {}
    attempts = _load_json(os.path.join(send2unity_dir, "level_design_log.json")) or []

    depth_color, depth_disp = _depth_assets(depth16, (width, height))
    sam_overlay, anchors = _sam_assets(image_bgr, labels, objects)

    meta = track.get("level") or {}
    level = {
        "generator": meta.get("generator", "rule-based"),
        "mission": meta.get("mission", "Ride the line"),
        "design_notes": meta.get("design_notes", ""),
        "surfaces": meta.get("surfaces", []),
        "occluders": meta.get("occluders", []),
        "path": track.get("points", []),
    }

    # The chosen riding surface (track region), glowing on the track layer.
    track_mask_path = os.path.join(send2unity_dir, "layer_00_mask.npy")
    track_mask = np.load(track_mask_path) > 127 if os.path.exists(track_mask_path) else np.zeros((height, width), bool)
    if track_mask.shape != (height, width):
        track_mask = cv2.resize(track_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0

    data = {
        "title": base,
        "aspect": width / height,
        "images": {
            "photo": _jpeg(image_bgr),
            "depth": depth_color,
            "disp": depth_disp,
            "sam": sam_overlay,
            "foreground": _layer_asset(os.path.join(send2unity_dir, "merged_layer_01.png")),
            "gameplay": _layer_asset(os.path.join(send2unity_dir, "merged_layer_02.png")),
            "background": _layer_asset(os.path.join(send2unity_dir, "merged_layer_03.png")),
            "surface": _mask_overlay(track_mask, None, (58, 255, 196)),
        },
        "concepts": scene.get("concepts", {"skateable": [], "scenery": [], "source": "unknown"}),
        "timings": scene.get("timings", {}),
        "objects": anchors,
        "level": level,
        "attempts": [
            {
                "attempt": a["attempt"],
                "errors": a["errors"],
                "surfaces": a["plan"]["surface_ids"],
                "seconds": a.get("seconds"),
            }
            for a in attempts
        ],
    }

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()
    # Keep "</script>" inside JSON strings from closing the data block.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace(DATA_PLACEHOLDER, payload)

    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✔ Showcase saved to {out_html} ({len(html) / 1e6:.1f} MB)")
    return out_html


def main():
    parser = argparse.ArgumentParser(description="Build the 3D pipeline showcase for one run.")
    parser.add_argument("image_path", help="The (resized) input image the pipeline ran on")
    parser.add_argument("inference_dir", help="Step 1 output directory")
    parser.add_argument("send2unity_dir", help="Step 2/3 output directory")
    parser.add_argument("--out", default=None, help="Output HTML path (default: <send2unity_dir>/showcase.html)")
    args = parser.parse_args()
    out = args.out or os.path.join(args.send2unity_dir, "showcase.html")
    build_showcase(args.image_path, args.inference_dir, args.send2unity_dir, out)


if __name__ == "__main__":
    main()
