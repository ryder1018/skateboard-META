"""Muse Image restyling: repaint the photo as game art, keep the composition.

Muse Image (Meta Model API, Responses endpoint) repaints the photo in a game
art style (comic, cyberpunk, snow, ...) and is told to keep every object where
it is. That is close but not guaranteed, so each result is
  1. registered back onto the photo (SIFT + RANSAC similarity transform), and
  2. scored against the photo: the local pixel offset along the SAM 3.1 layer
     outlines (outline_offset) and an overall edge F1, raw and registered,
so the claim "the SAM 3.1 masks still fit" is measured, not assumed. The
layers are then re-cut from the styled pixels with the exact alphas
cut_img.py produced from the SAM 3.1 + DA3 layering, so every style shares the
same foreground / gameplay / background split and the same track.

Outputs per run (next to the Unity layers):
    styles/<name>/styled_raw.png        Muse Image output, as returned
    styles/<name>/styled.png            registered to the photo, photo size
    styles/<name>/merged_layer_0{1,2,3}.png
    styles/<name>/meta.json             timing + token usage of the call
    styles.json                         manifest (alignment scores, track colors)

Usage:
    python scripts/muse_image.py <image> <send2unity_dir> [--styles comic,snow] [--force]
"""

import argparse
import base64
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import requests

from extract_track import track_color
from muse_spark import image_data_url

MUSE_IMAGE_API_URL = "https://api.meta.ai/v1/responses"
MUSE_IMAGE_MODEL_ID = "muse-image-1.0"

# Shared by every style: what must not change. The SAM 3.1 masks and the
# track are reused as-is, so the frame has to stay put.
KEEP_COMPOSITION = (
    "Keep the composition identical to the input photo: the same camera, framing, "
    "perspective and horizon, and every object, edge, step, rail and building in "
    "exactly the same position, size and outline. Do not add, remove, move or "
    "resize anything, do not crop or zoom, and do not add text, people or borders. "
    "Only the rendering style, materials, colors and lighting change."
)

# The background layer: Muse Image completes the scene hidden behind the front
# layers. Its edit tool ignores masks (input_image_mask is accepted but has no
# effect), so the hidden area is painted a key color and the prompt asks for it
# to be filled; only those pixels are taken back. The hidden objects are gone
# from the input, so they cannot reappear as ghosts.
FILL_KEY_BGR = np.array([255, 0, 255])  # magenta
# Neutral on purpose: naming landscape features made every fill a forest. The
# setting sentence (Muse Spark, from the photo) says what kind of place it is,
# since most of the frame can be magenta. A hole keeps its object's outline, so
# the model tended to paint the object back (a hydrant-shaped hole became a
# hydrant): the prompt names what was removed and asks for background only.
FILL_PROMPT = (
    "The solid magenta areas of this image are missing.{setting}{removed}{style} Fill them in with what lies "
    "behind them: extend the background that surrounds each area in this same place (the same floor, "
    "table top, wall, road, sky or buildings) with matching perspective, lighting, texture and art style. "
    "Do not guess from the shape of an area what object stood there; paint only the surrounding "
    "background continuing through it. Do not add people, animals, vehicles or new objects, and leave "
    "no magenta."
)
FILL_KEEP = (
    "Keep every other pixel exactly as it is, with the same framing and art style; do not crop or zoom, "
    "and do not add text or borders."
)

STYLES: dict[str, dict[str, str]] = {
    "comic": {
        "label": "Comic",
        "color": "#ffd23f",
        "prompt": "Repaint this photo as a hand-drawn comic book panel: bold black ink "
        "outlines, flat cel shading, halftone dots in the shadows, punchy saturated colors.",
    },
    "cyberpunk": {
        "label": "Cyberpunk",
        "color": "#ff3fd8",
        "prompt": "Repaint this photo as a rainy cyberpunk night scene: dark sky, wet "
        "reflective ground, neon magenta and cyan rim light on every edge, glowing signs "
        "and windows, light haze.",
    },
    "snow": {
        "label": "Snow",
        "color": "#9fe3ff",
        "prompt": "Repaint this photo as the same place in deep winter: fresh snow resting "
        "on every horizontal surface, frosted trees, an overcast pale-blue sky and soft "
        "cold light, like a stylized winter video game level.",
    },
    "pixel": {
        "label": "Pixel",
        "color": "#7cff6b",
        "prompt": "Repaint this photo as 16-bit pixel art from a retro console game, as if drawn "
        "at about 320 pixels wide and upscaled with nearest-neighbor: a clearly visible blocky "
        "pixel grid, hard stair-stepped outlines, a limited bright palette, dithered gradients "
        "in the sky and shadows.",
    },
    "watercolor": {
        "label": "Watercolor",
        "color": "#ff9a6b",
        "prompt": "Repaint this photo as a loose watercolor and ink sketch: fine pen lines on "
        "the outlines, soft pigment washes, visible paper texture, warm afternoon light.",
    },
}


def model_id() -> str:
    return os.getenv("MUSE_IMAGE_MODEL_ID", MUSE_IMAGE_MODEL_ID)


def styles_from_env(default: list[str] | None = None) -> list[str]:
    """MUSE_IMAGE_STYLES="comic,snow" picks styles; "none" turns restyling off."""
    raw = os.getenv("MUSE_IMAGE_STYLES")
    if raw is None:
        return list(STYLES) if default is None else default
    return parse_styles(raw)


def parse_styles(raw: str) -> list[str]:
    names = [s.strip().lower() for s in raw.split(",") if s.strip()]
    if names in (["none"], ["off"], []):
        return []
    if names == ["all"]:
        return list(STYLES)
    unknown = [n for n in names if n not in STYLES]
    if unknown:
        raise ValueError(f"Unknown style(s) {unknown}; choose from {list(STYLES)}")
    return names


def call_muse_image(image_bgr: np.ndarray, prompt: str, keep: str = KEEP_COMPOSITION) -> tuple[np.ndarray, dict[str, Any]]:
    """One Muse Image edit turn on the photo. Returns (styled BGR image, usage)."""
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("MODEL_API_KEY is not set; Muse Image runs through the Meta Model API.")

    try:
        timeout = float(os.getenv("MUSE_IMAGE_TIMEOUT", "300"))
    except ValueError:
        timeout = 300.0

    request = dict(
        url=os.getenv("MUSE_IMAGE_API_URL", MUSE_IMAGE_API_URL),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_id(),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"{prompt} {keep}"},
                        {"type": "input_image", "image_url": image_data_url(image_bgr)},
                    ],
                }
            ],
            # The tool only accepts 1024x1024 / 1536x1024 / 1024x1536 / auto;
            # with an input image, auto keeps that image's aspect ratio (4:3
            # in, 4:3 out), which is what the masks need. Search and code
            # tools add latency and nothing to a restyle.
            "tools": [
                {
                    "type": "image_generation",
                    "size": "auto",
                    "output_format": "png",
                    "enable_image_search": False,
                    "enable_web_search": False,
                    "enable_shell": False,
                }
            ],
            "store": False,
        },
        timeout=timeout,
    )
    # Server errors and timeouts are usually transient; 4xx are not.
    retries = int(os.getenv("MUSE_IMAGE_RETRIES", "2"))
    for attempt in range(retries + 1):
        try:
            response = requests.post(**request)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
            continue
        if response.status_code < 500 or attempt == retries:
            break
        time.sleep(2 * (attempt + 1))
    if response.status_code != 200:
        raise RuntimeError(f"Muse Image request failed (HTTP {response.status_code}): {response.text[:500]}")

    body = response.json()
    if body.get("status") != "completed":
        raise RuntimeError(f"Muse Image response not completed: {body.get('status')} {body.get('incomplete_details')}")
    for item in body.get("output", []):
        if item.get("type") == "image_generation_call" and item.get("result"):
            encoded = np.frombuffer(base64.b64decode(item["result"]), dtype=np.uint8)
            styled = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if styled is None:
                raise RuntimeError("Muse Image returned an image that could not be decoded.")
            return styled, body.get("usage") or {}
    raise RuntimeError("Muse Image response contained no image.")


def _edges(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    return cv2.Canny(gray, 50, 150) > 0


def edge_f1(reference: np.ndarray, candidate: np.ndarray, tolerance_px: int = 3) -> float:
    """F1 of the two edge maps, an edge counting as matched within tolerance_px.

    Style-invariant enough to compare a photo with its restyle: colors and
    textures change, the outlines of the objects should not.
    """
    ref, cand = _edges(reference), _edges(candidate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1,) * 2)
    ref_near = cv2.dilate(ref.astype(np.uint8), kernel) > 0
    cand_near = cv2.dilate(cand.astype(np.uint8), kernel) > 0
    precision = (cand & ref_near).sum() / max(int(cand.sum()), 1)
    recall = (ref & cand_near).sum() / max(int(ref.sum()), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-6))


def _gradient(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2.0)
    magnitude = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    return magnitude / (magnitude.mean() + 1e-6)


def outline_offset(
    photo: np.ndarray, styled: np.ndarray, masks: list[np.ndarray], patch: int = 64, max_points: int = 160
) -> dict[str, Any] | None:
    """How far the styled image has drifted along the layer masks' outlines, in pixels.

    Samples points on the outlines the SAM 3.1 layers were cut along and
    phase-correlates a gradient patch of the photo against the styled image
    there. Edge F1 punishes new texture (bricks, rain); this measures only the
    local shift, which is what decides whether the masks still fit. Sanity
    check: aligned restyles give ~1 px medians and ~80% of points within 3 px;
    the same images shifted by 8 px drop to ~5 px and ~35%.
    """
    height, width = photo.shape[:2]
    boundary = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        boundary |= cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    half = patch // 2
    boundary[:half], boundary[-half:], boundary[:, :half], boundary[:, -half:] = 0, 0, 0, 0
    ys, xs = np.nonzero(boundary)
    if len(xs) < 20:
        return None

    grad_photo, grad_styled = _gradient(photo), _gradient(styled)
    window = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    offsets = []
    for i in np.linspace(0, len(xs) - 1, min(max_points, len(xs))).astype(int):
        y, x = ys[i], xs[i]
        (dx, dy), response = cv2.phaseCorrelate(
            grad_photo[y - half:y + half, x - half:x + half], grad_styled[y - half:y + half, x - half:x + half], window
        )
        if response > 0.08:  # flat patches give no reliable peak
            offsets.append(float(np.hypot(dx, dy)))
    if len(offsets) < 10:
        return None
    offsets_np = np.array(offsets)
    return {
        "median_px": round(float(np.median(offsets_np)), 2),
        "within_3px": round(float((offsets_np <= 3).mean()), 3),
        "points": len(offsets),
    }


def register_to_photo(styled: np.ndarray, photo: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Map the styled image onto the photo's pixel grid.

    Resizes to the photo's size, then estimates a similarity transform (scale,
    rotation, shift) from SIFT matches with RANSAC and applies it only when it
    is plausible and improves edge agreement.
    """
    height, width = photo.shape[:2]
    resized = cv2.resize(styled, (width, height), interpolation=cv2.INTER_AREA)
    info: dict[str, Any] = {"edge_f1_raw": round(edge_f1(photo, resized), 4), "registered": False}

    sift = cv2.SIFT_create(4000)
    kp_photo, desc_photo = sift.detectAndCompute(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), None)
    kp_styled, desc_styled = sift.detectAndCompute(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), None)
    good = []
    if desc_photo is not None and desc_styled is not None and len(kp_photo) >= 2 and len(kp_styled) >= 2:
        pairs = cv2.BFMatcher().knnMatch(desc_styled, desc_photo, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    info["matches"] = len(good)

    aligned = resized
    if len(good) >= 12:
        src = np.float32([kp_styled[m.queryIdx].pt for m in good])
        dst = np.float32([kp_photo[m.trainIdx].pt for m in good])
        matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if matrix is not None:
            scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
            rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
            shift = float(np.hypot(matrix[0, 2], matrix[1, 2]))
            info.update(
                inliers=int(inliers.sum()),
                scale=round(scale, 4),
                rotation_deg=round(rotation, 3),
                shift_px=round(shift, 2),
            )
            plausible = 0.85 < scale < 1.18 and abs(rotation) < 5 and shift < 0.12 * width and inliers.sum() >= 12
            if plausible:
                warped = cv2.warpAffine(resized, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                score = edge_f1(photo, warped)
                if score > info["edge_f1_raw"]:
                    aligned = warped
                    info["registered"] = True

    info["edge_f1"] = round(edge_f1(photo, aligned), 4)
    return aligned, info


def fill_enabled() -> bool:
    """MUSE_IMAGE_FILL=0 keeps cv2.inpaint for the background layer."""
    return os.getenv("MUSE_IMAGE_FILL", "1").strip().lower() not in ("0", "false", "no", "off")


def front_holes(send2unity_dir: str) -> np.ndarray:
    """What the background layer must fill: everything the front layers cover (0/255).

    Grown a few pixels past the layers' feathered edges, so no outline of a
    hidden object is left for the fill to continue.
    """
    holes = None
    for name in ("merged_layer_01.png", "merged_layer_02.png"):
        layer = cv2.imread(os.path.join(send2unity_dir, name), cv2.IMREAD_UNCHANGED)
        if layer is None or layer.ndim != 3 or layer.shape[2] != 4:
            raise FileNotFoundError(f"Missing RGBA layer {name} in {send2unity_dir}; run cut_img.py first.")
        covered = (layer[:, :, 3] > 0).astype(np.uint8) * 255
        holes = covered if holes is None else np.maximum(holes, covered)
    return cv2.dilate(holes, np.ones((7, 7), np.uint8))


def fill_prompt(setting: str | None, removed: list[str] | None = None, style: str | None = None) -> str:
    """The fill prompt; `style` (a STYLES name) for a restyled image.

    The setting is written from the photo ("a bright lawn under a blue sky"),
    and with most of the frame magenta the model followed it and painted
    daylight into a cyberpunk night, so a style's look is spelled out and put
    above the setting's lighting.
    """
    look = STYLES[style]["prompt"].removeprefix("Repaint this photo as ") if style else ""
    return FILL_PROMPT.format(
        setting=f" The place: {setting.rstrip('.')}." if setting else "",
        removed=f" These areas held {', '.join(removed)}, which have been removed: do not draw them again." if removed else "",
        style=(
            f" This image is drawn as {look} Paint the filled areas exactly the same way: the place gives "
            "only the layout, while the lighting, weather, season and look follow this art style."
            if style
            else ""
        ),
    )


def front_objects(send2unity_dir: str, objects_path: str | None, classify: bool = False) -> list[str]:
    """Names of the SAM 3.1 objects the front layers hide, for the fill to leave out.

    Skips the riding surface and Muse Spark's `fixed` concepts (ground, walls,
    buildings, ...): those are background to continue, not objects to remove.
    Runs without a `fixed` list get one from Muse Spark when `classify` is set,
    saved back into the objects file; otherwise nothing is named.
    """
    layers_path = os.path.join(send2unity_dir, "layers.json")
    if not objects_path or not os.path.exists(objects_path) or not os.path.exists(layers_path):
        return []
    with open(layers_path, encoding="utf-8") as f:
        layers = json.load(f)
    with open(objects_path, encoding="utf-8") as f:
        scene = json.load(f)
    concepts = scene.setdefault("concepts", {})
    fixed = concepts.get("fixed")
    if fixed is None and classify:
        from sam31_segmentation import classify_fixed

        try:
            fixed = concepts["fixed"] = classify_fixed(concepts.get("skateable", []) + concepts.get("scenery", []))
        except Exception as exc:
            print(f"  ! Muse Spark fixed-concept sort failed ({exc}); naming no removed objects")
            return []
        with open(objects_path, "w", encoding="utf-8") as f:
            json.dump(scene, f, indent=2)
    if fixed is None:
        return []
    front = {rid for layer in layers["layers"][:-1] for rid in layer["region_ids"]} - set(layers["track_region_ids"])
    return sorted({obj["label"] for obj in scene["objects"] if obj.get("label_id") in front and obj["label"] not in fixed})


def load_setting(objects_path: str | None, image: np.ndarray | None = None) -> str | None:
    """The run's setting sentence from *_sam31_objects.json.

    Runs from before the sentence existed get one from Muse Spark (given the
    image), saved back into the file.
    """
    if not objects_path or not os.path.exists(objects_path):
        return None
    with open(objects_path, encoding="utf-8") as f:
        scene = json.load(f)
    concepts = scene.setdefault("concepts", {})
    if concepts.get("setting") or image is None:
        return concepts.get("setting")
    from sam31_segmentation import describe_setting

    try:
        concepts["setting"] = describe_setting(image)
    except Exception as exc:
        print(f"  ! Muse Spark setting failed ({exc}); filling without it")
        return None
    with open(objects_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2)
    return concepts["setting"]


def _ring_shift(image: np.ndarray, filled: np.ndarray, holes: np.ndarray) -> float:
    """Mean Lab color change just outside the holes, summed over channels.

    The fill is told to keep those pixels, so a large shift means Muse Image
    redrew the whole picture in another look (a cyberpunk fill coming back as
    a daylight photo), and its pixels would not match the layer they fill.
    Good fills stay under ~13; drifted ones measured 27-79.
    """
    ring = cv2.dilate(holes, np.ones((41, 41), np.uint8)).astype(bool) & (holes == 0)
    if not ring.any():
        return 0.0
    before = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[ring].astype(np.float32).mean(axis=0)
    after = cv2.cvtColor(filled, cv2.COLOR_BGR2LAB)[ring].astype(np.float32).mean(axis=0)
    return float(np.abs(before - after).sum())


def fill_background(
    image: np.ndarray,
    holes: np.ndarray,
    folder: str,
    force: bool,
    setting: str | None = None,
    removed: list[str] | None = None,
    style: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Background layer from Muse Image: the hidden area painted magenta and filled in.

    The raw result is cached in folder/fill_raw.png, keyed by the image, the
    holes and the prompt. A fill that leaves the key color behind or drifts
    from the image's look is made again, up to MUSE_IMAGE_FILL_ATTEMPTS
    calls; raises if none passes.
    """
    raw_path = os.path.join(folder, "fill_raw.png")
    meta_path = os.path.join(folder, "fill_meta.json")
    prompt = fill_prompt(setting, removed, style)
    key = hashlib.md5(
        np.packbits(holes > 0).tobytes() + cv2.resize(image, (64, 64)).tobytes() + prompt.encode()
    ).hexdigest()
    os.makedirs(folder, exist_ok=True)
    meta = None
    if not force and os.path.exists(raw_path) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("key") == key:
            raw = cv2.imread(raw_path)
            meta["cached"] = True
        else:
            meta = None

    hidden = holes > 0
    max_drift = float(os.getenv("MUSE_IMAGE_FILL_MAX_DRIFT", "20"))
    problems = []
    for _ in range(max(1, int(os.getenv("MUSE_IMAGE_FILL_ATTEMPTS", "2")))):
        if meta is None:
            masked = image.copy()
            masked[hidden] = FILL_KEY_BGR
            start = time.time()
            raw, usage = call_muse_image(masked, prompt, FILL_KEEP)
            meta = {"seconds": round(time.time() - start, 1), "usage": usage, "key": key}
            cv2.imwrite(raw_path, raw)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            meta["cached"] = False
        filled, _ = register_to_photo(raw, image)
        # Neon magenta in a cyberpunk fill is far enough from the pure key color.
        leftover = float((np.abs(filled.astype(np.int32) - FILL_KEY_BGR).sum(axis=2) < 60)[hidden].mean())
        drift = _ring_shift(image, filled, holes)
        if leftover > 0.02:
            problems.append(f"left {leftover:.0%} of the hidden area magenta")
        elif drift > max_drift:
            problems.append(f"changed the look around the holes (color shift {drift:.0f} > {max_drift:.0f})")
        else:
            break
        meta = None
    else:
        raise RuntimeError("Muse Image fill " + "; ".join(problems))

    weight = cv2.GaussianBlur(holes, (0, 0), 2)[..., None] / 255.0
    background = (filled * weight + image * (1 - weight)).astype(np.uint8)
    return background, {
        "source": "muse-image",
        "seconds": meta["seconds"],
        "cached": meta["cached"],
        "attempts": len(problems) + 1,
        "leftover": round(leftover, 4),
        "drift": round(drift, 1),
    }


def _inpaint_background(image: np.ndarray, holes: np.ndarray) -> np.ndarray:
    return cv2.inpaint(image, holes, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def fill_photo_background(
    image_path: str,
    send2unity_dir: str,
    force: bool = False,
    setting: str | None = None,
    removed: list[str] | None = None,
) -> dict[str, Any]:
    """Replace cut_img.py's inpainted merged_layer_03.png with a Muse Image fill.

    Falls back to cv2.inpaint. Returns the entry also written to background.json.
    """
    photo = cv2.imread(image_path)
    if photo is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    holes = front_holes(send2unity_dir)
    try:
        background, info = fill_background(photo, holes, os.path.join(send2unity_dir, "background"), force, setting, removed)
        print(f"  ✔ photo background: Muse Image filled {np.mean(holes > 0):.0%} of the frame ({'cached' if info['cached'] else str(info['seconds']) + 's'})")
    except Exception as exc:
        background, info = _inpaint_background(photo, holes), {"source": "inpaint", "error": str(exc)}
        print(f"  ! photo background: Muse Image fill failed ({exc}); inpainting instead")
    cv2.imwrite(os.path.join(send2unity_dir, "merged_layer_03.png"), cv2.cvtColor(background, cv2.COLOR_BGR2BGRA))
    with open(os.path.join(send2unity_dir, "background.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def export_styled_layers(styled: np.ndarray, send2unity_dir: str, out_dir: str, background: np.ndarray | None = None) -> None:
    """Re-cut the three layers from the styled image with cut_img.py's alphas.

    Foreground / gameplay keep their exact alpha; the background is the
    Muse Image fill when given, else the styled image inpainted under the front layers.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name in ("merged_layer_01.png", "merged_layer_02.png"):
        layer = cv2.imread(os.path.join(send2unity_dir, name), cv2.IMREAD_UNCHANGED)
        if layer is None or layer.ndim != 3 or layer.shape[2] != 4:
            raise FileNotFoundError(f"Missing RGBA layer {name} in {send2unity_dir}; run cut_img.py first.")
        cut = cv2.cvtColor(styled, cv2.COLOR_BGR2BGRA)
        cut[:, :, 3] = layer[:, :, 3]
        cv2.imwrite(os.path.join(out_dir, name), cut)

    if background is None:
        background = _inpaint_background(styled, front_holes(send2unity_dir))
    cv2.imwrite(os.path.join(out_dir, "merged_layer_03.png"), cv2.cvtColor(background, cv2.COLOR_BGR2BGRA))


def _stylize_one(name: str, photo: np.ndarray, style_dir: str, force: bool) -> dict[str, Any]:
    raw_path = os.path.join(style_dir, "styled_raw.png")
    meta_path = os.path.join(style_dir, "meta.json")
    os.makedirs(style_dir, exist_ok=True)

    # The API call is the only slow / paid part; reuse a previous result.
    if not force and os.path.exists(raw_path) and os.path.exists(meta_path):
        styled = cv2.imread(raw_path)
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["cached"] = True
    else:
        start = time.time()
        styled, usage = call_muse_image(photo, STYLES[name]["prompt"])
        meta = {"seconds": round(time.time() - start, 1), "usage": usage, "raw_size": [styled.shape[1], styled.shape[0]]}
        cv2.imwrite(raw_path, styled)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        meta["cached"] = False
    return {"styled": styled, **meta}


def prefetch_style(name: str, photo: np.ndarray, send2unity_dir: str) -> None:
    """Make the restyle call ahead of time; stylize_run then reuses the cached result.

    A restyle needs only the photo, so infer.py starts it while DA3, SAM 3.1
    and Muse Spark run.
    """
    _stylize_one(name, photo, os.path.join(send2unity_dir, "styles", name), force=True)


def _style_worker(
    name: str,
    photo: np.ndarray,
    style_dir: str,
    holes: np.ndarray | None,
    force: bool,
    setting: str | None,
    removed: list[str] | None,
) -> dict[str, Any]:
    """Restyle, register, then (holes given) fill that style's background with Muse Image."""
    result = _stylize_one(name, photo, style_dir, force)
    result["aligned"], result["alignment"] = register_to_photo(result["styled"], photo)
    result["background"], result["fill"] = None, {"source": "inpaint"}
    if holes is not None:
        try:
            result["background"], result["fill"] = fill_background(
                result["aligned"], holes, style_dir, force, setting, removed, style=name
            )
        except Exception as exc:
            result["fill"] = {"source": "inpaint", "error": str(exc)}
            print(f"  ! {name}: Muse Image fill failed ({exc}); inpainting instead")
    return result


def stylize_run(
    image_path: str,
    send2unity_dir: str,
    styles: list[str] | None = None,
    force: bool = False,
    photo_background: bool = False,
    setting: str | None = None,
    removed: list[str] | None = None,
) -> dict[str, Any]:
    """Restyle one pipeline run in every requested style; writes styles.json.

    Each style's background layer is filled by Muse Image (unless
    MUSE_IMAGE_FILL=0), told the run's `setting` and the `removed` objects not
    to redraw; photo_background also fills the photo's own, in parallel with the styles.
    """
    styles = list(STYLES) if styles is None else styles
    photo = cv2.imread(image_path)
    if photo is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    holes = front_holes(send2unity_dir) if fill_enabled() else None
    track_json = os.path.join(send2unity_dir, "track_points.json")
    points = []
    if os.path.exists(track_json):
        with open(track_json, encoding="utf-8") as f:
            points = json.load(f).get("points", [])

    layer_masks = []
    for name in ("merged_layer_01.png", "merged_layer_02.png"):
        layer = cv2.imread(os.path.join(send2unity_dir, name), cv2.IMREAD_UNCHANGED)
        if layer is not None and layer.ndim == 3 and layer.shape[2] == 4:
            layer_masks.append(layer[:, :, 3] > 127)

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    photo_background = photo_background and holes is not None
    tasks = len(styles) + int(photo_background)
    workers = max(1, min(tasks, int(os.getenv("MUSE_IMAGE_WORKERS", "6"))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        photo_future = (
            pool.submit(fill_photo_background, image_path, send2unity_dir, force, setting, removed) if photo_background else None
        )
        futures = {
            name: pool.submit(
                _style_worker, name, photo, os.path.join(send2unity_dir, "styles", name), holes, force, setting, removed
            )
            for name in styles
        }
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:  # one failed style must not sink the others
                errors[name] = str(exc)
                print(f"  ✘ {name}: {exc}")
        photo_entry = photo_future.result() if photo_future else None

    entries = []
    for name in styles:
        if name not in results:
            continue
        result = results[name]
        style_dir = os.path.join(send2unity_dir, "styles", name)
        aligned, alignment = result["aligned"], result["alignment"]
        alignment["outline_offset"] = outline_offset(photo, aligned, layer_masks)
        cv2.imwrite(os.path.join(style_dir, "styled.png"), aligned)
        export_styled_layers(aligned, send2unity_dir, style_dir, result["background"])
        entries.append(
            {
                "name": name,
                "label": STYLES[name]["label"],
                "color": STYLES[name]["color"],
                "prompt": STYLES[name]["prompt"],
                "dir": f"styles/{name}",
                "seconds": result.get("seconds"),
                "usage": result.get("usage"),
                "cached": result["cached"],
                "alignment": alignment,
                "background": result["fill"],
                "track_color": track_color(aligned, points) if points else None,
            }
        )
        registered = f", registered at scale {alignment['scale']}" if alignment["registered"] else ""
        source = "cached" if result["cached"] else f"{result.get('seconds')}s"
        offset = alignment["outline_offset"]
        fit = f", outlines off by {offset['median_px']} px ({offset['within_3px']:.0%} within 3 px)" if offset else ""
        fill = result["fill"]
        filled = f", background filled {'cached' if fill.get('cached') else str(fill['seconds']) + 's'}" if fill["source"] == "muse-image" else ""
        print(f"  ✔ {name}: edge F1 {alignment['edge_f1_raw']:.3f} → {alignment['edge_f1']:.3f}{fit}{registered} ({source}{filled})")

    # Restyling a subset keeps the styles already in the manifest.
    manifest_path = os.path.join(send2unity_dir, "styles.json")
    previous = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            previous = {e["name"]: e for e in json.load(f).get("styles", [])}
    previous.update({e["name"]: e for e in entries})
    for name in errors:
        previous.pop(name, None)
    merged = [previous[name] for name in STYLES if name in previous]

    manifest = {"model": model_id(), "source_image": os.path.basename(image_path), "styles": merged, "errors": errors}
    if photo_entry is not None:
        manifest["photo_background"] = photo_entry
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Restyle one pipeline run with Muse Image.")
    parser.add_argument("image_path", help="The (resized) input image the pipeline ran on")
    parser.add_argument("send2unity_dir", help="Directory holding merged_layer_0*.png and track_points.json")
    parser.add_argument("--styles", default="all", help=f"Comma-separated subset of {list(STYLES)}, 'all' or 'none'")
    parser.add_argument("--force", action="store_true", help="Call Muse Image again even if a result is cached")
    parser.add_argument("--no-photo-background", action="store_true", help="Keep the photo's inpainted background layer")
    parser.add_argument("--objects", default=None, help="The run's *_sam31_objects.json, for the setting and removed objects the background fill is told")
    args = parser.parse_args()
    setting = load_setting(args.objects, cv2.imread(args.image_path)) if fill_enabled() else None
    stylize_run(
        args.image_path,
        args.send2unity_dir,
        parse_styles(args.styles),
        force=args.force,
        photo_background=not args.no_photo_background,
        setting=setting,
        removed=front_objects(args.send2unity_dir, args.objects, classify=True) if fill_enabled() else None,
    )


if __name__ == "__main__":
    main()
