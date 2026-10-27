"""Muse Spark level designer: picks the riding surface and writes the mission.

The track itself keeps the original SkateP logic: it is the smoothed top edge
of the ground surface (extract_track.py), and every other object stays in the
scene as an occluder or backdrop, so the skater weaves through the photo
instead of climbing over it. Muse Spark replaces the fragile geometric guess
of *which* region is the ground:

  1. Muse Spark sees the photo and the SAM 3.1 objects and picks the surface
     object(s) the track should follow, plus a mission line.
  2. The choice is verified against the masks (does it span the scene? is it
     at ground level?). Violations go back to Muse Spark, up to N attempts.
  3. The chosen label ids are written next to the label map; cut_img.py uses
     them as the track region instead of its heuristic.

After the track is extracted, finalize_level() adds the "level" field to
track_points.json: mission, surfaces, and the occluders the skater passes behind.
"""

import json
import os
import time
from typing import Any

import cv2
import numpy as np
import scipy.ndimage

from muse_spark import call_muse_spark, image_data_url, model_id

GRID = 1000.0

INSTRUCTIONS = """You design levels for a 2.5D side-scrolling skateboarding game made from a photo.
The skater rolls from the left edge to the right edge along the TOP EDGE of the ground surface
you choose. Every other object stays in the scene: objects closer to the camera than the track
hide the skater as they pass behind them, which gives the level its depth. So pick the ground,
not the obstacles.

Coordinates use a 0..1000 grid (x=0 left, y=0 top). For each object you get its box, how much
of the image width it spans ("width_coverage", 0..1), and the median height of its top edge
("top_edge_y").

Choose "surface_ids": the object(s) that together form one continuous riding surface across
the scene, such as sidewalk, road, floor, plaza, desk top, table, grass. Prefer wide surfaces at
ground level. Do not choose standalone obstacles (mugs, hydrants, poles, people, signs) or
anything high up in the image. Combine two or three surfaces only if one alone does not span
the scene.

"mission" is one short, fun sentence the player sees, naming real objects they will ride past
or behind (e.g. "Cruise the desk and slip behind the Saucy mug"). "design_notes" briefly
explains the choice."""

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["surface_ids", "mission", "design_notes"],
    "properties": {
        "surface_ids": {"type": "array", "items": {"type": "string"}},
        "mission": {"type": "string"},
        "design_notes": {"type": "string"},
    },
}


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def surfaces_path_for(labels_path: str) -> str:
    """Sidecar file cut_img.py reads to use the chosen surfaces as the track region."""
    return labels_path.replace("_depth_mask.npy", "_track_surfaces.json")


def _top_edge(mask: np.ndarray) -> np.ndarray:
    """Topmost row per column (NaN where the column is empty)."""
    has = mask.any(axis=0)
    top = np.argmax(mask, axis=0).astype(np.float32)
    top[~has] = np.nan
    return top


# ----------------------------------------------------------------------
# Muse Spark: choose the riding surface
# ----------------------------------------------------------------------


def _describe_scene(objects: list[dict[str, Any]], labels: np.ndarray) -> list[dict[str, Any]]:
    height, width = labels.shape
    described = []
    for obj in objects:
        if obj.get("label_id") is None:
            continue
        mask = labels == obj["label_id"]
        top = _top_edge(mask)
        x1, y1, x2, y2 = obj["box"]
        described.append(
            {
                "id": obj["id"],
                "label": obj["label"],
                "skateable": obj["skateable"],
                "box": [round(x1 * GRID), round(y1 * GRID), round(x2 * GRID), round(y2 * GRID)],
                "width_coverage": round(float(mask.any(axis=0).mean()), 2),
                "top_edge_y": round(float(np.nanmedian(top)) / (height - 1) * GRID) if np.isfinite(top).any() else None,
            }
        )
    return described


def verify_surfaces(surface_ids: list[str], objects: dict[str, dict[str, Any]], labels: np.ndarray) -> list[str]:
    """Return human-readable violations; an empty list means the surface choice is usable."""
    height, _ = labels.shape
    errors = []
    unknown = [sid for sid in surface_ids if sid not in objects]
    if unknown:
        errors.append(f"unknown object id(s) {unknown}; use ids from the object list.")
    known = [sid for sid in surface_ids if sid in objects]
    if not known:
        errors.append("no surface chosen; pick the ground object(s) the skater rides on.")
        return errors

    union = np.isin(labels, [objects[sid]["label_id"] for sid in known])
    coverage = float(union.any(axis=0).mean())
    min_coverage = _get_float_env("LEVEL_MIN_TRACK_COVERAGE", 0.6)
    if coverage < min_coverage:
        errors.append(
            f"the chosen surface(s) span only {coverage:.0%} of the image width; the track must cross "
            f"at least {min_coverage:.0%}. Add another ground surface or choose a wider one."
        )

    top = _top_edge(union)
    top_y = float(np.nanmedian(top)) / (height - 1) * GRID
    min_top = _get_float_env("LEVEL_MIN_TRACK_TOP", 250.0)
    if top_y < min_top:
        errors.append(
            f"the chosen surface's top edge sits at y={top_y:.0f}, in the upper part of the image; "
            "the track must follow a ground-level surface."
        )

    for sid in known:
        if not objects[sid]["skateable"]:
            errors.append(f"'{sid}' is scenery ({objects[sid]['label']}), not something to ride on.")
    return errors


def plan_track(image_path: str, labels_path: str, objects_path: str) -> dict[str, Any]:
    """Let Muse Spark pick the track surface; writes the sidecar cut_img.py reads."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    labels = np.load(labels_path)
    if labels.shape != image_bgr.shape[:2]:
        labels = cv2.resize(labels.astype(np.int32), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    with open(objects_path, encoding="utf-8") as f:
        object_list = json.load(f)["objects"]
    objects = {obj["id"]: obj for obj in object_list if obj.get("label_id") is not None}
    if not objects:
        raise RuntimeError("No SAM 3.1 objects to choose a track surface from.")

    conversation: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Choose the riding surface and write the mission. Objects:\n"
                    + json.dumps(_describe_scene(object_list, labels)),
                },
                {"type": "input_image", "image_url": image_data_url(image_bgr)},
            ],
        }
    ]

    attempts: list[dict[str, Any]] = []
    max_attempts = max(1, _get_int_env("MUSE_SPARK_MAX_ATTEMPTS", 3))
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        raw_text, usage = call_muse_spark(
            INSTRUCTIONS,
            conversation,
            "track_plan",
            PLAN_SCHEMA,
            os.getenv("MUSE_SPARK_REASONING_EFFORT", "medium"),
        )
        plan = json.loads(raw_text)
        errors = verify_surfaces(plan["surface_ids"], objects, labels)
        attempts.append(
            {
                "attempt": attempt,
                "plan": plan,
                "errors": errors,
                "usage": usage,
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
        print(f"  ✔ Muse Spark attempt {attempt}: surfaces={plan['surface_ids']} ({len(errors)} violation(s))")
        for error in errors:
            print(f"    - {error}")
        if not errors:
            break

        conversation.append(
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": raw_text}]}
        )
        conversation.append(
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "The surface choice failed verification against the segmentation masks:\n- "
                        + "\n- ".join(errors)
                        + "\nReturn a corrected choice.",
                    }
                ],
            }
        )

    best = min(attempts, key=lambda entry: len(entry["errors"]))
    if best["errors"]:
        raise RuntimeError(f"Muse Spark could not choose a valid surface: {best['errors']}")

    chosen = [sid for sid in best["plan"]["surface_ids"] if sid in objects]
    result = {
        "surface_ids": chosen,
        "label_ids": [objects[sid]["label_id"] for sid in chosen],
        "mission": best["plan"]["mission"],
        "design_notes": best["plan"]["design_notes"],
        "attempts": attempts,
    }
    with open(surfaces_path_for(labels_path), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  ✔ Track surface: {chosen} — {result['mission']}")
    return result


# ----------------------------------------------------------------------
# Before track extraction: let the track continue behind occluders
# ----------------------------------------------------------------------


def _runs(columns: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Group sorted column indices into [start, end] runs, bridging small gaps."""
    runs: list[tuple[int, int]] = []
    for x in columns:
        if runs and x - runs[-1][1] <= max_gap:
            runs[-1] = (runs[-1][0], int(x))
        else:
            runs.append((int(x), int(x)))
    return runs


def repair_track_occlusions(track_mask_path: str, labels_path: str) -> int:
    """Let the track continue behind objects standing on the ground.

    extract_track.py follows the topmost ground pixel per column, so a hydrant,
    a person or a guitar case standing on the floor pulls the line down to
    their base. For every object sitting on the ground's top edge, the edge
    under it is rebuilt from the uncovered ground on both sides (one side at
    the image border) and the mask is filled up to it. Edits
    layer_00_mask.npy/.png in place; returns the number of repaired columns.
    """
    mask = np.load(track_mask_path) > 127
    labels = np.load(labels_path)
    height, width = mask.shape
    if labels.shape != mask.shape:
        labels = cv2.resize(labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)

    top = _top_edge(mask)
    valid = np.isfinite(top)
    if valid.sum() < 2:
        return 0
    track_labels = set(np.unique(labels[mask]).tolist())
    reach = max(3, int(height * 0.03))  # masks rarely touch exactly

    # Columns where some other object sits right on top of the ground edge.
    touching: dict[int, list[int]] = {}
    for x in np.flatnonzero(valid):
        y = int(top[x])
        for v in np.unique(labels[max(0, y - reach) : y, x]):
            if v > 0 and v not in track_labels:
                touching.setdefault(int(v), []).append(int(x))
    # A wall meeting the ground along most of the width is backdrop, not an
    # object standing on it; its edge is already the real ground line.
    standing = np.zeros(width, dtype=bool)
    for cols in touching.values():
        if len(cols) <= 0.6 * valid.sum():
            standing[cols] = True

    # Unsegmented clutter next to an object (a bag behind a guitar case) dips
    # the edge too: absorb neighbouring columns that sit clearly below the
    # ground's upper envelope, but only where they touch a standing object,
    # so genuine slopes elsewhere stay untouched.
    filled = np.interp(np.arange(width), np.flatnonzero(valid), top[valid])
    envelope = scipy.ndimage.minimum_filter1d(filled, size=max(3, width // 2), mode="nearest")
    low = valid & (filled > envelope + 0.02 * height)
    gap = max(2, width // 100)
    for a, b in _runs(np.flatnonzero(low), max_gap=gap):
        near = standing[max(0, a - gap) : b + gap + 1]
        if near.any():
            standing[a : b + 1] |= low[a : b + 1]

    free = valid & ~standing
    if free.sum() < 2:
        return 0
    free_cols = np.flatnonzero(free)
    repaired = 0
    for a, b in _runs(np.flatnonzero(standing), max_gap=max(2, width // 100)):
        left = free_cols[free_cols < a]
        right = free_cols[free_cols > b]
        ref_left = top[left[-1]] if left.size else None
        ref_right = top[right[0]] if right.size else None
        if ref_left is None and ref_right is None:
            continue
        cols = np.arange(a, b + 1)
        if ref_left is None:
            target = np.full(cols.size, ref_right)
        elif ref_right is None:
            target = np.full(cols.size, ref_left)
        else:
            target = np.interp(cols, [left[-1], right[0]], [ref_left, ref_right])
        for x, y in zip(cols, target):
            # Only lift the edge (fill behind the object), never cut ground away.
            if valid[x] and y < top[x]:
                mask[int(round(y)) : int(top[x]), x] = True
                repaired += 1

    if repaired:
        track = mask.astype(np.uint8) * 255
        np.save(track_mask_path, track)
        cv2.imwrite(track_mask_path.replace(".npy", ".png").replace("layer_00_mask", "track_mask"), track)
        print(f"  ✔ Track continues behind occluders ({repaired} columns repaired)")
    return repaired


# ----------------------------------------------------------------------
# After track extraction: occluders + level metadata
# ----------------------------------------------------------------------


def _track_y(points: list[list[float]], width: int, height: int) -> np.ndarray:
    xs = np.array([p[0] for p in points]) * (width - 1)
    ys = np.array([p[1] for p in points]) * (height - 1)
    return np.interp(np.arange(width), xs, ys, left=np.nan, right=np.nan)


def find_occluders(
    labels: np.ndarray,
    foreground_ids: set[int],
    track_mask: np.ndarray,
    objects: list[dict[str, Any]],
    points: list[list[float]],
) -> list[dict[str, Any]]:
    """Objects in the foreground layer that cover the skater's corridor above the track.

    The skater is drawn between the gameplay and foreground layers (as in
    Unity), so these are exactly the objects the player will see the skater
    pass behind.
    """
    height, width = labels.shape
    track_y = _track_y(points, width, height)
    corridor_h = int(height * _get_float_env("LEVEL_SKATER_HEIGHT", 0.15))

    rows = np.arange(height)[:, None]
    valid = np.isfinite(track_y)
    ty = np.where(valid, track_y, -1)[None, :]
    corridor = valid[None, :] & (rows <= ty) & (rows >= ty - corridor_h)

    occluders = []
    for obj in objects:
        if obj.get("label_id") is None or obj["label_id"] not in foreground_ids:
            continue
        mask = labels == obj["label_id"]
        if (mask & track_mask).sum() > 0.5 * mask.sum():
            continue  # part of the ground itself
        overlap = mask & corridor
        if overlap.sum() < 0.002 * height * width:
            continue
        columns = np.flatnonzero(overlap.any(axis=0))
        occluders.append(
            {
                "id": obj["id"],
                "label": obj["label"],
                "x_range": [round(columns[0] / (width - 1), 4), round(columns[-1] / (width - 1), 4)],
            }
        )
    return sorted(occluders, key=lambda o: o["x_range"][0])


MISSION_INSTRUCTIONS = """Write the one-line mission a player sees before riding a 2.5D skate level
made from their photo. Be short, fun and concrete: name the riding surface and, if any, the
objects the skater rides behind (they pass in front of the skater). Only mention objects from
the lists given."""

MISSION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mission"],
    "properties": {"mission": {"type": "string"}},
}


def write_mission(surfaces: list[str], occluders: list[dict[str, Any]], fallback: str) -> str:
    """Rewrite the mission once the occluders are known, so it never promises the wrong objects."""
    facts = {
        "riding_surface": surfaces,
        "rides_behind": [o["label"] for o in occluders],
    }
    try:
        raw_text, _ = call_muse_spark(
            MISSION_INSTRUCTIONS,
            [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": json.dumps(facts)}]}],
            "mission",
            MISSION_SCHEMA,
            "low",
        )
        return json.loads(raw_text)["mission"].strip() or fallback
    except Exception as exc:
        print(f"  ! Mission rewrite failed ({exc}); keeping the planned mission.")
        return fallback


def _draw_level(image_bgr: np.ndarray, track_mask: np.ndarray, labels: np.ndarray, level: dict[str, Any], points: list[list[float]], objects: dict[str, dict[str, Any]]) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    vis = (image_bgr * 0.55).astype(np.uint8)
    vis[track_mask] = (0.55 * image_bgr[track_mask] + 0.45 * np.array([90, 200, 90])).astype(np.uint8)
    for occ in level["occluders"]:
        mask = labels == objects[occ["id"]]["label_id"]
        vis[mask] = (0.6 * image_bgr[mask] + 0.4 * np.array([40, 140, 255])).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        cv2.putText(vis, f"behind: {occ['id']}", (int(xs.min()), max(14, int(ys.min()) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    stroke = max(2, width // 250)
    path = np.array([[int(x * (width - 1)), int(y * (height - 1))] for x, y in points], dtype=np.int32)
    cv2.polylines(vis, [path], False, (60, 230, 60), stroke, cv2.LINE_AA)
    cv2.putText(vis, level["mission"], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def finalize_level(
    image_path: str,
    labels_path: str,
    objects_path: str,
    send2unity_dir: str,
    plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Add mission, surfaces and occluders to track_points.json (keeps the legacy points)."""
    track_json = os.path.join(send2unity_dir, "track_points.json")
    with open(track_json, encoding="utf-8") as f:
        track = json.load(f)

    image_bgr = cv2.imread(image_path)
    height, width = image_bgr.shape[:2]
    labels = np.load(labels_path)
    if labels.shape != (height, width):
        labels = cv2.resize(labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
    track_mask = np.load(os.path.join(send2unity_dir, "layer_00_mask.npy")) > 127
    with open(os.path.join(send2unity_dir, "layers.json"), encoding="utf-8") as f:
        layer_info = json.load(f)
    # The ground as segmented, before repair_track_occlusions filled the area
    # behind standing objects (which would make them look like ground).
    ground = np.isin(labels, layer_info["track_region_ids"])
    with open(objects_path, encoding="utf-8") as f:
        object_list = json.load(f)["objects"]
    objects = {obj["id"]: obj for obj in object_list if obj.get("label_id") is not None}

    foreground_ids = set(layer_info["layers"][0]["region_ids"])
    occluders = find_occluders(labels, foreground_ids, ground, object_list, track["points"])
    surface_labels = [objects[sid]["label"] for sid in plan["surface_ids"]] if plan else []
    level = {
        "generator": model_id() if plan else "rule-based",
        "track": "top edge of the riding surface",
        # Draw order for the client: the track and skater sit between the
        # gameplay and foreground layers, so the foreground hides the skater.
        "render_order": ["background", "gameplay", "track", "foreground"],
        "mission": write_mission(surface_labels, occluders, plan["mission"]) if plan else "Ride the line",
        "design_notes": plan["design_notes"] if plan else "",
        "surfaces": plan["surface_ids"] if plan else [],
        "occluders": occluders,
        "validation": {
            "passed": plan is not None,
            "attempts": len(plan["attempts"]) if plan else 0,
            "errors": plan["attempts"][-1]["errors"] if plan else [],
        },
    }
    track["level"] = level
    # Unity's TrackData already has occluded_segments: [first, last] point
    # indices of each stretch the foreground layer draws over.
    xs = np.array([p[0] for p in track["points"]])
    segments = []
    for occ in occluders:
        inside = np.flatnonzero((xs >= occ["x_range"][0]) & (xs <= occ["x_range"][1]))
        if inside.size:
            segments.append([int(inside[0]), int(inside[-1])])
    track["occluded_segments"] = segments
    with open(track_json, "w", encoding="utf-8") as f:
        json.dump(track, f, indent=4)
    with open(os.path.join(send2unity_dir, "level_design_log.json"), "w", encoding="utf-8") as f:
        json.dump(plan["attempts"] if plan else [], f, indent=2)
    cv2.imwrite(os.path.join(send2unity_dir, "level_vis.png"), _draw_level(image_bgr, track_mask, labels, level, track["points"], objects))

    names = ", ".join(o["id"] for o in level["occluders"]) or "none"
    print(f"  ✔ Level saved: {level['mission']} (skater passes behind: {names})")
    return track
