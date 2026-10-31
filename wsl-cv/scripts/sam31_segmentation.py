import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import requests

SAM31_API_URL = "https://api.meta.ai/v1/responses"
SAM31_MODEL_ID = "sam-3.1"

# Muse Spark looks at each photo and proposes what SAM 3.1 should segment.
# These fixed lists are only the fallback when that proposal fails.
# "ground" alone misses streets, so road surfaces are queried explicitly.
SKATEABLE_PROMPTS = ("stairs", "handrail", "bench", "ledge", "ramp", "ground", "road", "sidewalk")
# Scene concepts that are not ridden but give the depth layering real objects
# to separate into foreground/background, like SAM2's automatic masks did.
SCENERY_PROMPTS = ("tree", "building", "car", "person")
GROUND_PROMPTS = ("floor", "ground")

# One sentence about the place itself, which muse_image.py gives Muse Image
# when it fills in the background behind the front layers.
SETTING_GUIDE = """setting: one sentence describing the place itself, so an artist could paint what lies
behind the objects: indoor or outdoor, the floor or ground, the walls, the sky, the surface things
stand on and the lighting. Do not name movable objects, furniture, vehicles or people. E.g. "a dim
indoor desk corner: a wooden desk top with a dark mat against a plain grey wall"."""

# Which concepts are the place itself; muse_image.py tells Muse Image not to
# redraw the others (the things standing in it) when it fills the background.
FIXED_GUIDE = """fixed: the phrases from your lists that are part of the place itself rather than things
standing in it: ground, floors, roads, sidewalks, curbs, lawns, fields, stairs, water, walls,
buildings, mountains, desk or table tops. People, furniture, vehicles, poles, signs, plants, trees
and other objects are not fixed. Use the exact phrases."""

CONCEPT_INSTRUCTIONS = """You plan image segmentation for a 2.5D side-scrolling skateboarding game
that turns a player's photo into a level. List what a text-prompted segmentation model should
find in this photo, as short concrete noun phrases (1-3 words, singular), e.g. "handrail",
"concrete ledge", "fire hydrant", "wooden table".

skateable: things a skater could roll on, grind, or jump over or onto. Always include the main
surface the skater would roll on (road, sidewalk, floor, grass, table top...), then obstacles
(stairs, rails, benches, ledges, curbs, hydrants, boxes, poles...). Small-scale scenes count too:
on a desk, a keyboard can be a ledge and the desk top the floor. Most important first.

scenery: other large, distinct objects that are not ridden but help separate foreground from
background (trees, buildings, cars, people, furniture...).

Only list things that are clearly visible. Never list the same thing twice under different
words, and never list "sky", "background" or the whole image.

""" + SETTING_GUIDE + "\n\n" + FIXED_GUIDE

CONCEPT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["skateable", "scenery", "setting", "fixed"],
    "properties": {
        "skateable": {"type": "array", "items": {"type": "string"}},
        "scenery": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
        "fixed": {"type": "array", "items": {"type": "string"}},
    },
}

FIXED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fixed"],
    "properties": {"fixed": {"type": "array", "items": {"type": "string"}}},
}

SETTING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["setting"],
    "properties": {"setting": {"type": "string"}},
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


def _get_prompts_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(prompt.strip() for prompt in value.split(",") if prompt.strip())


def _image_data_url(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode image for SAM 3.1.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _stream_output_text(api_key: str, prompt: str, image_url: str) -> str:
    """Send one concept prompt to SAM 3.1 and return the concatenated output text."""
    response = requests.post(
        os.getenv("SAM31_API_URL", SAM31_API_URL),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        json={
            "model": os.getenv("SAM31_MODEL_ID", SAM31_MODEL_ID),
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            "metadata": {"include_confidence": "true"},
            "stream": True,
        },
        stream=True,
        timeout=_get_float_env("SAM31_TIMEOUT", 120.0),
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"SAM 3.1 request for '{prompt}' failed "
            f"(HTTP {response.status_code}): {response.text[:500]}"
        )

    chunks: list[str] = []
    with response:
        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            event = json.loads(data)
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                chunks.append(event.get("delta") or "")
            elif event_type in ("error", "response.failed", "response.incomplete"):
                raise RuntimeError(f"SAM 3.1 stream for '{prompt}' ended with {event_type}: {data[:500]}")
    return "".join(chunks)


def _parse_instances(output_text: str, height: int, width: int) -> list[dict[str, Any]]:
    """Decode SAM 3.1 box/mask records into full-frame boolean masks."""
    from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format

    parser = image_segmentation_format().create_parser()
    parser.push(output_text, emit=False)
    result = parser.finish(CompletedOutcome()).result
    for diagnostic in result.diagnostics:
        print(f"  ! SAM 3.1 parser {diagnostic.severity}: {diagnostic.code} {diagnostic.message}")

    box_confidence = {
        record.object_id: record.confidence for record in result.records if record.kind == "box"
    }

    instances: list[dict[str, Any]] = []
    for record in result.records:
        if record.kind != "mask":
            continue
        left, top = int(record.bounds.left), int(record.bounds.top)
        right, bottom = min(int(record.bounds.right), width), min(int(record.bounds.bottom), height)
        if right <= left or bottom <= top:
            continue

        raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
        raster = raster.reshape(record.mask.height, record.mask.width)
        box_h, box_w = bottom - top, right - left
        if raster.shape != (box_h, box_w):
            # Masks are box-local; snap any size mismatch onto the box.
            raster = cv2.resize(raster, (box_w, box_h), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros((height, width), dtype=bool)
        mask[top:bottom, left:right] = raster.astype(bool)

        confidence = record.confidence
        if confidence is None:
            confidence = box_confidence.get(record.object_id)
        instances.append(
            {
                "box": [left, top, right, bottom],
                "score": None if confidence is None else float(confidence),
                "mask": mask,
            }
        )
    return instances


def segment_concepts(image_bgr: np.ndarray, prompts: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    """Query SAM 3.1 once per noun-phrase prompt (the API takes one concept per request)."""
    try:
        import meta_sam_parser  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "meta-sam-parser is not installed. Install the wsl-cv dependencies again."
        ) from exc

    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("MODEL_API_KEY is not set; SAM 3.1 runs through the Meta Model API.")

    height, width = image_bgr.shape[:2]
    image_url = _image_data_url(image_bgr)

    def run(prompt: str) -> list[dict[str, Any]]:
        return _parse_instances(_stream_output_text(api_key, prompt, image_url), height, width)

    max_workers = max(1, min(len(prompts), _get_int_env("SAM31_MAX_WORKERS", 4)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(zip(prompts, pool.map(run, prompts)))


def _clean_concepts(phrases: list[str], limit: int, taken: set[str]) -> tuple[str, ...]:
    cleaned = []
    for phrase in phrases:
        phrase = " ".join(phrase.lower().split())
        if not phrase or len(phrase.split()) > 4 or phrase in taken:
            continue
        taken.add(phrase)
        cleaned.append(phrase)
    return tuple(cleaned[:limit])


def propose_concepts(image_bgr: np.ndarray) -> tuple[tuple[str, ...], tuple[str, ...], str, list[str]]:
    """Ask Muse Spark which concepts in this photo SAM 3.1 should segment."""
    from muse_spark import call_muse_spark, image_data_url

    raw_text, _ = call_muse_spark(
        CONCEPT_INSTRUCTIONS,
        [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "List the concepts to segment in this photo."},
                    {"type": "input_image", "image_url": image_data_url(image_bgr)},
                ],
            }
        ],
        "segmentation_concepts",
        CONCEPT_SCHEMA,
        os.getenv("SAM31_CONCEPT_REASONING_EFFORT", "low"),
    )
    proposal = json.loads(raw_text)
    taken: set[str] = set()
    skateable = _clean_concepts(proposal["skateable"], _get_int_env("SAM31_MAX_SKATEABLE_CONCEPTS", 8), taken)
    scenery = _clean_concepts(proposal["scenery"], _get_int_env("SAM31_MAX_SCENERY_CONCEPTS", 6), taken)
    if not skateable:
        raise RuntimeError("Muse Spark proposed no skateable concepts.")
    fixed = [" ".join(p.lower().split()) for p in proposal["fixed"]]
    return skateable, scenery, proposal["setting"].strip(), [p for p in fixed if p in taken]


def classify_fixed(concepts: list[str]) -> list[str]:
    """Muse Spark's `fixed` subset of a run's concepts, for runs made before propose_concepts wrote one."""
    from muse_spark import call_muse_spark

    raw_text, _ = call_muse_spark(
        "You sort the segmentation concepts of a photo.\n\n" + FIXED_GUIDE,
        [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": json.dumps(concepts)}]}],
        "fixed",
        FIXED_SCHEMA,
        "low",
    )
    fixed = {" ".join(p.lower().split()) for p in json.loads(raw_text)["fixed"]}
    return [c for c in concepts if c in fixed]


def describe_setting(image_bgr: np.ndarray) -> str:
    """Muse Spark's one-sentence setting, for runs made before propose_concepts wrote one."""
    from muse_spark import call_muse_spark, image_data_url

    raw_text, _ = call_muse_spark(
        "Describe the place in this photo.\n\n" + SETTING_GUIDE,
        [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe the setting of this photo."},
                    {"type": "input_image", "image_url": image_data_url(image_bgr)},
                ],
            }
        ],
        "setting",
        SETTING_SCHEMA,
        "low",
    )
    return json.loads(raw_text)["setting"].strip()


def _resolve_prompts(
    image_bgr: np.ndarray,
) -> tuple[tuple[str, ...], tuple[str, ...], str, str | None, list[str] | None]:
    """Return (skateable, scenery, source, setting, fixed); source is "muse-spark", "env" or "default"."""
    if os.getenv("SAM31_SKATEABLE_PROMPTS") is not None:
        print("  ✔ Using SAM 3.1 concepts from SAM31_SKATEABLE_PROMPTS / SAM31_SCENERY_PROMPTS")
        return (
            _get_prompts_env("SAM31_SKATEABLE_PROMPTS", SKATEABLE_PROMPTS),
            _get_prompts_env("SAM31_SCENERY_PROMPTS", SCENERY_PROMPTS),
            "env",
            None,
            None,
        )
    try:
        skateable, scenery, setting, fixed = propose_concepts(image_bgr)
    except Exception as exc:
        print(f"  ! Muse Spark concept proposal failed ({exc}); using default concepts.")
        return SKATEABLE_PROMPTS, SCENERY_PROMPTS, "default", None, None
    # The track needs a ground to follow; a specific phrase ("foam floor") can
    # miss, so the generic ones are always queried too.
    skateable = tuple(dict.fromkeys(skateable + GROUND_PROMPTS))
    print(f"  ✔ Muse Spark concepts: skateable={list(skateable)} scenery={list(scenery)}")
    print(f"  ✔ Muse Spark setting: {setting}")
    return skateable, scenery, "muse-spark", setting, list(dict.fromkeys(fixed + list(GROUND_PROMPTS)))


def generate_sam31_label_map(
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, int, list[dict[str, Any]], dict[str, Any]]:
    """Generate a non-overlapping label map from SAM 3.1 text-prompted masks.

    The returned label map uses 0 for uncovered pixels and 1..N for accepted
    SAM 3.1 instances. Downstream layer export sorts these regions by DA3 depth.
    The object list keeps each instance's concept, box and label id so later
    stages know "this is a handrail", not just "this is a region".
    """
    if image_bgr is None:
        raise ValueError("image_bgr is required")

    started = time.perf_counter()
    skateable, scenery, source, setting, fixed = _resolve_prompts(image_bgr)
    concepts_seconds = time.perf_counter() - started
    prompts = tuple(dict.fromkeys(skateable + scenery))
    started = time.perf_counter()
    by_prompt = segment_concepts(image_bgr, prompts)
    sam_seconds = time.perf_counter() - started

    height, width = image_bgr.shape[:2]
    min_area = int(height * width * _get_float_env("SAM31_MIN_AREA_RATIO", 0.001))
    min_score = _get_float_env("SAM31_MIN_CONFIDENCE", 0.3)

    objects: list[dict[str, Any]] = []
    for prompt in prompts:
        instances = by_prompt[prompt]
        print(f"  ✔ SAM 3.1 '{prompt}': {len(instances)} instance(s)")
        for instance in instances:
            score = instance["score"]
            area = int(instance["mask"].sum())
            if area < min_area or (score is not None and score < min_score):
                continue
            objects.append(
                {
                    "label": prompt,
                    "skateable": prompt in skateable,
                    "score": score,
                    "box": instance["box"],
                    "area": area,
                    "mask": instance["mask"],
                }
            )

    # Thin and small objects (handrails, ledges) are claimed before the large
    # ones they overlap (ground, buildings), so they keep their own region.
    objects.sort(key=lambda obj: obj["area"])

    labels = np.zeros((height, width), dtype=np.int32)
    occupied = np.zeros((height, width), dtype=bool)
    next_label = 1
    counts: dict[str, int] = {}

    for obj in objects:
        counts[obj["label"]] = counts.get(obj["label"], 0) + 1
        obj["id"] = f"{obj['label']}_{counts[obj['label']]}"
        new_pixels = obj.pop("mask") & ~occupied
        obj["label_id"] = None
        if int(new_pixels.sum()) < min_area:
            continue

        labels[new_pixels] = next_label
        occupied[new_pixels] = True
        obj["label_id"] = next_label
        next_label += 1

    if next_label == 1:
        raise RuntimeError("SAM 3.1 found no usable objects for the configured prompts.")

    info = {
        "concepts": {"skateable": list(skateable), "scenery": list(scenery), "source": source, "setting": setting, "fixed": fixed},
        "timings": {"concepts": round(concepts_seconds, 2), "sam31": round(sam_seconds, 2)},
    }
    return labels, next_label - 1, objects, info
