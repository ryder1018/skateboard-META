import os
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch

_SAM2_GENERATOR_CACHE = {}
_SAM2_GENERATOR_LOCK = Lock()


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


def _mask_score(mask_record: dict[str, Any]) -> float:
    predicted_iou = float(mask_record.get("predicted_iou", 0.0))
    stability = float(mask_record.get("stability_score", 0.0))
    return predicted_iou * 0.65 + stability * 0.35


def _get_sam2_generator(model_id: str, device: str):
    from sam2.build_sam import build_sam2_hf
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    cache_key = (
        model_id,
        device,
        # _get_int_env("SAM2_POINTS_PER_SIDE", 32),
        _get_int_env("SAM2_POINTS_PER_SIDE",24),
        _get_int_env("SAM2_POINTS_PER_BATCH", 64),
        _get_float_env("SAM2_PRED_IOU_THRESH", 0.82),
        _get_float_env("SAM2_STABILITY_SCORE_THRESH", 0.9),
        _get_float_env("SAM2_BOX_NMS_THRESH", 0.70),
        # _get_int_env("SAM2_CROP_N_LAYERS", 1),
        _get_int_env("SAM2_CROP_N_LAYERS", 0),
        _get_float_env("SAM2_CROP_NMS_THRESH", 0.70),
        _get_int_env("SAM2_MIN_MASK_REGION_AREA", 128),
    )

    with _SAM2_GENERATOR_LOCK:
        if cache_key not in _SAM2_GENERATOR_CACHE:
            sam2_model = build_sam2_hf(model_id, device=device)
            sam2_model.eval()
            _SAM2_GENERATOR_CACHE[cache_key] = SAM2AutomaticMaskGenerator(
                sam2_model,
                points_per_side=cache_key[2],
                points_per_batch=cache_key[3],
                pred_iou_thresh=cache_key[4],
                stability_score_thresh=cache_key[5],
                box_nms_thresh=cache_key[6],
                crop_n_layers=cache_key[7],
                crop_nms_thresh=cache_key[8],
                min_mask_region_area=cache_key[9],
                output_mode="binary_mask",
            )
        return _SAM2_GENERATOR_CACHE[cache_key]


def generate_sam2_label_map(
    image_bgr: np.ndarray,
    depth16: np.ndarray,
    model_id: str | None = None,
) -> tuple[np.ndarray, int]:
    """Generate a non-overlapping label map from SAM2 automatic masks.

    The returned label map uses 0 for uncovered pixels and 1..N for accepted
    SAM2 regions. Downstream layer export sorts these regions by DA3 depth.
    """
    try:
        import sam2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 is not installed. Install the wsl-cv dependencies again, "
            "or install facebookresearch/sam2 in this environment."
        ) from exc

    if image_bgr is None or depth16 is None:
        raise ValueError("image_bgr and depth16 are required")

    height, width = depth16.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    model_id = model_id or os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = _get_sam2_generator(model_id, device)

    autocast_enabled = device == "cuda"
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled)
    with torch.inference_mode(), autocast_context:
        mask_records = generator.generate(image_rgb)

    min_area = int(height * width * _get_float_env("SAM2_MIN_AREA_RATIO", 0.001))
    max_area = int(height * width * _get_float_env("SAM2_MAX_AREA_RATIO", 0.55))
    min_new_ratio = _get_float_env("SAM2_MIN_NEW_AREA_RATIO", 0.20)

    filtered: list[dict[str, Any]] = []
    for record in mask_records:
        mask = np.asarray(record.get("segmentation"), dtype=bool)
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

        area = int(mask.sum())
        if area < min_area or area > max_area:
            continue

        filtered.append({**record, "segmentation": mask, "area": area})

    # Prefer high-quality small/detail masks first, then fill remaining pixels
    # with larger masks that still add meaningful uncovered area.
    filtered.sort(key=lambda record: (-_mask_score(record), int(record["area"])))

    labels = np.zeros((height, width), dtype=np.int32)
    occupied = np.zeros((height, width), dtype=bool)
    next_label = 1

    for record in filtered:
        mask = record["segmentation"]
        new_pixels = mask & ~occupied
        new_area = int(new_pixels.sum())
        if new_area < min_area:
            continue
        if new_area / max(int(record["area"]), 1) < min_new_ratio:
            continue

        labels[new_pixels] = next_label
        occupied[new_pixels] = True
        next_label += 1

    if next_label == 1:
        raise RuntimeError("SAM2 generated no usable masks after filtering.")

    return labels, next_label - 1
