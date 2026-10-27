"""Minimal Muse Spark client (Meta Model API, Responses endpoint) with structured output."""

import base64
import os
from typing import Any

import cv2
import numpy as np
import requests

MUSE_SPARK_API_URL = "https://api.meta.ai/v1/responses"
# The contributor variant is the same Muse Spark 1.3 model at ~1/12 the price;
# requests may be used by Meta to improve its products.
MUSE_SPARK_MODEL_ID = "muse-spark-1.3-contributor"


def model_id() -> str:
    return os.getenv("MUSE_SPARK_MODEL_ID", MUSE_SPARK_MODEL_ID)


def image_data_url(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode image for Muse Spark.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def call_muse_spark(
    instructions: str,
    conversation: list[dict[str, Any]],
    schema_name: str,
    schema: dict[str, Any],
    reasoning_effort: str = "medium",
) -> tuple[str, dict[str, Any]]:
    """Run one Muse Spark turn constrained to a JSON schema. Returns (json_text, usage)."""
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("MODEL_API_KEY is not set; Muse Spark runs through the Meta Model API.")

    try:
        timeout = float(os.getenv("MUSE_SPARK_TIMEOUT", "180"))
    except ValueError:
        timeout = 180.0

    response = requests.post(
        os.getenv("MUSE_SPARK_API_URL", MUSE_SPARK_API_URL),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model_id(),
            "instructions": instructions,
            "input": conversation,
            "reasoning": {"effort": reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Muse Spark request failed (HTTP {response.status_code}): {response.text[:500]}")

    body = response.json()
    if body.get("status") != "completed":
        raise RuntimeError(f"Muse Spark response not completed: {body.get('status')} {body.get('incomplete_details')}")
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                return part["text"], body.get("usage") or {}
    raise RuntimeError("Muse Spark response contained no output text.")
