# WSL CV Backend Documentation

This module handles the computer vision pipeline for the SkateP project. It receives images, extracts depth maps using Depth Anything 3, finds skateable objects (stairs, handrails, benches, ...) with SAM 3.1 text prompts through the Meta Model API, segments them into visual layers, and generates track points for Unity.

## 1. Environment Setup & Execution

We use `uv` for dependency management.

### Installation
```bash
# Inside the wsl-cv directory
uv sync
```

SAM 3.1 runs through the Meta Model API, so no SAM weights are downloaded. Set your API key before starting the server:

```bash
export MODEL_API_KEY=<your Meta Model API key>
```

`uv sync` installs `meta-sam-parser`, Meta's decoder for SAM 3.1 box/mask output.

### Running the API Server
Start the FastAPI backend server:
```bash
uv run uvicorn scripts.infer:app --host 0.0.0.0 --port 9000
```
Outputs and temporary job directories (`_jobs/`) are managed automatically during API execution.

### Running Scripts Manually
```bash
uv run python scripts/pipeline.py ./assets/pictures/whisky.jpg
```
Intermediate outputs will be saved to `outputs/` and `Send2Unity/`.

### Trying Photos
Drop photos into `wsl-cv/photos/` and run `./run_photos.sh` (inside the project env). Each new photo gets `showcase/<name>/showcase.html`, and `showcase/index.html` links them all; already-built photos are skipped (`--force` rebuilds, `--video` also renders MP4s). For a single photo anywhere on disk: `./try_photo.sh <photo> [name] [--video]`.

### 3D Pipeline Showcase
Every `scripts/pipeline.py` run also writes `<send2unity_dir>/showcase.html`: a self-contained three.js page that replays the run (photo → DA3 depth relief → Muse Spark concepts → SAM 3.1 objects → 2.5D layers → Muse Spark design/verify loop → ride). Open it in a browser (it loads three.js from a CDN). Controls: space play/pause, R restart, 1–7 jump to a step, drag to orbit, C back to the cinematic camera, H hide the UI for screen recording.

To render the same sequence to MP4 locally (headless Chromium + ffmpeg, ~1 s per 1080p frame):
```bash
pip install playwright && python -m playwright install chromium   # once
python scripts/render_showcase.py <send2unity_dir>/showcase.html --out showcase.mp4
```

## 2. Repository Map

*   `scripts/infer.py`: Active runtime API entrypoint (FastAPI server). It creates one isolated `_jobs/` directory per request and packages the Unity-facing JSON response.
*   `scripts/pipeline.py`: Standalone pipeline execution script.
*   `scripts/run_inference_depth_sam31.py`: DA3 depth generation plus SAM 3.1 segmentation; also writes `{base}_sam31_objects.json` (label, box, score, label id per object).
*   `scripts/sam31_segmentation.py`: Muse Spark first proposes this photo's concepts to segment (skateable + scenery); then SAM 3.1 API calls (one noun-phrase prompt per request), mask decoding, and non-overlapping label-map export. Falls back to a fixed concept list if the proposal fails.
*   `scripts/muse_spark.py`: Shared Muse Spark client (Responses API, JSON-schema structured output). Model defaults to `muse-spark-1.3-contributor`; override with `MUSE_SPARK_MODEL_ID`.
*   `scripts/cut_img.py`: Depth-sorted visual layer generation (foreground, gameplay, background) plus background inpainting.
*   `scripts/level_designer.py`: Muse Spark picks which SAM 3.1 object(s) are the riding surface (verified against the masks: must span the scene at ground level; violations are fed back for repair, up to `MUSE_SPARK_MAX_ATTEMPTS`, default 3). Also continues the track behind objects standing on the ground, finds the occluders the skater passes behind (closer than the track in DA3 depth), and has Muse Spark write the mission. Adds a `level` field to `track_points.json` (mission, surfaces, occluders, and `render_order`: background → gameplay → track/skater → foreground, matching Unity's Background / Midground / Foreground sorting layers) and fills Unity's existing `occluded_segments` (point-index ranges the foreground draws over). Writes `level_vis.png` / `level_design_log.json`. `cut_img.py` also writes `layers.json` (which SAM region went into which layer image).
*   `scripts/extract_track.py`: The original track extraction: the smoothed top edge of the track region (`layer_00_mask.npy`).
*   `scripts/showcase.py` + `scripts/showcase_template.html`: Builds the self-contained 3D pipeline showcase page from one run's outputs.
*   `scripts/render_showcase.py`: Renders the showcase page to MP4 frame by frame (Playwright + ffmpeg).
*   `scripts/depth_adapter.py`: Wrapper and integration point for the DA3 model.
*   `models/depth_anything_3/`: Vendor/model code (Do not modify).
*   `Send2Unity/`: Generated Unity-facing outputs for manual runs.
*   `outputs/`: Generated intermediate outputs.

## 3. Pipeline Flow & Responsibilities

### Execution Flow
1. Windows Server sends `POST /infer` with an image.
2. `scripts/infer.py` creates an isolated job environment in `_jobs/` with the uploaded image, symlinked scripts/models, and wrapper scripts for path-sensitive steps.
3. Step 1 runs `scripts/run_inference_depth_sam31.py` in process to generate a 16-bit DA3 depth map and a SAM 3.1 label map.
4. Step 2 asks Muse Spark (`level_designer.plan_track`) which object is the ground and writes `{base}_track_surfaces.json`; then `scripts/cut_img.py` runs through a wrapper to sort regions by average depth, export foreground/gameplay/background layers, inpaint, and build the track region from the chosen surface (its original geometric selection is the fallback).
5. Step 3 fills dips where objects stand on the ground (`repair_track_occlusions`), runs the original `scripts/extract_track.py` through a wrapper, then `finalize_level` adds mission, surfaces and occluders to `track_points.json`. The legacy `points` format is unchanged for Unity.
6. `scripts/infer.py` packages everything into Base64 JSON and responds.

### Known Fragile Points
*   **Job Orchestration**: `scripts/infer.py` uses isolated job directories, symlinks, and wrapper scripts to override hardcoded paths in path-sensitive legacy scripts. This is sensitive to script changes.
*   **DA3 Vendor Dependency**: `models/depth_anything_3/` must remain untouched. Any API changes there require updates to `scripts/depth_adapter.py`.
*   **SAM 3.1 Runtime Dependency**: SAM 3.1 is called at `https://api.meta.ai/v1/responses` with model `sam-3.1`, so the server needs network access and `MODEL_API_KEY`. Setting `SAM31_SKATEABLE_PROMPTS` / `SAM31_SCENERY_PROMPTS` (comma-separated) skips the Muse Spark concept proposal and uses those lists.
*   **Generated Folders**: Manual runs rely on `outputs/` and `Send2Unity/` folders being created correctly.

## 4. API Contract

The Windows server communicates with this module via a `POST` request to `http://<WSL_IP>:9000/infer`, attaching the image as `multipart/form-data` under the key `photo`.

### Current Response Format
```json
{
  "images": {
    "preview_base64": "iVBORw0KGgo...",
    "foreground_base64": "...",
    "gameplay_base64": "...",
    "background_base64": "..."
  },
  "metadata": {
    "version": "1.0",
    "timestamp": "2026-04-23T12:00:00.000Z",
    "source_image": "input.jpg",
    "aspect_ratio": 1.7778,
    "track_color": "#9e5752",
    "points": [ [0.1, 0.5], [0.2, 0.45] ]
  }
}
```

### Intended Future `LevelPayload` Format
*Note: The current response keeps Unity-compatible `metadata.points` arrays instead of moving track data under this intended future `track` block.*
```json
{
  "status": "ok",
  "images": {
    "preview_base64": "...",
    "foreground_base64": "...",
    "gameplay_base64": "...",
    "background_base64": "..."
  },
  "track": {
    "points": [ {"x": 0.1, "y": 0.8} ],
    "terrain_type": "road",
    "friction": 0.8
  },
  "metadata": {
    "source_width": 1024,
    "source_height": 768
  }
}
```
**Mismatches to resolve in the future:**
1. Adding a WSL `status` field if direct clients need it.
2. Changing `points` from `[x, y]` arrays to `{"x": x, "y": y}` objects under a `track` block.
3. Adding `terrain_type` and `friction`.
4. Using `source_width`/`source_height` instead of `aspect_ratio`/`timestamp`.

## 5. TODOs
*   **Model Checkpoints:** Clarify if `DepthAnything3.from_pretrained` downloads the model automatically to a cache directory on first run, or if manual placement of weights is required in a disconnected environment. Currently, it assumes the model is downloadable or cached via Hugging Face Hub.
