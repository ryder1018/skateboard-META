# WSL CV Backend Documentation

This module handles the computer vision pipeline for the SkateP project. It receives images, extracts depth maps using Depth Anything 3, generates object masks with SAM2 automatic mask generation, segments them into visual layers, and generates track points for Unity.

## 1. Environment Setup & Execution

We use `uv` for dependency management.

### Installation
```bash
# Inside the wsl-cv directory
uv sync
```

`uv sync` installs the SAM2 package from `facebookresearch/sam2`. The first inference may also download the configured SAM2 Hugging Face model, `facebook/sam2.1-hiera-small` by default.

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

## 2. Repository Map

*   `scripts/infer.py`: Active runtime API entrypoint (FastAPI server). It creates one isolated `_jobs/` directory per request and packages the Unity-facing JSON response.
*   `scripts/pipeline.py`: Standalone pipeline execution script.
*   `scripts/run_inference_depth_sam2.py`: DA3 depth generation plus SAM2 mask generation.
*   `scripts/sam2_segmentation.py`: SAM2 automatic masks, mask filtering, and non-overlapping label-map export.
*   `scripts/cut_img.py`: Depth-sorted visual layer generation (foreground, gameplay, background) plus background inpainting.
*   `scripts/extract_track.py`: Smooth playable track point extraction from the dedicated track/ground-region mask.
*   `scripts/depth_adapter.py`: Wrapper and integration point for the DA3 model.
*   `models/depth_anything_3/`: Vendor/model code (Do not modify).
*   `Send2Unity/`: Generated Unity-facing outputs for manual runs.
*   `outputs/`: Generated intermediate outputs.

## 3. Pipeline Flow & Responsibilities

### Execution Flow
1. Windows Server sends `POST /infer` with an image.
2. `scripts/infer.py` creates an isolated job environment in `_jobs/` with the uploaded image, symlinked scripts/models, and wrapper scripts for path-sensitive steps.
3. Step 1 runs `scripts/run_inference_depth_sam2.py` in process to generate a 16-bit DA3 depth map and a SAM2 label map.
4. Step 2 runs `scripts/cut_img.py` through a wrapper to sort accepted regions by average depth, export foreground/gameplay/background layers, and perform occlusion inpainting.
5. Step 3 runs `scripts/extract_track.py` through a wrapper to extract smooth playable track points from `layer_00_mask.npy`, the dedicated track/ground-region mask selected during Step 2.
6. `scripts/infer.py` packages everything into Base64 JSON and responds.

### Known Fragile Points
*   **Job Orchestration**: `scripts/infer.py` uses isolated job directories, symlinks, and wrapper scripts to override hardcoded paths in path-sensitive legacy scripts. This is sensitive to script changes.
*   **DA3 Vendor Dependency**: `models/depth_anything_3/` must remain untouched. Any API changes there require updates to `scripts/depth_adapter.py`.
*   **SAM2 Runtime Dependency**: SAM2 is installed from GitHub and loads model weights from Hugging Face by default. Offline machines need the model cached ahead of time or `SAM2_MODEL_ID` pointed at an accessible local/HF model.
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
