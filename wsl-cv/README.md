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

`./run_photos.sh --restyle` adds Muse Image art styles to every showcase that is already built, reusing its DA3 / SAM 3.1 / track outputs (only the Muse Image calls run; cached styles are kept unless `--force`).

### Muse Image Art Styles
After the layers and the track exist, `scripts/muse_image.py` asks Muse Image (`muse-image-1.0`, Responses API, $0.01/image) to repaint the photo in each style (`comic`, `cyberpunk`, `snow`, `pixel`, `watercolor`; the prompts are in `STYLES`). The styles run in parallel, and each takes 15–45 s. Every prompt also tells the model to keep the composition (same camera, objects, outlines).

Because the masks are reused, the composition is measured instead of assumed:
1. `size: "auto"` keeps the input's aspect ratio. The tool only accepts `1024x1024`, `1536x1024`, `1024x1536` or `auto`; arbitrary `WxH` values are rejected with HTTP 400.
2. The result is resized to the photo and registered with SIFT + RANSAC (a similarity transform). The transform is applied only when it is plausible and improves the fit.
3. The fit is scored where it matters: points are sampled along the outlines the foreground/gameplay layers were cut on. At each point a 64 px gradient patch of the photo is phase-correlated against the styled image to get the local shift (`alignment.outline_offset`: `median_px`, `within_3px`). Across the 8 sample levels × 5 styles, the median drift is 0.5–2.5 px, with 54–99 % of points within 3 px. The registration scales stay within 1–4 % of 1.0. A plain edge F1 (`edge_f1_raw` / `edge_f1`) is also stored, but it penalizes new texture (bricks, rain, neon), so read it as a style-change score, not an alignment score. As a sanity check, shifting a restyle by 8 px moves the median to about 5 px and drops it to about 35 % within 3 px.
4. The layers are re-cut from the styled pixels with the exact alphas of `merged_layer_01/02.png`. The background is filled in by a second Muse Image turn on the styled image (see Background fill below). All styles therefore share one foreground / gameplay / background split, one track and one set of occluders.

Outputs: `<send2unity_dir>/styles/<name>/{styled_raw,styled,fill_raw,merged_layer_01..03}.png` and `<send2unity_dir>/styles.json` (alignment, timing, token usage, a per-style `track_color` and `background` source). Server errors and timeouts are retried (`MUSE_IMAGE_RETRIES`, default 2). A style that still fails, for example one blocked by Muse Image's content filter, is left out, and the run keeps the others.

**Background fill:** `cut_img.py` fills the background layer with `cv2.inpaint`, which smears across large holes. Muse Image replaces it: everything the foreground and gameplay layers cover (grown by a few pixels) is painted solid magenta, and Muse Image is asked to fill the magenta by continuing whatever surrounds it in the same place, adding no new objects. Most of the frame can be magenta, so the prompt also carries Muse Spark's one-sentence `setting` of the photo (e.g. "a bright indoor room with grey and yellow foam floor tiles and mint-green walls"), written in the same call that proposes the SAM 3.1 concepts and stored in `*_sam31_objects.json`; older runs get one from a separate Muse Spark call (`--objects`). The prompt names no landscape features: an earlier version that listed mountains and vegetation turned desks and rooms into forests. The result is registered like a restyle and only the filled pixels are used; the rest keeps the original pixels. Because the hidden objects are erased from the input, they cannot come back as ghosts, including objects SAM 3.1 never named. Muse Image ignores `input_image_mask` (tested: accepted, no effect), hence the key color. Each style is filled from its own image, and its fill prompt spells out that style's look and puts it above the setting's lighting (the setting is written from the photo, so "a bright lawn under a blue sky" otherwise turned the fill of a cyberpunk night or a snow scene back into a sunny photo when most of the frame was magenta), and the photo gets one too (`<send2unity_dir>/background/`, `background.json`). Fills are cached per image and holes. A hole keeps its object's outline, which invited the model to paint the object back (a hydrant-shaped hole became a hydrant), so the prompt also names the removed objects (the front-layer SAM 3.1 objects that Muse Spark did not list as `fixed`, i.e. not ground, walls or buildings) and asks for background only; an object can still come back now and then. Two checks run on every fill: no more than 2 % of the area may stay magenta, and the colors just outside the holes, which the fill was told to keep, may not shift by more than 20 (Lab, `MUSE_IMAGE_FILL_MAX_DRIFT`; good fills stay under ~13, while fills that redrew the picture in another look, such as a cyberpunk scene coming back as a daylight photo, measured 27–79). A failed fill is made again (`MUSE_IMAGE_FILL_ATTEMPTS`, default 2 calls) and then falls back to `cv2.inpaint`; `MUSE_IMAGE_FILL=0` turns fills off.

```bash
python scripts/muse_image.py <image> <send2unity_dir> [--styles comic,snow|all|none] [--force] [--no-photo-background] [--objects <run>_sam31_objects.json]
```
`pipeline.py` runs every style by default (`--styles` or `MUSE_IMAGE_STYLES` narrows it; `none` skips it). The `/infer` API restyles only when asked: send the form field `style=<name>`, or set `MUSE_IMAGE_STYLE=<name>` on the server. The returned layers are then the styled ones, `metadata.track_color` follows the art, and `metadata.art_style` records the style. The restyle call starts as soon as the photo arrives, alongside DA3 / SAM 3.1 / Muse Spark, and the background fill follows it (for an unstyled level it runs alongside track extraction). In tests, with Muse Spark verifying on the first attempt, a styled level took ~60–75 s and an unstyled one ~45–60 s. If Muse Image fails, the photo's layers are returned.

### 3D Pipeline Showcase
Every `scripts/pipeline.py` run also writes `<send2unity_dir>/showcase.html`: a self-contained three.js page that replays the run (photo → DA3 depth relief → Muse Spark concepts → SAM 3.1 objects → 2.5D layers → Muse Image restyles → Muse Spark design/verify loop → ride). Open it in a browser (it loads three.js from a CDN). Controls: space play/pause, R restart, 1–8 jump to a step, drag to orbit, C back to the cinematic camera, H hide the UI for screen recording, M sound on/off.

**Skater:** in the Ride step a rider stands on the board: a sideways stance facing the camera, built from capsules and posed each frame with 2-bone IK (two kick pushes at the start, knee bob and arm sway while cruising, a crouch-pop over jump spans), with a lime outline so it reads on any art style. It is drawn between the gameplay and foreground layers like the board, so foreground objects hide it.

**Art style switcher:** the bar at the top (Photo · Comic · Cyberpunk · …) swaps the art on the depth relief and on all three layers at any time, with a wipe transition. `S` cycles through the styles. During the Restyle step the timeline tours every style, then keeps one for the track and ride steps: the first style by default, or the one in `?style=<name>` (`?style=original` keeps the photo). Picking a style overrides the tour until `R`. The gallery (`showcase/index.html`) links each level once per style.

**Soundtrack:** the page synthesizes its own music and sound effects with the Web Audio API, so there are no audio files and no third-party audio licenses. A lo-fi beat (96 BPM, Fmaj7 · Em7 · Dm7 · Cmaj9) builds up step by step: pad first, then drums and bass, a muffled beat during the Track step, and a drop on the first downbeat of the Ride. The sound effects are timed from the level's own data: a shutter for the photo, a rumble as the depth relief rises, one note per Muse Spark concept and per SAM 3.1 object, a swoosh along each restyle wipe, a buzz for each rejected Muse Spark attempt and a chime for the verified one, then rolling wheels, ollie pops, landings and grind scrapes, muffled while the skater is behind a foreground object. The whole score is rendered once with an `OfflineAudioContext` (~1–5 s) and then plays locked to the timeline (scrubbing, jumping to a step and pausing all stay in sync). Browsers only start audio after a click or a key press; the speaker button or `M` toggles it.

To render the same sequence to MP4 locally (headless Chromium + ffmpeg, ~1 s per 1080p frame). The MP4 gets the same soundtrack as an AAC track; `--no-audio` leaves it out:
```bash
pip install playwright && python -m playwright install chromium   # once
python scripts/render_showcase.py <send2unity_dir>/showcase.html --out showcase.mp4 [--style cyberpunk] [--no-audio]
```

## 2. Repository Map

*   `scripts/infer.py`: Active runtime API entrypoint (FastAPI server). It creates one isolated `_jobs/` directory per request and packages the Unity-facing JSON response.
*   `scripts/pipeline.py`: Standalone pipeline execution script.
*   `scripts/run_inference_depth_sam31.py`: DA3 depth generation plus SAM 3.1 segmentation; also writes `{base}_sam31_objects.json` (label, box, score, label id per object).
*   `scripts/sam31_segmentation.py`: Muse Spark first proposes this photo's concepts to segment (skateable + scenery); then SAM 3.1 API calls (one noun-phrase prompt per request), mask decoding, and non-overlapping label-map export. Falls back to a fixed concept list if the proposal fails.
*   `scripts/muse_spark.py`: Shared Muse Spark client (Responses API, JSON-schema structured output). Model defaults to `muse-spark-1.3-contributor`; override with `MUSE_SPARK_MODEL_ID`.
*   `scripts/cut_img.py`: Depth-sorted visual layer generation (foreground, gameplay, background) plus background inpainting (replaced by Muse Image fills in `muse_image.py`).
*   `scripts/level_designer.py`: Muse Spark picks which SAM 3.1 object(s) are the riding surface (verified against the masks: must span the scene at ground level; violations are fed back for repair, up to `MUSE_SPARK_MAX_ATTEMPTS`, default 3). Also continues the track behind objects standing on the ground, finds the occluders the skater passes behind (closer than the track in DA3 depth), and has Muse Spark write the mission. Adds a `level` field to `track_points.json` (mission, surfaces, occluders, and `render_order`: background → gameplay → track/skater → foreground, matching Unity's Background / Midground / Foreground sorting layers) and fills Unity's existing `occluded_segments` (point-index ranges the foreground draws over). Writes `level_vis.png` / `level_design_log.json`. `cut_img.py` also writes `layers.json` (which SAM region went into which layer image).
*   `scripts/extract_track.py`: The original track extraction: the smoothed top edge of the track region (`layer_00_mask.npy`).
*   `scripts/muse_image.py`: Muse Image restyling. Repaints the photo per art style, registers it back onto the photo and scores the fit, then re-cuts the three layers with the original alphas (`styles/`, `styles.json`).
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
