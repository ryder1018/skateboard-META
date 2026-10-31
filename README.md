# SkateboardP Monorepo - AI Context Document

> **For AI Assistants:** This README is designed to give you a rapid, high-level understanding of the architecture, data flow, and file layout of this monorepo. Please read this before modifying cross-boundary code.

## 1. Project Overview
This project implements a **photo-to-playable-level pipeline** for a Unity skateboard game. 
- **Input:** A real-world 2D image captured by a mobile device.
- **Processing:** Depth extraction, semantic segmentation, occlusion inpainting, track path generation, and optional Muse Image art restyling (comic, cyberpunk, snow, pixel, watercolor) that reuses the same layer masks.
- **Output:** A JSON payload containing Base64 encoded layered images (foreground/gameplay/background) and 2D track points that Unity procedurally generates into a 3D playable level.

---

## 2. Monorepo Architecture

This repository contains multiple interconnected environments (C# & Python) running across different operating systems (Windows & Linux/WSL).

### `wsl-cv/` (WSL / Linux - Python)
**Role:** The core Computer Vision backend. 
- **Tech Stack:** Python 3.10+, PyTorch, FastAPI, `uv` for dependency management.
- **Key Flow:** Exposes a `POST /infer` endpoint. Uses `Depth Anything 3` (in `models/depth_anything_3`) to extract depth, uses SAM 3.1 (Meta Model API) text prompts to find skateable objects, then runs layer export and track extraction scripts (`scripts/run_inference_sobal.py`, `scripts/cut_img.py`, `scripts/extract_track.py`) in isolated subprocess sandbox directories (`_jobs/`).
- **Important:** Do NOT modify `models/depth_anything_3` as it is vendor code. Look at `wsl-cv/README.md` for specific CV details.

### `win-server/` (Windows - Python)
**Role:** The Gateway Server.
- **Tech Stack:** Python (Likely Flask/FastAPI).
- **Purpose:** Exposes a public endpoint for the mobile Unity client. Since the WSL backend cannot easily expose ports to the external network, this Windows server receives the request, routes it via `localhost`/internal call to the WSL API, and proxies the JSON payload back to the client.

### `unity_client/` (Windows/Mobile - C#)
**Role:** The Game Client Connection & Generation Scripts.
- **Tech Stack:** Unity (C#).
- **Purpose:** Contains **only** the connection and level generation scripts extracted from the main Unity project. It is *not* the entire Unity game repository. `MobileCameraCapture.cs` handles capturing photos, sending them to `win-server`, and decoding the returned Base64/JSON. It then hands off the payload to other generation scripts (like `LevelCoordinator`) to actually instantiate the physical gameplay objects.

---

## 3. Data Transmission Pipeline

When modifying API contracts, note that data traverses three distinct boundaries. The active Unity client is
`unity_client/MobileCameraCapture.cs`.

`MobileCameraCapture.cs` supports taking one camera photo or selecting one or more gallery images. For multiple
gallery images, Unity does **not** send one multipart request containing many files. It processes the selected
paths one at a time: resize/encode the image, upload it as the `photo` field, poll for that image's result, enqueue
the returned level payload, then continue with the next image.

```mermaid
sequenceDiagram
    participant Mobile as MobileCameraCapture (Unity)
    participant Win as Windows Server (Public API)
    participant WSL as WSL Model Service (FastAPI)
    participant Level as LevelCoordinator

    Note over Mobile: User captures one photo or selects one/many gallery images
    loop For each selected image
        Mobile->>Mobile: Load image, resize longest side, encode JPEG
        Mobile->>Win: POST /upload (multipart/form-data: `photo`, X-API-KEY)
        Win-->>Mobile: {"status":"accepted"}
        Win->>WSL: Internal POST /infer (multipart/form-data: `photo`)

        Note over WSL: 1. DA3 Depth + SAM 3.1 Masks<br/>2. Depth-Sorted Layer Inpainting<br/>3. Track Extraction<br/>4. (optional) Muse Image restyle

        WSL-->>Win: JSON (Base64 layers + track metadata)
        Win->>Win: Store latest job state/result
        loop Until success/error/timeout
            Mobile->>Win: GET /download (X-API-KEY)
            Win-->>Mobile: {"status":"processing"} or final JSON
        end
        Mobile->>Mobile: Decode Base64 textures + save metadata backup
        Mobile->>Level: EnqueueLevel(LevelPayload)
    end
```

Important runtime details:
- Unity still uses the field name `photo` for every upload. Batch selection is a client-side loop, not a backend
  batch endpoint.
- `win-server/server.py` keeps a single global job state (`idle`, `processing`, `success`, `error`). The current
  Unity script avoids overlapping uploads by processing images sequentially.
- `MobileCameraCapture.cs` calls `LevelCoordinator.EnqueueLevel(payload)` for each successful response. The old
  reference script called `ApplyPayload(payload)` and handled only one image at a time.
- Metadata is backed up to `Application.persistentDataPath/schema_XX.json` for each batch item, and also to
  `schema.json` for compatibility with the older single-result behavior.
- Art style: when the WSL server runs with `MUSE_IMAGE_STYLE=<comic|cyberpunk|snow|pixel|watercolor>` (or a request
  sends the form field `style`), the three layer images are Muse Image restyles cut with the same masks, and
  `metadata.art_style` / `metadata.track_color` describe them. The response shape is unchanged.
- Unity map style picker: `ArtStyleSelector.cs` adds a "MAP STYLE" row (Original, Comic, Cyberpunk, Snow, Pixel,
  Watercolor) under the upload panel. `MobileCameraCapture` sends the choice as the `style` field on `/upload`;
  `win-server` forwards it to `/infer` ("original" = the photo's own pixels). The result panel shows the style and
  `metadata.level.mission`.
- Timing: one photo takes about 45-150 s (SAM 3.1 + Muse Spark retries + Muse Image), so Unity polls by elapsed time
  (`processingTimeoutSeconds`, default 320 s) instead of a fixed attempt count.
- `LocalLevelTester.applyOnStart` only auto-loads its baked levels in the Editor; device builds start at the
  photo capture flow unless `applyOnStartInBuilds` is ticked.

### JSON Payload Contract (WSL -> Win -> Unity)
The CV backend returns the following structure:
```json
{
  "images": {
    "preview_base64": "<base64_string>",
    "foreground_base64": "<base64_string>",
    "gameplay_base64": "<base64_string>",
    "background_base64": "<base64_string>"
  },
  "metadata": {
    "points": [ [x1, y1], [x2, y2] ],
    "aspect_ratio": 1.7778,
    "timestamp": "..."
  }
}
```
*(Note: Refer to `wsl-cv/api_contract.md` for details on future payload refactoring goals).*

---

## AI Guidelines for this Repo
1. **Cross-Platform Awareness:** Assume path operations in `wsl-cv/` use Linux conventions, while `win-server/` uses Windows. 
2. **Subprocess Orchestration:** The `wsl-cv/scripts/infer.py` orchestrates the pipeline using string manipulation and `subprocess.run`. Be extremely careful when changing directory structures as it breaks hardcoded paths.
3. **Dependency Management:** Use `uv` for Python dependency management in the `wsl-cv` project.
