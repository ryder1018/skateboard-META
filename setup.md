# Set Up Whole Project

## Start the WSL CV API

Run this in WSL. Run `uv sync` after dependency changes or on a fresh machine; it installs DA3 dependencies, FastAPI upload support, and the SAM 3.1 output parser.
On this machine the conda env `da3_skate` already has everything, so `conda activate da3_skate` works instead of `uv sync` / `uv run`.

```bash
cd <repo>/wsl-cv
uv sync
uv run python -m uvicorn scripts.infer:app --host 0.0.0.0 --port 9000
```

SAM 3.1 runs through the Meta Model API. Export `MODEL_API_KEY` in the same shell before starting the server.

Check that Windows can reach the WSL API:

```powershell
curl.exe http://127.0.0.1:9000/health
```

Expected response:

```json
{"status":"ok","timestamp":"..."}
```

If this does not work from Windows, get the WSL IP in WSL:

```bash
hostname -I
```

Then update `win-server/server.py`:

```python
WSL_API_URL = "http://<WSL_IP>:9000/infer"
```

If `127.0.0.1` works, keep the default:

```python
WSL_API_URL = "http://127.0.0.1:9000/infer"
```

## Start the Windows Gateway

Run this in PowerShell from the `win-server` directory:

```powershell
cd C:\path\to\SkateboardP\win-server
.\venv\Scripts\Activate.ps1
python server.py
```

The gateway listens on port `5000`. Unity sends requests to this server, not directly to WSL.
The venv needs `flask` and `requests` (`pip install flask requests`).

For Unity Editor testing on this PC only, the gateway can also run in WSL (the `da3_skate` env has Flask);
Windows reaches it at `127.0.0.1:5000` through WSL's localhost forwarding:

```bash
cd win-server && python server.py
```

An Android phone cannot reach a gateway inside WSL (NAT networking), so for the phone run it on Windows, allow
inbound TCP 5000 in Windows Firewall, and enter the PC's Wi-Fi IPv4 address (`ipconfig`) in the app's Change IP box.

## Test the Unity-Style Upload Flow

Run these in PowerShell after both servers are running.

### 1. Upload one photo

Replace the photo path with your real image path:

```powershell
curl.exe -X POST "http://127.0.0.1:5000/upload" `
  -H "X-API-KEY: 112550191" `
  -F "photo=@C:\2026_Spring\sp25test\scenery.jpg;type=image/jpeg" `
  -F "style=comic"
```

`style` is optional: `original`, `comic`, `cyberpunk`, `snow`, `pixel` or `watercolor` (the Unity map style
picker sends it). Without it the WSL server uses `$MUSE_IMAGE_STYLE`, or the photo's own pixels.

Expected immediate response:

```json
{"status":"accepted"}
```

### 2. Poll for the result

This matches what `MobileCameraCapture.cs` does after upload:

```powershell
curl.exe "http://127.0.0.1:5000/download" `
  -H "X-API-KEY: 112550191" `
  -o result.json
```

If `result.json` contains:

```json
{"status":"processing"}
```

wait a few seconds and run the download command again.

When processing succeeds, `result.json` should contain:

```json
{
  "status": "success",
  "images": {
    "preview_base64": "...",
    "foreground_base64": "...",
    "gameplay_base64": "...",
    "background_base64": "..."
  },
  "metadata": {
    "points": []
  }
}
```

## Unity Editor Smoke Test

With the WSL API and the gateway running and `Parallax.unity` open in the Editor, use
**Tools > SkateP > Pipeline Smoke Test...** and pick a photo. It enters Play mode, uploads the photo through
`MobileCameraCapture` with the style selected in the picker, waits for the result panel, presses Start Game,
checks that the level in play is the generated one, and writes screenshots plus `report.txt` to
`Temp/PipelineSmokeTest/` in the Unity project. It skips `LocalLevelTester`'s baked levels for that run.

It can also be started without touching the Editor by writing `Temp/pipeline_smoke_test.json` in the Unity project:

```json
{"photo": "C:/Users/Public/skatep_test/park.jpg", "style": "cyberpunk", "server": "127.0.0.1"}
```

An empty `Temp/pipeline_smoke_refresh` file makes the Editor re-import scripts edited outside Unity.

## Optional: Test WSL Directly

This bypasses `win-server`, so it is not the exact Unity flow. Use it only to check whether `wsl-cv/scripts/infer.py` works by itself.

```powershell
curl.exe -X POST "http://127.0.0.1:9000/infer" `
  -F "photo=@C:\2026_Spring\sp25test\scenery.jpg;type=image/jpeg" `
  -o result.json
```
