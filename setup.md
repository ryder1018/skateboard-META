# Set Up Whole Project

## Start the WSL CV API

Run this in WSL. Run `uv sync` after dependency changes or on a fresh machine; it installs DA3 dependencies, FastAPI upload support, and SAM2.

```bash
cd /home/mike/Projects/SkateP/SP25/wsl-cv
uv sync
uv run python -m uvicorn scripts.infer:app --host 0.0.0.0 --port 9000
```

The first SAM2 inference may download model weights for `facebook/sam2.1-hiera-large`.

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
cd C:\path\to\SP25\win-server
.\venv\Scripts\Activate.ps1
python server.py
```

The gateway listens on port `5000`. Unity sends requests to this server, not directly to WSL.

## Test the Unity-Style Upload Flow

Run these in PowerShell after both servers are running.

### 1. Upload one photo

Replace the photo path with your real image path:

```powershell
curl.exe -X POST "http://127.0.0.1:5000/upload" `
  -H "X-API-KEY: 112550191" `
  -F "photo=@C:\2026_Spring\sp25test\scenery.jpg;type=image/jpeg"
```

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

## Optional: Test WSL Directly

This bypasses `win-server`, so it is not the exact Unity flow. Use it only to check whether `wsl-cv/scripts/infer.py` works by itself.

```powershell
curl.exe -X POST "http://127.0.0.1:9000/infer" `
  -F "photo=@C:\2026_Spring\sp25test\scenery.jpg;type=image/jpeg" `
  -o result.json
```
