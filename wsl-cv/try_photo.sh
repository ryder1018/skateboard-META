#!/usr/bin/env bash
# Try one photo end to end and open its 3D showcase.
#
#   ./try_photo.sh <photo> [name] [--video]
#
# Copies the photo (pipeline.py resizes its input in place), runs the full
# pipeline into showcase/<name>/, and opens showcase/<name>/showcase.html in
# the Windows browser (unless NO_OPEN=1). --video also renders
# showcase/<name>/showcase.mp4. Names are reduced to [A-Za-z0-9_-].
# Run inside the project's Python env (e.g. `conda activate da3_skate`).
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <photo> [name] [--video]" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
PHOTO="$1"; shift
NAME=""
VIDEO=0
for arg in "$@"; do
  case "$arg" in
    --video) VIDEO=1 ;;
    *) NAME="$arg" ;;
  esac
done
[ -f "$PHOTO" ] || { echo "photo not found: $PHOTO" >&2; exit 1; }
EXT="${PHOTO##*.}"
NAME="${NAME:-$(basename "${PHOTO%.*}")}"
NAME="$(printf '%s' "$NAME" | tr -cs 'A-Za-z0-9_-' '_' | sed 's/^_*//; s/_*$//')"
OUT="$HERE/showcase/$NAME"
PYTHON="${PYTHON:-python}"
# Ignore ~/.local packages (an old torch there would shadow the env's).
export PYTHONNOUSERSITE=1

# MODEL_API_KEY for SAM 3.1 / Muse Spark lives in the repo root .env.
if [ -z "${MODEL_API_KEY:-}" ] && [ -f "$HERE/../.env" ]; then
  set -a; . "$HERE/../.env"; set +a
fi

mkdir -p "$OUT"
cp "$PHOTO" "$OUT/$NAME.$EXT"
cd "$HERE"
"$PYTHON" scripts/pipeline.py "$OUT/$NAME.$EXT" --out_dir "$OUT/outputs" --send2unity_dir "$OUT"

if [ "$VIDEO" -eq 1 ]; then
  "$PYTHON" scripts/render_showcase.py "$OUT/showcase.html" --out "$OUT/showcase.mp4"
fi

echo "Showcase: $OUT/showcase.html"
if [ "${NO_OPEN:-0}" != "1" ] && command -v explorer.exe >/dev/null 2>&1; then
  explorer.exe "$(wslpath -w "$OUT/showcase.html")" || true
fi
