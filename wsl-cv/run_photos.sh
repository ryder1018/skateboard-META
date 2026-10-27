#!/usr/bin/env bash
# Build a showcase for every photo dropped into photos/, then open the gallery.
#
#   ./run_photos.sh            # only photos without a showcase yet
#   ./run_photos.sh --force    # rebuild all of them
#   ./run_photos.sh --video    # also render an MP4 for each processed photo
#
# Photos stay where they are; each one gets showcase/<name>/ (name = file name
# reduced to [A-Za-z0-9_-]) and showcase/index.html links them all.
# Run inside the project's Python env (e.g. `conda activate da3_skate`).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PHOTOS="$HERE/photos"
PYTHON="${PYTHON:-python}"
export PYTHONNOUSERSITE=1
FORCE=0
VIDEO_ARG=""
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --video) VIDEO_ARG="--video" ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p "$PHOTOS"
done_count=0
failed=()
shopt -s nullglob nocaseglob
for photo in "$PHOTOS"/*.{jpg,jpeg,png,webp,bmp}; do
  base="$(basename "${photo%.*}")"
  name="$(printf '%s' "$base" | tr -cs 'A-Za-z0-9_-' '_' | sed 's/^_*//; s/_*$//')"
  if [ "$FORCE" -eq 0 ] && [ -f "$HERE/showcase/$name/showcase.html" ]; then
    echo "skip  $name (already built; --force to rebuild)"
    continue
  fi
  echo "build $name  <-  $(basename "$photo")"
  mkdir -p "$HERE/showcase/$name"
  log="$HERE/showcase/$name/run.log"
  if NO_OPEN=1 PYTHON="$PYTHON" "$HERE/try_photo.sh" "$photo" "$name" $VIDEO_ARG > "$log" 2>&1 \
      && grep -q "Showcase saved" "$log"; then
    grep -E "Level saved" "$log" | sed 's/^ */      /'
    done_count=$((done_count + 1))
  else
    echo "      FAILED, see showcase/$name/run.log"
    failed+=("$name")
  fi
done

"$PYTHON" "$HERE/scripts/showcase_index.py" "$HERE/showcase"
echo "Built $done_count new showcase(s)${failed:+, failed: ${failed[*]}}."
if [ "${NO_OPEN:-0}" != "1" ] && command -v explorer.exe >/dev/null 2>&1; then
  explorer.exe "$(wslpath -w "$HERE/showcase/index.html")" || true
fi
