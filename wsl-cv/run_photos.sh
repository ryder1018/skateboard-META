#!/usr/bin/env bash
# Build a showcase for every photo dropped into photos/, then open the gallery.
#
#   ./run_photos.sh            # only photos without a showcase yet
#   ./run_photos.sh --force    # rebuild all of them
#   ./run_photos.sh --video    # also render an MP4 for each processed photo
#   ./run_photos.sh --restyle  # (re)generate Muse Image art styles and background fills for every built
#                              # showcase, reusing its DA3/SAM/track outputs;
#                              # cached styles are kept unless --force is given.
#                              # MUSE_IMAGE_STYLES=comic,snow picks styles.
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
RESTYLE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --video) VIDEO_ARG="--video" ;;
    --restyle) RESTYLE=1 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p "$PHOTOS"
done_count=0
failed=()
shopt -s nullglob nocaseglob

if [ "$RESTYLE" -eq 1 ]; then
  if [ -z "${MODEL_API_KEY:-}" ] && [ -f "$HERE/../.env" ]; then
    set -a; . "$HERE/../.env"; set +a
  fi
  FORCE_ARG=""
  [ "$FORCE" -eq 1 ] && FORCE_ARG="--force"
  for dir in "$HERE"/showcase/*/; do
    name="$(basename "$dir")"
    photo=""
    for candidate in "$dir$name".{jpg,jpeg,png,webp,bmp}; do photo="$candidate"; break; done
    if [ -z "$photo" ] || [ ! -d "${dir}outputs" ] || [ ! -f "${dir}merged_layer_01.png" ]; then
      echo "skip  $name (no pipeline outputs to restyle)"
      continue
    fi
    echo "style $name"
    if (cd "$HERE" && "$PYTHON" scripts/muse_image.py "$photo" "$dir" --styles "${MUSE_IMAGE_STYLES:-all}" $FORCE_ARG \
        --objects "${dir}outputs/${name}_sam31_objects.json" \
        && "$PYTHON" scripts/showcase.py "$photo" "${dir}outputs" "$dir") > "${dir}restyle.log" 2>&1; then
      grep -E "✔|✘" "${dir}restyle.log" | tail -n 6 | sed 's/^ */      /'
      done_count=$((done_count + 1))
    else
      echo "      FAILED, see showcase/$name/restyle.log"
      failed+=("$name")
    fi
  done
  "$PYTHON" "$HERE/scripts/showcase_index.py" "$HERE/showcase"
  echo "Restyled $done_count showcase(s)${failed:+, failed: ${failed[*]}}."
  exit 0
fi

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
