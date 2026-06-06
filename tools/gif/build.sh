#!/usr/bin/env bash
# Regenerate the README GIFs from the real pipeline.
#   ./build.sh imagination   # GPU-free
#   ./build.sh nav2          # GPU-free
#   ./build.sh ijepa         # needs a GPU + I-JEPA weights
#   ./build.sh all
#
# Prereqs: ROS 2 sourced, `npm install playwright-core` here, system Chrome,
# ffmpeg, and (for ijepa) torch + transformers. Output GIFs land in ../../docs.
set -euo pipefail
cd "$(dirname "$0")"

HERE="$(pwd)"
DOCS="$(cd ../../docs && pwd)"
FPS_imagination=12 ; FRAMES_imagination=32
FPS_nav2=11        ; FRAMES_nav2=43
FPS_ijepa=6        ; FRAMES_ijepa=18
OUT_imagination=imagination.gif
OUT_nav2=nav2_scoring.gif
OUT_ijepa=ijepa_surprise.gif

build_one() {
  local kind="$1" work; work="$(mktemp -d)"
  local fps="FPS_${kind}" frames="FRAMES_${kind}" out="OUT_${kind}"
  echo ">> $kind"
  python3 gen_data.py "$kind" --out "$work/data.json"
  node capture.js "render_${kind}.html" "$work/data.json" "$work/frames" "${!frames}"
  ffmpeg -y -framerate "${!fps}" -i "$work/frames/f%03d.png" \
    -vf "scale=720:-1:flags=lanczos,palettegen=stats_mode=full" "$work/palette.png" -loglevel error
  ffmpeg -y -framerate "${!fps}" -i "$work/frames/f%03d.png" -i "$work/palette.png" \
    -lavfi "scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
    -loop 0 "$DOCS/${!out}" -loglevel error
  echo "   -> $DOCS/${!out}"
  rm -rf "$work"
}

case "${1:-all}" in
  all) build_one imagination; build_one nav2; build_one ijepa ;;
  imagination|nav2|ijepa) build_one "$1" ;;
  *) echo "usage: $0 [imagination|nav2|ijepa|all]"; exit 2 ;;
esac
