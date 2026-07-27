#!/usr/bin/env bash
set -euo pipefail

# Builds a hand-only synthetic motion clip from Apache-2.0 MediaPipe gesture
# recognizer test images. The source photos are downloaded into a temporary
# directory and deleted; only face-free hand crops are encoded into the output.

fixture_output="${HAWKEYE_FINGER_SMOKE_VIDEO:-/tmp/hawkeye_smoke/deidentified_hand_motion.mp4}"
fixture_tmp_dir="$(mktemp -d)"

cleanup_fixture_tmp() {
  find "$fixture_tmp_dir" -type f -delete
  rmdir "$fixture_tmp_dir"
}
trap cleanup_fixture_tmp EXIT

command -v curl >/dev/null
command -v ffmpeg >/dev/null

mkdir -p "$(dirname "$fixture_output")"

curl -fsSL \
  "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/model_maker/python/vision/gesture_recognizer/testdata/raw_data/four/06aa70cc-a12a-4b1e-85cf-e54d44c19a3a.jpg" \
  -o "$fixture_tmp_dir/open-source.jpg"
curl -fsSL \
  "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/model_maker/python/vision/gesture_recognizer/testdata/raw_data/rock/026fd791-8f64-4fae-8cb0-0e01dc4362ce.jpg" \
  -o "$fixture_tmp_dir/closed-source.jpg"

ffmpeg -hide_banner -loglevel error -y \
  -i "$fixture_tmp_dir/open-source.jpg" \
  -vf "crop=135:190:245:115,scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2:color=white" \
  "$fixture_tmp_dir/open-hand.png"
ffmpeg -hide_banner -loglevel error -y \
  -i "$fixture_tmp_dir/closed-source.jpg" \
  -vf "crop=155:240:0:30,scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2:color=white" \
  "$fixture_tmp_dir/closed-hand.png"

ffmpeg -hide_banner -loglevel error -y \
  -loop 1 -framerate 30 -i "$fixture_tmp_dir/open-hand.png" \
  -loop 1 -framerate 30 -i "$fixture_tmp_dir/closed-hand.png" \
  -filter_complex "[0:v][1:v]blend=all_expr='A*(1-abs(sin(PI*T)))+B*abs(sin(PI*T))':shortest=1" \
  -t 8 -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  "$fixture_output"

echo "$fixture_output"
