#!/usr/bin/env bash
set -euo pipefail

FRONTEND_URL="${HAWKEYE_FRONTEND_URL:-https://hawkeye-labeling-tool.vercel.app}"
SMOKE_VIDEO="${HAWKEYE_SMOKE_VIDEO:-/tmp/hawkeye_smoke/gait_smoke_6s.mp4}"
POLL_SECONDS="${HAWKEYE_POLL_SECONDS:-120}"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/hawkeye-smoke.XXXXXX")"

echo "Frontend: $FRONTEND_URL"
echo "Artifacts: $tmp_dir"

curl -fsS "$FRONTEND_URL/" >/dev/null
echo "OK home"

curl -fsS "$FRONTEND_URL/api/vlm/status" -o "$tmp_dir/vlm.json"
node - "$tmp_dir/vlm.json" <<'NODE'
const fs = require("fs")
const data = JSON.parse(fs.readFileSync(process.argv[2], "utf8"))
if (!data.success || data.available !== true) {
  throw new Error(`VLM unavailable: ${JSON.stringify(data)}`)
}
console.log(`OK vlm ${data.model || ""}`.trim())
NODE

if [[ ! -f "$SMOKE_VIDEO" ]]; then
  echo "Missing smoke video: $SMOKE_VIDEO" >&2
  echo "Set HAWKEYE_SMOKE_VIDEO=/path/to/video.mp4 or create /tmp/hawkeye_smoke/gait_smoke_6s.mp4" >&2
  exit 2
fi

curl -fsS -X POST "$FRONTEND_URL/api/analyze" \
  -F "video_file=@${SMOKE_VIDEO};type=video/mp4" \
  -F "patient_id=production_smoke" \
  -F "test_type=gait" \
  -F "scoring_method=rule" \
  -o "$tmp_dir/analyze.json"

analysis_id="$(node - "$tmp_dir/analyze.json" <<'NODE'
const fs = require("fs")
const data = JSON.parse(fs.readFileSync(process.argv[2], "utf8"))
if (!data.success || !data.id) {
  throw new Error(`Analyze did not start: ${JSON.stringify(data)}`)
}
console.log(data.id)
NODE
)"
echo "OK analyze started $analysis_id"

deadline=$((SECONDS + POLL_SECONDS))
analysis_status=""
while (( SECONDS < deadline )); do
  curl -fsS "$FRONTEND_URL/api/analysis/progress/$analysis_id" -o "$tmp_dir/progress.json"
  analysis_status="$(node - "$tmp_dir/progress.json" <<'NODE'
const fs = require("fs")
const data = JSON.parse(fs.readFileSync(process.argv[2], "utf8"))
console.log(data.status || "")
NODE
)"

  if [[ "$analysis_status" == "completed" ]]; then
    break
  fi

  if [[ "$analysis_status" == "error" ]]; then
    cat "$tmp_dir/progress.json" >&2
    exit 3
  fi

  sleep 3
done

if [[ "$analysis_status" != "completed" ]]; then
  echo "Timed out waiting for analysis completion: $analysis_id" >&2
  cat "$tmp_dir/progress.json" >&2
  exit 4
fi
echo "OK analysis completed"

curl -fsS "$FRONTEND_URL/api/analysis/result/$analysis_id" -o "$tmp_dir/result.json"
video_path="$(node - "$tmp_dir/result.json" <<'NODE'
const fs = require("fs")
const data = JSON.parse(fs.readFileSync(process.argv[2], "utf8"))
if (!data.success || data.video_type !== "gait") {
  throw new Error(`Unexpected result: ${JSON.stringify({ success: data.success, video_type: data.video_type })}`)
}
const videoPath = data.skeleton_data && data.skeleton_data.skeleton_video_url
if (!videoPath) {
  throw new Error("Missing skeleton video URL")
}
console.log(videoPath)
NODE
)"
echo "OK result $video_path"

curl -fsS -r 0-99 "$FRONTEND_URL$video_path" -o "$tmp_dir/video-range.bin"
bytes="$(wc -c < "$tmp_dir/video-range.bin" | tr -d ' ')"
if [[ "$bytes" != "100" ]]; then
  echo "Unexpected video range size: $bytes" >&2
  exit 5
fi
echo "OK video range 206-compatible"
echo "Production smoke passed: $analysis_id"
