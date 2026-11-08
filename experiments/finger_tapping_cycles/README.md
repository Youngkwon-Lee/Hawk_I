# Finger-tapping cycle analysis (macOS)

`analyze_finger_tapping_cycles.swift` reads one finger-tapping video with AVFoundation and estimates tapping cycles.
Moved from `hawkeye-labeling-tool` on 2026-11-04 (source HEAD `d7ff130`).

```bash
swift experiments/finger_tapping_cycles/analyze_finger_tapping_cycles.swift VIDEO
```

macOS only. Run it on authorized local media; do not commit videos or per-clip outputs.

## Optional image-plane hand trace export (research only)

```bash
swift experiments/finger_tapping_cycles/analyze_finger_tapping_cycles.swift VIDEO \
  --trace-json /path/to/private/trace.json --side right
```

The existing aggregate JSON on stdout remains unchanged. The optional flag writes a **new** local JSON file (will not overwrite an existing file) with `task=finger_tapping`, an **operator-declared, unverified** `side`, `decoded_frames`, and samples containing `time_s` (source video presentation timestamp, seconds), `gap_norm`, and `confidence`. The `unit` value is `thumb_index_gap_over_palm_width`, compatible at the field-name level with the existing `hawkeye-labeling-tool/scripts/analyze_finger_tapping_trace.py` input contract. That separate module remains the owner of trace QC and descriptive metrics; do not copy it here.

`gap_norm` is the 2D image-plane thumb-tip/index-tip separation divided by the 2D index-MCP/little-MCP span, with pixel aspect correction. This is a *hand-span proxy*, **not** calibrated mm/cm or 3D hand kinematics. Frames without sufficiently confident four-point measurements, valid timestamps or a usable palm span are omitted rather than filled with zeros. The export includes explicit `provenance` fields for 2D coordinates, the normalization reference, unverified hand identity and operator-declared side.

**Important quality boundary:** `VNDetectHumanHandPoseRequest` returns one detected hand per frame, but does not verify continuous hand identity or the declared left/right side; camera mirroring and hand swapping have not been resolved. Do not treat syntactically accepted trace data or a downstream QC `pass` as a verified per-side measurement. Verify the recording, tracking and side externally before research interpretation. The downstream QC will hold on missing detections, discontinuous timestamps or too few cycles, and will **not** produce an MDS-UPDRS 3.4 score. It currently reports median frequency, interval CV, and early/late opening peaks/decrement, not all five proposed features (notably per-tap excursion and pause events are absent).

No video-to-3D lifting, clinical scoring, patient-data access, real-video regression or model validation is implemented by this export.

### Synthetic-only verification (2026-11-08)

```bash
swift experiments/finger_tapping_cycles/analyze_finger_tapping_cycles.swift --geometry-selftest
```

The self-test exercises **the same pure image-plane normalization function used by the video exporter** without any camera or data access. It checks known thumb–index/palm-span ratios, pixel aspect correction, uniform-resolution changes, mirror-invariant *distance* (not handedness), zero palm span, low landmark confidence, nonfinite timestamps, out-of-frame locations and impossible gap ratios. On macOS Apple Swift 6.4.2: `xcrun swiftc -typecheck` PASS (one pre-existing deprecated AVFoundation API warning); `--geometry-selftest` PASS.

The existing **schematic synthetic** `SYNTH_FT_001.mp4` was decoded with the optional export path: **240 frames decoded, 0 Vision hand detections, 0 output samples, `insufficient_hand_detection`**. This confirms fail-closed empty-trace output, **not** successful hand tracking or real-video measurement. It must not be used as a validated positive-detection fixture.

### Per-frame hand side diagnostics (synthetic-only, 2026-11-08)

The optional JSON trace now also carries `frame_index` (zero-based decoded source index), `palm_span_px` (2D **image-plane** pixel denominator, not hand length in cm), and `vision_chirality` (Vision-reported `left`, `right`, or `unknown`) on each accepted sample. A `vision_chirality_audit` counts reported left/right/unknown values, reports disagreement with the **unverified operator-declared** side, and distinguishes mixed reports from absent estimates. These diagnostics **never mark a subject side or tracking identity verified**. The trace also records `image_orientation_assumption: up_unverified` and `camera_mirroring_verified: false`, as the exporter does not yet inspect camera-mirror/orientation provenance.

A repeatable, purely synthetic `--geometry-selftest` on Mac exercises matched, opposite, mixed and unknown chirality reports (all stay `side_verified=false`), alongside normalized-distance geometry checks. Apple Vision's `VNHumanHandPoseObservation.chirality` is only an algorithmic estimate; mixed chirality reports can signal ambiguity but cannot prove a hand swap. A consistent reported side is likewise not verification of side identity, camera mirroring or continuity. **Positive hand tracking, missing-frame identity linkage, clinical measurements and MDS-UPDRS scoring remain untested.** The earlier schematic 240-frame video continues to show zero hand detections, correctly yielding an empty trace and a hold-like failure.

### Public CC0 hand-video smoke — positive detection, NOT tapping (2026-11-08)

For a **nonclinical, non-Finger-Tapping** positive detector check, the authorized Mac tested Wikimedia Commons [*Woman counting on fingers.webm*](https://commons.wikimedia.org/wiki/File:Woman_counting_on_fingers.webm), by Mvolz, presented on Commons as **CC0 1.0**. The public clip depicts finger counting, not MDS-UPDRS 3.4, and has no clinical tap-event ground truth. Only a temporary 8-second 10fps 960-pixel-wide H.264 conversion (no audio) was used. Source media SHA-256: `71c19b00a8b575cbe6bbc061ce07e507cd303d8ba3f5263069b9fa3a6cab742e`; converted unmirrored MP4 SHA-256: `038a5f2e16ed1c9dc4520f681c3a8590e721e74c57912776aa29b8f2ddabfa0d`. No media, frames, raw traces or personal identifiers were committed.

| Technical output | Original 8 s | Horizontally mirrored copy |
| --- | ---: | ---: |
| Decoded frames | 80 | 80 |
| Four-point normalized hand traces | 72 | 73 |
| Vision-reported Right frames | 72 | 0 |
| Vision-reported Left frames | 0 | 73 |
| Research QC result (Labeling Tool Draft #5) | `hold/unverified_task` | `hold/unverified_task` |
| Clinical score `updrs_3_4` | `null` | `null` |

The observed *hand-detection coverage* is 90.0% and 91.25% for these two versions of **one source video**. That is neither landmark accuracy nor external model validation. Mirroring changed the Vision chirality report and some normalized distance estimates; the median absolute distance difference was 0.0542 for 71 shared detected frame indices, with **no independent ground-truth reference**. The legacy cycle summary estimated one open-close event in the unmirrored counting clip; it is **not a valid Finger Tapping tap**.

The video CLI previously populated `task: "finger_tapping"` without independent task verification. The trace now explicitly records `provenance.task_verification: "operator_declared_unverified"`. The existing Labeling Tool Draft PR #5 QC treats this marker as `hold/unverified_task`, even if landmarks are detected and handedness reports appear consistent. Thus the clip is **not admitted as a valid motor task or score sample**. The updated Swift code blob `b5d3f2300ca57a5f7d0a274f730534a209992089` passed macOS typecheck and geometry/chirality selftest, and the updated QC code blob `88e9131da99435803bcd61693c4554b2eecc847e` passed 18/18 synthetic tests on the authorized Mac.

**Remaining gates:** verified actual finger-tapping task, independent frame/landmark reference, continuous hand identity, true left/right hand confirmation, video orientation/mirroring and palm-span stability, appropriate research/data rights, independent clinical ratings, participant-level split and any MDS-UPDRS scale-use permission. No 3D calibration, diagnostic validity, treatment effect or score performance is established. This is a local research smoke check only.
