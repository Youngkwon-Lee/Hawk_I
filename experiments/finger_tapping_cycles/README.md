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

The self-test exercises **the same pure image-plane normalization function used by the video exporter** without any camera or data access. It checks known thumb–index/palm-span ratios, pixel aspect correction, uniform-resolution changes, mirror-invariant *distance* (not handedness), zero palm span, low landmark confidence, nonfinite timestamps, out-of-frame locations and impossible gap ratios. On macOS Apple Swift 6.4.2: \`xcrun swiftc -typecheck\` PASS (one pre-existing deprecated AVFoundation API warning); \`--geometry-selftest\` PASS.

The existing **schematic synthetic** \`SYNTH_FT_001.mp4\` was decoded with the optional export path: **240 frames decoded, 0 Vision hand detections, 0 output samples, \`insufficient_hand_detection\`**. This confirms fail-closed empty-trace output, **not** successful hand tracking or real-video measurement. It must not be used as a validated positive-detection fixture.

Actual hand detections, left/right verification, continuity across missing frames, palm-span stability with hand orientation, mirror/camera metadata, physical calibration, and MDS-UPDRS validity remain unverified. Only authorized media and approved use may be used for any later positive-detection check.
