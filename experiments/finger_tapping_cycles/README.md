# Finger-tapping cycle analysis (macOS)

`analyze_finger_tapping_cycles.swift` reads one finger-tapping video with AVFoundation and estimates tapping cycles.
Moved from `hawkeye-labeling-tool` on 2026-11-04 (source HEAD `d7ff130`).

```bash
swift experiments/finger_tapping_cycles/analyze_finger_tapping_cycles.swift VIDEO
```

macOS only. Run it on local media; do not commit videos or per-clip outputs.
