# HawkEye filming demo assets

These assets are for the privacy-safe documentary filming screen at
`/filming-demo`. They must not be replaced with patient recordings.

- Source: [Pexels video 9616883](https://www.pexels.com/video/side-view-of-businessman-walking-with-prosthetic-leg-9616883/)
- Creator: Lara Jameson
- Usage: Pexels marks the source as free to use.
- `movement-original.mp4`: five-second, muted 406×720 crop of the source.
- `movement-analysis.mp4`: skeleton overlay generated from the same frames.
- `movement-analysis.json`: joint-angle and normalized image-speed values
  generated from the same frames. These values are a technical demonstration,
  not a clinical diagnosis.

Regenerate the derived assets with:

```bash
.venv/bin/python scripts/prepare_filming_demo.py \
  --input frontend/public/demo/movement-original.mp4 \
  --overlay frontend/public/demo/movement-analysis.mp4 \
  --data frontend/public/demo/movement-analysis.json
```
