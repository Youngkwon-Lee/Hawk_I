# Public Repo Safety

This repo can only be public when the current branch passes the public safety scan and contains no subject-identifiable media or real API keys.

## Allowed Demo Data

Allowed:
- Synthetic or de-identified skeleton/keypoint JSON under `frontend/public/data/`
- Small non-subject UI assets such as SVG icons
- Environment templates ending in `.env.example`

Not allowed:
- Real or clinical subject videos
- Face, full-body, or audio-bearing subject media
- Raw PD4T, TULIP, SNUH, or other clinical dataset media
- Real `.env`, `api_keys.env`, token, password, or API key files

## Local Check

Run before pushing:

```bash
python3 scripts/security/scan_public_safety.py
```

The same check runs in GitHub Actions on pushes and pull requests.
