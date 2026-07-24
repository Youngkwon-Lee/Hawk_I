# Hawk_I Local Runbook

This runbook records the local development path verified on 2026-07-23.

## Prerequisites

- Python 3.10
- uv
- Node.js 18+
- npm

## Install

From the repo root:

```bash
uv venv --python 3.10 .venv
uv pip install -r backend/requirements.txt pytest pytest-cov pytest-mock

cd frontend
npm install
```

## Run Backend

The backend defaults to port `5000`, but macOS may already use that port. Use `5001` when `5000` is occupied:

```bash
PORT=5001 \
FLASK_ENV=development \
FRONTEND_URL=http://localhost:3000 \
uv run --no-sync python backend/app.py
```

Expected health check:

```bash
curl -sS http://127.0.0.1:5001/health | python3 -m json.tool
```

Expected status is `healthy`. `OPENAI_API_KEY` is not required for the health endpoint or basic upload validation. Without it, chat and VLM scoring log warnings and are disabled.

## Run Frontend

In another terminal:

```bash
cd frontend
BACKEND_URL=http://127.0.0.1:5001 \
NEXT_PUBLIC_API_URL=http://127.0.0.1:5001 \
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open:

```text
http://127.0.0.1:3000
```

The frontend uses `BACKEND_URL` for Next.js rewrites and `NEXT_PUBLIC_API_URL` for browser-side API calls.

## Verified Smoke Checks

```bash
npm --prefix frontend run build
npm --prefix frontend run lint

TEST_API_URL=http://127.0.0.1:5001 \
uv run --no-sync python -m pytest \
  backend/tests/test_api_e2e.py::TestHealthEndpoints \
  backend/tests/test_api_e2e.py::TestAnalysisAPI::test_analyze_without_file
```

For the full backend suite, run the Flask server first and point the E2E tests at it:

```bash
TEST_API_URL=http://127.0.0.1:5001 \
uv run --no-sync python -m pytest backend/tests -q
```

To smoke test the asynchronous upload flow with a local gait clip:

```bash
mkdir -p /tmp/hawkeye_smoke
ffmpeg -hide_banner -loglevel error -y \
  -ss 0 -t 6 \
  -i /path/to/local-gait-video.mp4 \
  -an -c:v libx264 -preset veryfast -crf 28 \
  /tmp/hawkeye_smoke/gait_smoke_6s.mp4

curl -sS -X POST http://127.0.0.1:5001/api/analyze \
  -F 'video_file=@/tmp/hawkeye_smoke/gait_smoke_6s.mp4;type=video/mp4' \
  -F 'patient_id=smoke_test' \
  -F 'test_type=gait' \
  -F 'scoring_method=rule'
```

Observed results on 2026-07-23:

- Frontend build passes.
- Frontend lint passes with warnings only.
- Backend tests pass: `32 passed, 5 skipped`.
- Backend `/health` returns `healthy`.
- Backend `/api/analyze` without a video returns HTTP 400 with `No video file provided`.
- Backend `/api/analyze` accepts a local 6 second gait clip, completes asynchronously, returns `video_type=gait`, gait metrics, rule-based UPDRS score, events, and skeleton/original video URLs.
- Frontend `/` and `/test` render in browser.
- Frontend proxy `/api/backend/analyze` reaches the backend when `BACKEND_URL` points at the active backend port.

## Production Topology

Current production setup verified on 2026-07-23:

- Frontend: Vercel project `hawkeye-labeling-tool`
- Public app URL: `https://hawkeye-labeling-tool.vercel.app`
- Backend runtime: home desktop WSL, systemd user service `hawkeye-backend.service`
- Backend local port: `127.0.0.1:5891`
- Public backend tunnel: Tailscale Funnel path `https://desktop-t43sn5m-1.tailde3b80.ts.net/hawkeye-api`
- Browser API path: same-origin `https://hawkeye-labeling-tool.vercel.app/api/*`
- Browser file path: same-origin `https://hawkeye-labeling-tool.vercel.app/files/*`

Vercel env:

```text
NEXT_PUBLIC_API_URL=https://hawkeye-labeling-tool.vercel.app
BACKEND_URL=https://desktop-t43sn5m-1.tailde3b80.ts.net/hawkeye-api
```

Optional backend env for writing completed analyses into physio_app
`public.activity_sessions` and `public.observations`:

```text
HAWKEYE_SUPABASE_URL=https://iwtyzcwiovuvmsodtusx.supabase.co
HAWKEYE_SUPABASE_SERVICE_KEY=<server-side secret/service key>
HAWKEYE_SUPABASE_ORGANIZATION_ID=<organizations.id>
HAWKEYE_SUPABASE_CREATED_BY_PERSON_ID=<provider/operator persons.id for creator>
HAWKEYE_SUPABASE_PERFORMER_PERSON_ID=<persons.id for AI/camera performer, defaults to creator>
HAWKEYE_SUPABASE_SUBJECT_PERSON_ID=<optional selector hint; only used if it is an active org_clients.person_id>
HAWKEYE_SUPABASE_ACTIVITY_SESSION_ID=<optional existing activity_sessions.id>
HAWKEYE_SUPABASE_ACTIVITY_SESSIONS_TABLE=activity_sessions
HAWKEYE_SUPABASE_OBSERVATIONS_TABLE=observations
```

Do not set these in the Vercel frontend project. They belong on the Flask
backend runtime only. If `HAWKEYE_SUPABASE_ACTIVITY_SESSION_ID` is omitted, the
backend creates one completed camera assessment session per saved analysis.
The frontend reads selectable people from the backend-only endpoint
`GET /api/physio/subjects`; that route uses the server Supabase key and returns
only active `org_clients` in the configured organization. Completed analyses are
written only when the request includes an explicit physio_app subject/organization
context; the backend does not fall back to `HAWKEYE_SUPABASE_SUBJECT_PERSON_ID`
as a write target.

The browser must call the Vercel origin, not the Tailscale URL directly. Direct browser requests to the Tailscale Funnel URL can be blocked by browser Private Network Access checks. Next.js rewrites proxy `/api/*` and `/files/*` server-side to the Tailscale backend.

Home desktop checks:

```bash
ssh yk@100.125.26.99 'systemctl --user status hawkeye-backend.service --no-pager'
ssh yk@100.125.26.99 'tailscale funnel status'
curl -sS https://hawkeye-labeling-tool.vercel.app/api/physio/subjects
curl -sS https://hawkeye-labeling-tool.vercel.app/api/vlm/status
```

Production smoke test:

```bash
bash scripts/hawkeye_production_smoke.sh
```

By default this uses `/tmp/hawkeye_smoke/gait_smoke_6s.mp4`. To use a different file:

```bash
HAWKEYE_SMOKE_VIDEO=/path/to/gait.mp4 bash scripts/hawkeye_production_smoke.sh
```

## Known Gaps

- The public repo still does not include a de-identified sample video fixture, so the full upload smoke uses a local external gait clip.
- Production analysis availability depends on the home desktop WSL runtime and Tailscale Funnel staying online.
- OpenAI-backed chat and VLM paths require `OPENAI_API_KEY`; without it, fallback interpretation is used.
- `npm audit` reports dependency vulnerabilities; review before production deployment.
- Next.js warns that `middleware.ts` should migrate to the newer `proxy` convention.
- Several frontend lint warnings remain for unused imports/variables and `<img>` usage, but there are no lint errors.
