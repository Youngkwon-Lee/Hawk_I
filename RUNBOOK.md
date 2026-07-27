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

The frontend uses `BACKEND_URL` for Next.js rewrites and `NEXT_PUBLIC_API_URL` for browser-side API calls. Set `NEXT_PUBLIC_UPLOAD_API_URL` to the public backend prefix when large multipart uploads must bypass the Vercel request-body proxy. Only `POST /api/analyze` uses this direct URL; progress, result, history, and file reads remain on the active frontend origin.

For the isolated home-desktop preview:

```bash
NEXT_PUBLIC_UPLOAD_API_URL=https://desktop-t43sn5m-1.tailde3b80.ts.net/hawkeye-preview
```

The preview backend runs separately from production on port `5892` and stores
uploads, progress, and completed result JSON under the persistent path
`/home/yk/previews/hawkeye-uploads`. Install the checked-in user
service after updating the preview worktree:

```bash
install -m 0644 deploy/systemd/hawkeye-preview-backend.service \
  ~/.config/systemd/user/hawkeye-preview-backend.service
systemctl --user daemon-reload
systemctl --user enable --now hawkeye-preview-backend.service
```

The service deliberately uses one analysis worker so it can run beside the
two-worker production backend without exhausting the WSL runtime. Its upload
directory is overridden after loading the shared backend secrets, so preview
results never mix with production result files.

The backend CORS policy allows the bounded Hawk I and ParkiCheck Vercel origins by default. Additional exact origins can be supplied as a comma-separated `CORS_ALLOWED_ORIGINS` backend environment variable.

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
BACKEND_URL=https://desktop-t43sn5m-1.tailde3b80.ts.net/hawkeye-api
```

Browser requests use relative same-origin `/api/*` and `/files/*` paths. Do not
set a preview deployment's browser API URL to the production application.

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
HAWKEYE_PHYSIO_CONTEXT_TOKEN=<long random backend-only operator token>
```

Do not set these in the Vercel frontend project. They belong on the Flask
backend runtime only. If `HAWKEYE_SUPABASE_ACTIVITY_SESSION_ID` is omitted, the
backend creates one completed camera assessment session per saved analysis.
The subject directory endpoint `GET /api/physio/subjects` and any analysis
request containing `physio_*` write context require `Authorization: Bearer
<HAWKEYE_PHYSIO_CONTEXT_TOKEN>`. Public browser clients must not receive or
embed this token. Without an authenticated server-side operator proxy, the
frontend runs an anonymous research analysis and does not write Hawk I results
directly into a selected patient record. ParkiCheck links its own signed-in
observation to Hawk I with a shared `assessment_session_id` and stores the Hawk I
provenance itself. The backend never falls back to
`HAWKEYE_SUPABASE_SUBJECT_PERSON_ID` as an implicit write target.

When the ParkiCheck user explicitly enables Hawk I research review, the request
may also include a small `medication_context` JSON object containing only the
patient-reported medication name, dose, reported dose time, assessment time,
and elapsed hours. The backend validates and whitelists those fields, returns a
descriptive `medication_timing` relationship, and stores both objects alongside
the shared assessment session. This is a single-observation time relationship;
it does not infer efficacy, an ON/OFF state, or a dosing recommendation.

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
- Medication-effect inference requires repeated, comparable assessments and clinician review. The current integration only preserves patient-reported context and displays its time relationship to one assessment.

## De-identified finger and medication smoke

Build a hand-only synthetic clip from the Apache-2.0 MediaPipe gesture
recognizer test assets. The downloaded source photos stay in a temporary
directory and the script deletes them after encoding; the output contains only
cropped hands. This proves pipeline connectivity, not clinical validity.

```bash
bash scripts/build_deidentified_finger_fixture.sh
PYTHONPATH=backend /path/to/project-venv/bin/python \
  scripts/run_local_medication_e2e.py \
  /tmp/hawkeye_smoke/deidentified_hand_motion.mp4
```

The smoke clears Supabase and external LLM credentials in its own process,
requires detected hand landmarks, preserves the synthetic assessment and
medication timing contract, and fails if any Supabase observation is saved.

Repeated medication comparison is deliberately observational. ParkiCheck and
Hawk I group only the same patient, task, patient-reported medication name, and
dose; they show numeric first-to-latest differences while keeping
`can_infer_medication_effect=false` and requiring clinician review.
- OpenAI-backed chat and VLM paths require `OPENAI_API_KEY`; without it, fallback interpretation is used.
- `npm audit` reports dependency vulnerabilities; review before production deployment.
- Next.js warns that `middleware.ts` should migrate to the newer `proxy` convention.
- Several frontend lint warnings remain for unused imports/variables and `<img>` usage, but there are no lint errors.
