# Finger Hardcase10 v0.5 Interpretation

## Run info
- run_id: `finger_hardcase10_v05_20260412_0944`
- task: finger_tapping hardcases (n=10)

## Key metrics
- Baseline match: **1/10 (10%)**
- Orchestrator Set A match: **4/10 (40%)**
- Orchestrator Set B match: **4/10 (40%)**
- Recheck rate (A/B): **90%**

## What this means
1. **Accuracy improvement is real**
   - Orchestration + guardrails improved hardcase score-match from 10% → 40%.
   - This supports the hypothesis that single-pass VLM is unstable on boundary/hard finger samples.

2. **Reliability still limited by stage2 parse stability**
   - Several samples still have `stage2_parse_ok=false`.
   - When specialist output is missing/unstable, the pipeline falls back to conservative recheck behavior.

3. **Current policy is high-safety / high-review-cost**
   - 90% recheck rate is likely too high for routine operation.
   - Practical next step is reducing unnecessary rechecks while preserving the 40% gain.

## Recommended next actions (v0.6)
- Add stage2 answer-only fallback prompt (`answer` integer only) for parse recovery.
- Relax recheck trigger for `sampling_mode_ratio=0.5` (warn-only) and reserve hard recheck for severe disagreement/parse failure.
- Re-run the same 10 hardcases for direct A/B comparability.
