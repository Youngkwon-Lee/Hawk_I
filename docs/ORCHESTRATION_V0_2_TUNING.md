# Hawkeye Orchestration v0.2 Tuning Plan

Updated: 2026-04-11

## Goal
Reduce boundary-case misses in finger tapping while keeping latency and recheck rate manageable.

## Candidate threshold sets (A/B)

### Set A (balanced)
- `sampling_mode_ratio_threshold`: **0.67**
- `disagreement_guardrail_gap`: **>= 2**
- `route_to_stage2_if_parse_fail`: true
- `route_to_stage2_if_no_timestamp_evidence`: true

### Set B (safer / stricter)
- `sampling_mode_ratio_threshold`: **0.75**
- `disagreement_guardrail_gap`: **>= 1**
- `route_to_stage2_if_parse_fail`: true
- `route_to_stage2_if_no_timestamp_evidence`: true

## 20-sample validation protocol
- Split: finger_tapping 10 + gait 10
- Include all `HARDCASE_REGISTRY_V0_2.json` samples first, then fill remaining randomly
- Compare:
  1. Stage1 only baseline
  2. Orchestrator v0.2 Set A
  3. Orchestrator v0.2 Set B

## Metrics to report
- `score_match_rate`
- `parse_ok_rate`
- `recheck_rate`
- `p95_latency_sec`
- `finger_hardcase_match_rate`

## Decision rule
Promote config if:
- `finger_hardcase_match_rate` improves by >= 10%p vs baseline, and
- `p95_latency_sec` increase <= 40%, and
- `recheck_rate` <= 35%

## Notes
- Keep `finger::15-008175_l_042` pinned as sentinel hard-case.
- If Set B increases recheck_rate too much, fallback to Set A and tune only disagreement rule.
