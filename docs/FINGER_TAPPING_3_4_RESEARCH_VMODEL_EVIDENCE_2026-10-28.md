# Finger Tapping 3.4 — research V-Model evidence crosswalk (2026-10-28)

> **Status:** research documentation candidate, not an approved medical-device file or a clinical performance report.
> **Owner:** Hawk_I research pipeline. Annotation ontology/schema and pilot records remain owned by `hawkeye-labeling-tool`; a second annotation database or medical product is **not** being created.
> **Immutable source context:** Hawk_I `1feab81f867720625a774911a4fac55fcce15b58`; labeler `master@27bee119fdcd5aac5a9af5bbc4c334ef8a661587`; Hawkeye_paper `0b1c549393555c01371ce6bf20f51fff0bc296f4`.
> **Execution:** documentation only. No model training, dataset access, video review, clinical/IRB approval, deployment, or new test run in this change.

## Product and research boundary (not an approval)

- **Research question:** can a quality-gated, per-side finger-tapping video yield reproducible kinematic observations and reviewed ordinal-score estimates consistent with the relevant expert/reference rating, with uncertainty and failure cases reported?
- **Task:** finger tapping related to MDS-UPDRS Part III item **3.4**. Left and right must remain distinct. A research `score_anchor.updrs_3_4` is **not** a confirmed patient diagnosis, disability severity, change in treatment, or the whole Part III score.
- **Method option:** kinematics-first for observable amplitude, speed, decrement, rhythm, hesitation and halt features; video-language explanation is **review material**, not a primary clinical scoring authority. Compare against a frozen baseline rather than presuming a VLM is better.
- **Out of scope:** autonomously assigning clinical severity, Parkinson diagnosis, use in patient care/physio_app confirmed state, regulatory classification, commercially distributing an MDS rating rubric, or starting human research.
- **Approval gates:** source/consent and license review; MDS permission for intended research/scale use; authorized study/IRB determination; research owner review. None is satisfied by this document.

## Existing sources to reuse, not copy

| Owned source | Reusable fact or contract | What it does *not* prove |
| --- | --- | --- |
| [Hawk_I project charter](../PROJECT_CHARTER.md) | Task-specific research, clinician review and abstention boundaries | Clinical validity or regulatory product approval |
| [Feature-first finger plan](MOTOREXAM_VQA_FINGER_FEATURE_FIRST_PLAN_V0_1.md) | Older comparison recommends kinematic features first; reported held-out exact 0.570 / MAE 0.469 for one baseline | New test run, external validation or stable performance across populations |
| [Hawk_I evaluation config](EVALUATION_CONFIG.md) / [PD4T test report](PD4T_TEST_SET_REPORT.md) | Historical model/dataset setup and score imbalance evidence | Current dataset manifest or a locked clinical reference set |
| [Annotation ontology v0.1](https://github.com/Youngkwon-Lee/hawkeye-labeling-tool/blob/27bee119fdcd5aac5a9af5bbc4c334ef8a661587/docs/finger_tapping_primitive_ontology_v0.md) | Observability, primitive ratings, per-span evidence, `pass/hold` quality gate, separate score anchor | Authorized media, clinical validation or automatic scale conversion |
| [Schema v0.1](https://github.com/Youngkwon-Lee/hawkeye-labeling-tool/blob/27bee119fdcd5aac5a9af5bbc4c334ef8a661587/schemas/finger_tapping_primitive_ontology_v0.schema.json) / [pilot runbook](https://github.com/Youngkwon-Lee/hawkeye-labeling-tool/blob/27bee119fdcd5aac5a9af5bbc4c334ef8a661587/docs/finger_tapping_pilot_runbook_v0.md) | Machine-readable fields, blinded two-reviewer assignment and adjudication workflow | Completed approved human-video pilot |
| [Public source due diligence](https://github.com/Youngkwon-Lee/hawkeye-labeling-tool/blob/27bee119fdcd5aac5a9af5bbc4c334ef8a661587/docs/finger_tapping_public_source_review_2026-07-27.md) | Historical source candidates and explicit governance hold | Current consent/license approval or permission to reuse original PD4T media |
| [Hawkeye_paper metrics](https://github.com/Youngkwon-Lee/Hawkeye_paper/blob/0b1c549393555c01371ce6bf20f51fff0bc296f4/docs/evaluation_metrics.md) / [data rules](https://github.com/Youngkwon-Lee/Hawkeye_paper/blob/0b1c549393555c01371ce6bf20f51fff0bc296f4/data/README.md) | QWK, Spearman, MAE, subject split, provenance principles | IRB approval or current clinical performance |

## V-Model: requirement → design / risk → proposed verification

All IDs below are **planning identifiers**, not signed software requirements, executed test IDs or ISO 14971 risk decisions. A planning row is always `NOT RUN / NOT APPROVED` until evidence shows otherwise.

| ID | Intended research requirement (URS/SRS candidate) | Design/control owned by | Risk candidate | Proposed test/evidence; status |
| --- | --- | --- | --- | --- |
| FT-RQ-01 | A side-specific clip must pass task/quality checks before any rating | Ontology `quality_gate`, `side`, schema hold/null constraints | FT-H01 wrong task/hand or unobservable frames appear as valid | Wrong-side/occluded/blur/short task fixtures, schema rejection and reviewer hold readback — **NOT RUN** |
| FT-RQ-02 | Observable primitives must retain temporal evidence independent from an ordinal score | Ontology `primitives`, evidence spans and `measurements` | FT-H02 model invents movement features or treats missing as normal | Time-interval/cycle assertions, unobservable `null` semantics, evidence-review fixtures — **NOT RUN** |
| FT-RQ-03 | Two qualified reviewers' labels must be independent before adjudication | Labeler pilot assignment and adjudication contract | FT-H03 label leakage, unblinded rater bias | Two-reviewer blinded export, disagreement queue, provenance review — **NOT RUN** |
| FT-RQ-04 | Training, tuning and locked evaluation must separate subjects | PD4T `stratified/` splits; frozen manifest in Hawk_I research | FT-H04 subject leakage / inflated score performance | Duplicate subject checks, train/val/test manifest hash, immutable holdout — **NOT RUN** |
| FT-RQ-05 | Predictions must include confidence/abstain reason and exclude held clips from scoring denominator | Ontology quality gate, feature-first research plan | FT-H05 false certainty on poor media / out-of-distribution clips | Report coverage + selective error, fail-by-condition, abstention consistency — **NOT RUN** |
| FT-RQ-06 | Score estimation must be reproducible and compared with an expert anchor | Model/config/feature versions, historical score reference, evaluation metrics | FT-H06 untracked model or label version changes | Locked model/data/feature/threshold IDs, QWK, MAE, per-score confusion, CI and independent benchmark — **NOT RUN** |
| FT-RQ-07 | Study data access, retention and clinical scale use must be authorized | Governance review in labeling owner + external MDS licensing | FT-H07 non-consented reuse / prohibited rubric distribution | Approved governance and MDS permissions evidence, restricted storage audit — **NOT VERIFIED** |

## Research safety and acceptance-plan boundaries

- **Risk review candidates:** FT-H01–H07 need specified causal chains, potential harms, controls, verification methods and residual-risk decisions. The table above is **not** an approved ISO 14971 hazard analysis.
- **Before evaluation:** predeclare data access decision, per-task/side inclusion and exclusion, original vs derived clip provenance, subject-level deduplication, reference rater qualification/adjudication, locked train/validation/test, version fingerprints, baseline and failure thresholds. Older data tables are dated snapshots.
- **Primary research endpoints (proposed):** QWK and Spearman for ordinal agreement; MAE/MA-MAE, exact and one-level-off accuracy, per-class breakdown. Report subject-level confidence intervals and held-out coverage / selective error; define pass/fail targets **before** execution (no numeric target approved yet).
- **Failure coverage:** camera cropping, missing cycles, occlusion, left/right reversal, jitter, low frame rate, scale imbalance, external dataset shifts, medication timing where relevant. Reviewer must be able to inspect the evidence and decline a score.
- **Permission safeguard:** MDS requires permission for use of its rating scales in research/software, with licensing dependent on use; confirm the actual study/language/protocol grant before reproducing or integrating the rubric. Reference: https://www.movementdisorders.org/MDS/Education/Rating-Scales/Training-Programs/Terms-of-Use.htm and https://www.movementdisorders.org/MDS/MDS-Clinical-Outcome-Assessment/Rating-Scales-Pricing.htm . This file does not reproduce the copyrighted scoring rubric.
- **Data safeguard:** no clinical/raw video, identifiable subject data, label JSONL, reviewer identifiers, access tokens or signed media links in Git. Research media access is a separate approved storage and consent concern. The dated labeler runbook states no approved human clip for the ontology pilot; **do not infer that condition is resolved today**.

## Validation and change-control record to create when a real experiment runs

Record: `FT-RUN-ID` / revision / source and model SHA / feature and rubric reference / dataset ID, access and consent reference / subject split hash / frame sampling and camera setup / approved plan and thresholds / execution time and operator / all outputs incl. holds and missing / errors and retests / rater and adjudication provenance / reviewer decision / linked research release. Preserve old failed runs and decisions. Distinguish model evaluation from clinical performance, product release and regulator authorization.

**Cross-program relation:** TUG video timing is a separate Kinelo product *candidate* owned by `physio_app`. No transfer of Hawkeye ratings into TUG validity, and no automatic import of experimental results into `physio_app` clinical truth. Both can reuse the **practice** of requirement IDs, quality gates, versioned evidence and risk→test traceability.

**Next owner action:** approve the precise research question, source/license/IRB and evaluator scope, then execute only the applicable protocol. No QMS, product or human-use decision has been made here.
