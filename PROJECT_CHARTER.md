
# Hawkeye Project Charter

Status: research-active
Updated: 2026-09-24

## Inheritance
For Kinelo portfolio decisions, this project inherits Youngkwon-Lee/second-brain@main:company/KINELO_CONSTITUTION.md. Research methods and claims remain governed by the applicable study/data-use rules.

## Role
Hawkeye is a Parkinson movement-assessment research project focused on video-based, task-specific, interpretable movement analysis and reliability support.

## Owns
- research pipelines for Parkinson motor tasks
- video/pose/feature experiments
- task-specific evaluation and labeling research
- reproducible research outputs and model comparisons

## Does not own
- autonomous Parkinson diagnosis
- clinical care decisions
- physio_app product priority or confirmed patient state
- a generic medical VLM platform
- permission to publish or reuse restricted datasets outside their terms

## Current validation gate
Demonstrate task-specific validity and reliability against appropriate expert/reference labels, with explicit dataset split, quality gate, uncertainty/abstention behavior, and reproducible evaluation.

## Hard boundaries
- predicted scores remain research outputs unless separately clinically validated
- patient/video data and credentials must not be exposed in public artifacts
- do not infer broad clinical capability from one task or dataset
- preserve human expert review and uncertainty
- public demo assets must be de-identified and allowed for public use

## Success signals
- reproducible held-out performance
- clinically interpretable feature behavior
- error/uncertainty analysis
- agreement/reliability evidence appropriate to the target task
- a clear research contribution beyond generic VLM prompting

## Stop conditions
Do not broaden to more tasks/modalities merely to increase scope. Expand only when the current task has a defensible research question and evidence.

## Canonical sources
- docs/PROJECT_SUMMARY.md
- docs/EVALUATION_CONFIG.md
- docs/PD4T_DATASET.md
- current experiment configs/results and paper artifacts
