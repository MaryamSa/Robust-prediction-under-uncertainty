# Analysis review checklist

Use this before presenting or adapting the project.

## Problem and data

- [ ] Is the positive class stated precisely?
- [ ] Is “candidate” kept distinct from “confirmed planet”?
- [ ] Are the source release, DOI, query, and row count recorded?
- [ ] Are obvious leakage fields absent from the download and model?
- [ ] Are IDs excluded from model inputs?
- [ ] Are missingness and invalid values reported?

## Validation design

- [ ] Are related signals from one host star confined to one fold?
- [ ] Are estimation, selection, calibration, and test roles separate?
- [ ] Is the challenger rule fixed before final testing?
- [ ] Are preprocessing steps fitted only on training folds?
- [ ] Are ranking, probability, calibration, and uncertainty results all present?
- [ ] Are slice sizes shown alongside slice metrics?

## Uncertainty and interpretation

- [ ] Are bootstrap intervals labelled as sampling uncertainty?
- [ ] Are Monte Carlo assumptions and omitted uncertainty sources explicit?
- [ ] Is feature importance kept distinct from causation?
- [ ] Are subgroup differences presented as monitoring signals, not explanations?
- [ ] Are no universal thresholds invented?

## Reproducibility

- [ ] Does `make analysis` regenerate reports, tables, and figures?
- [ ] Do `make test` and `make lint` pass?
- [ ] Do repeated runs give identical numerical and visual outputs?
- [ ] Are raw data, cache files, and fitted binary models excluded from Git?
- [ ] Does the README lead with the aim, methods, transferability, and limitations?

## Adaptation to another sector

- [ ] Has the target and time horizon been redefined?
- [ ] Has entity grouping been changed to the correct customer, asset, or site?
- [ ] Have temporal/spatial validation needs been addressed?
- [ ] Are costs, policy, ethics, regulation, and operational consequences included?
- [ ] Are only demonstrated skills claimed?

