# Analysis decision log

| ID | Decision | Reason | Alternative | Remaining risk |
|---|---|---|---|---|
| D01 | Model DR25 pipeline disposition | Provides a clear, balanced supervised-learning task | Claim confirmed-planet classification | Candidate is not physical ground truth; wording must remain precise |
| D02 | Use the uniform DR25 table | One documented release is more coherent for statistical study | Use the cumulative KOI table | DR25 still reflects one mission and one vetting process |
| D03 | Exclude scores, flags, comments, and final disposition | Prevent obvious target leakage | Use every high-performing field | Other derived features remain related to the vetting process |
| D04 | Group by host star | Prevent shared stellar properties across train and test | Random row split | Grouping cannot simulate another mission or future population |
| D05 | Use median imputation plus missingness indicators | Simple, reproducible, and fold-safe | Complete-case analysis or complex imputation | Median imputation can weaken relationships and hide missingness mechanisms |
| D06 | Compare logistic regression with constrained boosting | Shows transparent baseline versus flexible challenger | Start with a black-box model | Two candidates do not exhaust all sensible models |
| D07 | Require +0.02 Gini and non-worse Brier | Makes complexity justification explicit | Select highest AUC automatically | Hurdle is project judgement, not a universal standard |
| D08 | Calibrate after model selection | Keeps probability scaling separate from selection | Report raw model scores | Platt scaling may miss complex calibration patterns |
| D09 | Fix display bins on calibration data | Prevents tailoring bins to final outcomes | Equal-sized test bins | Test-bin sizes may differ, correctly reflecting fixed boundaries |
| D10 | Propagate reported errors with Monte Carlo | Connects measurement uncertainty to output uncertainty | Ignore measurement errors | Independent normal draws omit covariance, systematics, labels, and model uncertainty |
| D11 | Report multiple metrics, slices, and intervals | One metric cannot describe ranking, probability quality, and stability | Report accuracy only | Diagnostics still require domain judgement and external validation |
