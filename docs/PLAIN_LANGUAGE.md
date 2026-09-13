# The analysis in plain language

No machine-learning, statistics, or astronomy background is assumed.

## 1. What is the question?

Kepler measured small changes in stellar brightness. Some repeating dimmings look like a planet crossing a star; others are caused by eclipsing stars, contamination, stellar variability, or instrumental effects.

The model estimates the probability that a transit-like signal has the Kepler DR25 pipeline label **CANDIDATE** rather than **FALSE POSITIVE**. Candidate means worth treating as a planet candidate in this catalogue. It does not mean “confirmed planet.”

## 2. Supervised learning, features, and labels

**Supervised learning** learns from examples whose outcomes are already labelled.

- The **label** is candidate (1) or false positive (0).
- The **features** are measurements such as orbital period, transit depth, candidate radius, stellar temperature, and signal-to-noise.

After learning patterns from one part of the data, the model estimates probabilities for signals it has not seen.

## 3. Data leakage

Leakage happens when a model receives information that would not honestly be available for the intended prediction—or that already contains the answer.

NASA's disposition score, false-positive flags, comments, and final archive disposition strongly encode how the label was assigned. The project does not even download them. Identifiers are used only for grouping and traceability.

Excellent metrics from a leaking model are not excellent analysis.

## 4. Why keep each host star in one fold?

One star can have several Kepler Objects of Interest. If one signal from a star is used for training and another from the same star is used for testing, shared stellar measurements make the test easier than a truly new-star prediction.

**Group-aware splitting** keeps all signals around the same host star together. The same idea applies to all records from one customer, patient, machine, household, or site.

## 5. Why use four data roles?

- **Estimation folds (40%)** fit both candidates.
- **Selection fold (20%)** applies the pre-declared simple-versus-complex rule.
- **Calibration fold (20%)** corrects the chosen probability scale.
- **Test fold (20%)** gives the final untouched assessment.

Five grouped folds make these roles convenient. The initial training share is modest, but the selected estimator is refitted on estimation plus selection data before calibration.

## 6. Missing values and median imputation

Some catalogue measurements are absent. Deleting every incomplete row would waste data and could change the population.

**Median imputation** replaces a missing feature with the middle training value. A missingness indicator lets the model know a replacement occurred. The median is learned inside each training fold, never from held-out data.

This is a transparent baseline, not proof that missingness is harmless.

## 7. Log transforms

Period, depth, radius, temperature, and signal-to-noise can span wide ranges. A few very large values can dominate a simple model.

The transform `log(1 + x)` compresses large differences while preserving order. The `+1` safely handles zero. Non-positive values that are physically invalid for a transformed quantity become missing and follow the documented imputation rule.

## 8. Logistic regression

Logistic regression starts with a base tendency, adds weighted evidence from every feature, and converts the total into a probability from 0 to 1.

Why use it?

- coefficient directions are reviewable;
- it is fast and stable;
- it is a strong baseline for yes/no outcomes.

Its limitation is a relatively simple functional form unless curves and interactions are designed manually.

## 9. Standardisation

Stellar temperature and impact parameter use very different scales. Standardisation measures each logistic-regression feature relative to its own centre and spread. It is like converting several rulers to comparable units; it changes the numerical scale, not the information.

## 10. Gradient boosting

A decision tree asks a sequence of threshold questions. **Gradient boosting** builds many small trees; each new tree focuses on patterns the previous trees handled poorly.

It can learn non-linear thresholds and interactions automatically, but it is harder to inspect than one equation. The model is therefore a **challenger**, not an automatic winner.

## 11. The model-selection rule

Gradient boosting is chosen only if, on selection data:

1. Gini improves by at least 0.02; and
2. Brier score does not worsen.

The rule makes complexity a decision supported by evidence. The 0.02 value is a documented project choice, not a universal standard.

## 12. Probability calibration

A model may rank signals correctly but say “80%” for groups in which only 60% are candidates. **Calibration** asks whether predicted probabilities match observed frequencies.

**Platt scaling** fits one small logistic relationship between the selected model's original score and outcomes in the separate calibration fold. It corrects probabilities that are broadly too high, low, extreme, or narrow without changing their order.

## 13. ROC-AUC and Gini

Choose one candidate and one false positive at random. **ROC-AUC** is the chance that the model ranks the candidate higher.

- 0.50: random ordering;
- 1.00: perfect ordering.

**Gini** contains the same ranking information on a different scale:

```text
Gini = 2 × AUC − 1
```

Gini is included because it is common in credit-risk work. Neither AUC nor Gini says whether a 70% probability really occurs about 70% of the time.

## 14. Brier score and log loss

The **Brier score** is the mean squared distance between each probability and its 0/1 outcome. Lower is better.

**Log loss** also rewards accurate probabilities but punishes confident mistakes more strongly. Reporting ranking and probability metrics prevents one appealing number from hiding a weakness.

## 15. Calibration intercept, slope, and error

- An **intercept near 0** suggests no broad tendency to predict too high or low.
- A **slope near 1** suggests probabilities are not systematically too extreme or narrow.
- **Expected calibration error** is the average predicted-versus-observed gap across ten bins; lower is better.

The calibration plot makes the same comparison visible. These are diagnostics, not universal pass/fail thresholds.

## 16. Cross-validation

One split can be lucky. Five-fold grouped cross-validation repeatedly fits on four parts of the estimation data and checks the fifth while keeping host stars together. The mean AUC estimates typical ranking; its spread shows sensitivity to the particular fold.

The untouched test fold is still needed for the final result.

## 17. Bootstrap intervals

A metric from a finite sample is uncertain. The **bootstrap** draws 500 new test samples with replacement and recalculates the metric each time. The middle 95% is reported as an interval.

This describes sampling variation in the test data. It does not include every source of model uncertainty.

## 18. Measurement-error Monte Carlo

NASA reports upper and lower errors for many fitted quantities. For eight inputs, the project approximates one standard deviation by the mean absolute upper/lower error, draws a perturbed measurement, and predicts again. Repeating this 100 times produces a probability distribution for every test signal.

The 5th-to-95th percentile width shows how strongly reported measurement errors move a prediction. The method assumes independent, approximately normal errors and omits systematic, label, and model uncertainty.

## 19. Permutation importance

One feature is shuffled across test rows while everything else is left unchanged. If AUC falls strongly, the model relied on that feature for ranking.

Importance is not causation. Correlated features can share or mask importance, and the result describes this model and sample only.

## 20. Performance slices

Metrics are recalculated for bright/mid/faint targets and low/medium/high signal-to-noise. A difference can flag changes in measurement quality, missingness, sample composition, or model behaviour.

The same monitoring pattern transfers to customer segments, regions, assets, seasons, and operating regimes. A slice difference is a question to investigate, not a causal explanation.

## 21. Why no single decision threshold?

This project evaluates probabilities; it does not decide which objects receive telescope time. A threshold requires costs, capacity, scientific priorities, and consequences of false positives versus missed candidates.

The same is true outside astronomy: a statistically sensible probability becomes a decision only after domain goals, risk appetite, policy, and human impact are defined.
