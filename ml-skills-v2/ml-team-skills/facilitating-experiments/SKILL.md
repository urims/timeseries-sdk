---
name: facilitating-experiments
description: Structures, tracks, and synthesizes ML experiments to ensure they are purposeful, reproducible, and lead to actionable conclusions. Prevents undirected exploration by requiring hypotheses before experiments and decisions after them. Use when designing an experiment plan, reviewing experiment results, deciding what to try next, writing experiment summaries, or ensuring the team doesn't repeat experiments or lose findings. Trigger on "run an experiment", "what should we try next", "experiment plan", "ablation study", "A/B test the model", "compare these runs", "experiment log", "what did we learn", "hypothesis", "experiment summary", "MLflow", "experiment tracking", "we already tried that". Do not use for validation and acceptance testing — use `validating-models`. Do not use for training configuration details — use `training-ts-models`.
---

# Facilitating Experiments

## Experiment-Driven Development

Every experiment starts with a **falsifiable hypothesis** and ends with a **decision**. No undirected exploration.

## Experiment Lifecycle

```
1. Hypothesis → 2. Design → 3. Execute → 4. Analyze → 5. Decide → 6. Record
                                                           ↓
                                                  Next hypothesis
```

### Step 1: Write the Hypothesis

Use this template:

```
Hypothesis: [Changing X] will [improve/reduce Y] by [expected magnitude]
            because [reasoning based on domain knowledge or prior evidence].

Example: Adding weather covariates to the TFT model will reduce MASE by
         at least 0.05 because demand for outdoor products correlates with
         temperature (observed in EDA: r=0.72).
```

If you can't fill in the "because", you need more exploratory analysis first.

### Step 2: Design the Experiment

Define before running:

```yaml
experiment:
  id: "EXP-007"
  hypothesis: "Weather covariates reduce MASE by ≥0.05"
  variable: "model input features"
  control: "TFT without weather (run EXP-006 baseline)"
  treatment: "TFT with temperature + precipitation"
  metric: "MASE on 5-fold expanding window backtest"
  success_threshold: "MASE improvement ≥ 0.05"
  compute_budget: "2 hours on ml.g4dn.xlarge"
  blocked_by: "EXP-006 must complete first (baseline)"
```

### Step 3: Execute

- Log everything to the experiment tracker (MLflow, W&B, or SageMaker Experiments)
- Use the same seed and data split as the control
- Save the full config, not just deltas from baseline

### Step 4: Analyze

Compare against the control on the pre-defined metric:

```
Results:
  Control (EXP-006):   MASE = 0.87 ± 0.03
  Treatment (EXP-007): MASE = 0.81 ± 0.04
  Delta: -0.06 (meets threshold of ≥0.05)

  Side effects:
  - Training time increased 15% (acceptable)
  - Coverage 90% unchanged at 91% (no degradation)
```

### Step 5: Decide

One of three outcomes:
- **Accept**: Incorporate the change going forward
- **Reject**: Document why, keep the control configuration
- **Iterate**: Promising but needs refinement — define the next hypothesis

### Step 6: Record

Every experiment gets a summary entry:

```markdown
## EXP-007: Weather Covariates for TFT

**Status**: Accepted
**Hypothesis**: Weather covariates reduce MASE by ≥0.05
**Result**: MASE improved from 0.87 to 0.81 (Δ = -0.06)
**Decision**: Accepted — weather covariates now included in production config
**Artifacts**: s3://experiments/EXP-007/
**Next**: EXP-008 — test adding holiday calendar features
```

## Preventing Common Experiment Mistakes

- **No hypothesis**: "Let's just try a bigger model" → Stop. Write the hypothesis first.
- **Moving goalposts**: Changing the success metric after seeing results → Use pre-registered thresholds.
- **Unreported negatives**: Negative results are valuable. Always record them.
- **Confounded variables**: Changed 3 things at once → Change one variable per experiment.
- **Forgotten baselines**: Always compare to a fixed control, not the previous experiment.

## Quality Checklist

- [ ] Hypothesis is falsifiable and written before execution
- [ ] Control and treatment clearly defined
- [ ] Success metric and threshold defined before results
- [ ] All artifacts saved to experiment tracker
- [ ] Decision recorded (accepted / rejected / iterate)
- [ ] Negative results documented with equal rigor
