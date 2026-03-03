---
name: explaining-technically
description: Produces rigorous, reproducible technical explanations of ML model behavior, architecture choices, training dynamics, and forecast quality for ML engineers, data scientists, and technical reviewers. Generates error analysis, feature attributions, attention interpretations, and architecture trade-off documentation. Use when a technical audience needs to understand model internals, debug predictions, review architecture decisions, or interpret experiment results. Trigger on "explain the model technically", "why did the model predict X", "architecture decision", "error analysis", "attention weights", "feature attribution", "gradient analysis", "debug this prediction", or any deep-dive into model mechanics. Do not use for non-technical audiences — use `explaining-for-stakeholders` instead. Do not use for teaching concepts from scratch — use `teaching-timeseries-dl` instead.
---

# Explaining Technically

## Goal

Provide technically precise, traceable, and reproducible explanations of model behavior that enable peer review, debugging, and informed architectural decisions.

## Explanation Modes

Select the mode that matches the request:

### Mode A: Prediction Explanation
Why did the model produce this specific output?

1. Show the input window and any covariates
2. Identify top contributing features (SHAP, attention, gradient-based)
3. Compare against a naive baseline prediction
4. Note if the prediction falls within historical variance

### Mode B: Error Analysis
Why is the model wrong on these cases?

1. Segment errors by magnitude, time period, series characteristics
2. Identify systematic patterns (always undershoots promotions, misses regime changes)
3. Propose targeted fixes with expected impact
4. Link errors to data quality issues if applicable

### Mode C: Architecture Justification
Why this architecture and not another?

1. State the decision using the ADR template from `designing-ts-architectures`
2. Compare against at least 2 alternatives on the specific problem dimensions
3. Cite inductive biases that match the data characteristics
4. Acknowledge trade-offs honestly

### Mode D: Training Dynamics
What happened during training and why?

1. Describe the loss curve behavior (convergence, plateaus, instabilities)
2. Explain learning rate schedule effects
3. Identify overfitting onset with train/val divergence point
4. Recommend next training adjustments

## Output Format

Every technical explanation must include:

```
## Technical Explanation: [Title]

**Context**: [What was asked / what triggered this analysis]
**Method**: [Technique used — e.g., SHAP, attention rollout, gradient × input]
**Findings**: [Structured results with numbers]
**Reproducibility**: [Seed, config hash, data version, git commit]
**Implications**: [What this means for the next decision]
```

## Quality Checklist

- [ ] All claims are backed by specific numbers or visualizations
- [ ] Reproducibility metadata included
- [ ] At least one comparison point (baseline, alternative, historical)
- [ ] Technical terms are used precisely (not interchangeably)
- [ ] Implications section connects to a concrete next step
