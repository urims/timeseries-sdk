# ML Team Agent Skills Framework

## Purpose

This framework standardizes how Claude assists an ML team that builds **time series forecasting solutions** and delivers results to **non-technical stakeholders**. Every skill traces back to one question: *does this help us ship a trustworthy forecast that stakeholders can act on?*

## Pillars

```
DELIVER           BUILD                OPERATE            GOVERN
─────────         ──────               ────────           ──────
Stakeholder       TS DL Architecture   SageMaker          Reproducibility
  Communication   TS DL Teaching       Resource           Data Quality
Non-Tech          TS DL Training         Optimization
  Explainability                       API Design
Technical                              Developer UX
  Explainability

CROSS-CUTTING: Validating Models · Facilitating Experiments
```

## Forecasting Pipeline (skills chain in this order)

```
guarding-data-quality
  → designing-ts-architectures
    → training-ts-models
      → validating-models
        → explaining-for-stakeholders
          → communicating-to-stakeholders

Supporting at any stage:
  facilitating-experiments    → before every experiment
  enforcing-reproducibility   → enforced throughout
  optimizing-resources        → on-demand constraint handling
  operating-sagemaker         → cloud execution
  teaching-timeseries-dl      → upskilling at any stage
```

## Design Principles

1. **Stakeholder alignment first** — every technical decision traces to a business outcome.
2. **Token efficiency** — concise SKILL.md bodies, heavy content in `references/` and `scripts/`.
3. **Reproducibility by default** — all code-generating skills emit seed, config, data checksum, and git commit.
4. **Explainability at every layer** — what happened → why → what it means for the business.
5. **Progressive disclosure** — Claude loads only what the current task requires.
