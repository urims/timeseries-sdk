# ML Team Agent Skills Framework

> **16 Claude skills for ML teams delivering time series forecasting to non-technical stakeholders.**
> Standardized. Reproducible. Stakeholder-aligned.

## What This Is

A library of [Claude Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — structured prompt libraries that guide Claude toward consistent, high-quality, context-aware outputs across the full ML development lifecycle.

Each skill lives in its own folder with a `SKILL.md` file. Skills are organized into **4 pillars** and **2 cross-cutting layers**, following [Anthropic's skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices).

## Skill Index

| # | Skill | Pillar | Has Scripts | Has References |
|---|-------|--------|:-----------:|:--------------:|
| 1 | `explaining-for-stakeholders` | Deliver | | |
| 2 | `explaining-technically` | Deliver | | |
| 3 | `communicating-to-stakeholders` | Deliver | | |
| 4 | `designing-ts-architectures` | Build | | ✓ |
| 5 | `teaching-timeseries-dl` | Build | | |
| 6 | `training-ts-models` | Build | ✓ | |
| 7 | `operating-sagemaker` | Operate | | ✓ |
| 8 | `optimizing-resources` | Operate | | |
| 9 | `designing-apis` | Operate | | |
| 10 | `improving-developer-ux` | Operate | | |
| 11 | `validating-models` | Cross-Cutting | ✓ | |
| 12 | `facilitating-experiments` | Cross-Cutting | | |
| 13 | `guarding-data-quality` | Govern | ✓ | |
| 14 | `enforcing-reproducibility` | Govern | ✓ | |
| 15 | *(reserved)* | | | |
| 16 | *(reserved)* | | | |

## Repo Structure

```
ml-team-skills/
├── FRAMEWORK.md                          ← Architecture overview
├── evals/
│   └── evals.json                        ← Test prompts for skill validation
│
├── explaining-for-stakeholders/
│   └── SKILL.md
├── explaining-technically/
│   └── SKILL.md
├── communicating-to-stakeholders/
│   └── SKILL.md
├── designing-ts-architectures/
│   ├── SKILL.md
│   └── references/
│       └── architectures.md              ← Deep architecture comparisons
├── teaching-timeseries-dl/
│   └── SKILL.md
├── training-ts-models/
│   ├── SKILL.md
│   └── scripts/
│       ├── set_seeds.py                  ← Reproducibility seed setter
│       └── validate_dataloader.py        ← Temporal leakage checker
├── operating-sagemaker/
│   ├── SKILL.md
│   └── references/
│       ├── training-jobs.md
│       ├── endpoints.md
│       └── pipelines.md
├── optimizing-resources/
│   └── SKILL.md
├── designing-apis/
│   └── SKILL.md
├── improving-developer-ux/
│   └── SKILL.md
├── validating-models/
│   ├── SKILL.md
│   └── scripts/
│       └── run_backtest.py               ← Time series backtesting
├── facilitating-experiments/
│   └── SKILL.md
├── guarding-data-quality/
│   ├── SKILL.md
│   └── scripts/
│       └── check_data_quality.py         ← Data validation checks
└── enforcing-reproducibility/
    ├── SKILL.md
    └── scripts/
        └── pin_environment.py            ← Environment metadata capture
```

## The Forecasting Pipeline

Skills chain in this order for a complete delivery:

```
guarding-data-quality
  → designing-ts-architectures
    → training-ts-models
      → validating-models
        → explaining-for-stakeholders
          → communicating-to-stakeholders

Supporting at every stage:
  facilitating-experiments    → before every experiment
  enforcing-reproducibility   → enforced throughout
  optimizing-resources        → on-demand constraint handling
  operating-sagemaker         → cloud execution
  teaching-timeseries-dl      → upskilling at any stage
```

## Compliance with Best Practices

This framework follows [Anthropic's skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices):

| Best Practice | Implementation |
|---------------|---------------|
| Gerund naming convention | All skills use verb-ing form |
| Name = directory name | Enforced across all 14 skills |
| Third-person descriptions | All descriptions use third person |
| Descriptions include "when to use" + negative triggers | Every description has "Use when..." and "Do not use for..." |
| Descriptions under 1024 characters | All descriptions verified under limit |
| SKILL.md under 500 lines | All files well under 200 lines |
| Progressive disclosure | 2 skills use `references/`, 5 use `scripts/` |
| One-level-deep references | All references link directly from SKILL.md |
| Utility scripts for deterministic operations | 5 bundled scripts for validation, seeding, backtesting |
| Feedback loops | `validating-models` and `guarding-data-quality` implement validate → fix → repeat |
| Evaluations | `evals/evals.json` provides 3 test prompts per key skill |
| Consistent terminology | Standardized across all skills |
| No time-sensitive information | All content is principle-based |

## How to Add a New Skill

1. Create the folder: `ml-team-skills/your-skill-name/`
2. Write `SKILL.md` with YAML frontmatter:
   ```yaml
   ---
   name: your-skill-name          # Must match directory name
   description: >
     [Third-person description of what this skill does].
     Use when [specific triggers and contexts].
     Trigger on "[phrase 1]", "[phrase 2]".
     Do not use for [explicit exclusions with skill references].
   ---
   ```
3. Keep SKILL.md under 500 lines — use `references/` for deep content
4. Add utility scripts to `scripts/` for deterministic operations
5. Add at least 3 test prompts to `evals/evals.json`
6. Update this README's Skill Index table

## License

MIT
