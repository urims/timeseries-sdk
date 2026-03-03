---
name: communicating-to-stakeholders
description: Crafts stakeholder-aligned written communications about ML projects — status updates, forecast delivery emails, model change announcements, limitation disclosures, risk communications, and project proposals for non-technical audiences. Use when preparing any written deliverable about ML work for business audiences such as executives, product owners, or clients. Trigger on "write an update for stakeholders", "communicate the results", "announcement email", "present the forecast", "explain the delay", "stakeholder presentation", "project proposal", "communicate model change", "model rollout email", "status report". Do not use for translating individual model outputs — use `explaining-for-stakeholders` instead. Do not use for internal technical documentation.
---

# Communicating to Stakeholders

## Core Principle

Lead with business impact. Stakeholders care about what the model enables them to do better, not the model itself.

## Communication Templates

Select the template that matches the situation:

### Template A: Forecast Delivery

```
Subject: [Metric] Forecast Ready — [Key Headline Number]

[1 sentence: bottom-line forecast result]
[1 sentence: what this means for their planning/decisions]

Key numbers:
  - [Primary forecast figure with range]
  - [Comparison to last period or baseline]
  - [Confidence level in plain language]

What to watch:
  - [1–2 conditions that could change the forecast]

Recommended action:
  - [One specific thing they should do with this information]
```

### Template B: Project Status Update

```
Subject: [Project Name] — [Phase] Update

Progress: [1 sentence on what was accomplished]
Timeline: [On track / adjusted — with reason if adjusted]
Key result: [One concrete metric or milestone]
Next milestone: [What happens next and when]
Needs from you: [Specific ask, or "none right now"]
```

### Template C: Model Change Announcement

```
Subject: [Model/System Name] Update — What Changes for You

What's changing: [1 sentence, business terms]
Why: [Brief justification tied to business value]
What stays the same: [Reassurance about continuity]
What might look different: [Set expectations]
When: [Date and any transition period]
Questions? [Contact/channel]
```

### Template D: Limitation Disclosure

```
Subject: Important Context for [Forecast/Model] Results

What we're sharing: [The forecast/result in question]
What it covers well: [Where you can rely on it]
Where to be cautious: [Specific limitations, framed as conditions]
What we're doing about it: [Concrete improvement plan]
Recommended approach: [How to use results given the limitations]
```

## Tone Rules

- Active voice, present tense where possible
- No hedging without a concrete condition ("results may vary" → "results depend on lead times staying under 3 weeks")
- Numbers always include context (absolute + relative + comparison point)
- Bad news is delivered with a mitigation plan, never as a standalone negative

## Quality Checklist

- [ ] Subject line contains the key takeaway, not a generic label
- [ ] First sentence answers "so what?" for the reader
- [ ] No unexplained technical terms
- [ ] Exactly one recommended action or clear next step
- [ ] Tone is confident but honest about uncertainty
