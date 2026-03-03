---
name: explaining-for-stakeholders
description: Translates ML model results, forecasts, errors, and decisions into plain business language for non-technical stakeholders. Converts prediction intervals, error metrics, and model behavior into narratives executives can act on. Use when presenting model outputs to business audiences, writing executive summaries of forecasting results, or explaining model limitations without jargon. Trigger on "explain this to stakeholders", "make this non-technical", "business summary", "executive summary", "board presentation", "what does this forecast mean", or when delivering any ML output to a non-ML audience. Do not use for technical peer review — use `explaining-technically` instead. Do not use for drafting emails or status updates — use `communicating-to-stakeholders` instead.
---

# Explaining for Stakeholders

## Goal

Convert ML complexity into clear, confidence-inspiring narratives that drive stakeholder decisions — without sacrificing accuracy.

## Output Structure

Follow this order for every explanation:

### 1. The Bottom Line (2–3 sentences)
State the finding, its business meaning, and the recommended action. Present tense, active voice.

### 2. What We Did (1 short paragraph)
Describe the process in business terms. No algorithm names. Use analogies from the bank below.

### 3. What We Found
Use one of these formats depending on what fits:
- **Traffic light table**: Red / Yellow / Green status per metric
- **Before / After comparison**: Baseline vs model performance
- **Forecast range**: "Sales are expected to rise 12% in Q2, with a likely range of 8%–16%"

### 4. Confidence & Caveats (max 4 bullets)
Be honest about uncertainty. Frame caveats as conditions, not failures: "This holds as long as supply chain lead times stay below 3 weeks."

### 5. What Happens Next
One clear recommended next step for the stakeholder.

---

## Language Translation Table

| Avoid | Use Instead |
|-------|-------------|
| RMSE, MAE, MAPE | "Our predictions are typically within X% of actual values" |
| Hyperparameter tuning | "We adjusted the model's settings to improve accuracy" |
| Overfitting | "The model worked well in testing but may struggle with unusual future patterns" |
| Training / validation split | "We tested predictions on data the model had never seen" |
| Neural network / LSTM / Transformer | "A pattern-learning system trained on historical data" |
| Confidence interval | "The likely range" or "we expect values between X and Y" |
| Feature importance | "The factors that most influenced this forecast" |
| Anomaly | "Unusual pattern" or "unexpected spike/drop" |

## Analogy Bank

- **Time series forecasting** → "Like a weather forecast, but for your business metrics"
- **Model training** → "Like teaching an intern by showing them thousands of past examples"
- **Prediction intervals** → "Like a GPS saying 'arrive in 20–30 min' — a range, not a guarantee"
- **Feature importance** → "Which dials matter most when the model makes a prediction"
- **Model drift** → "Like a recipe that works one season but needs adjusting when ingredients change"

## Quality Checklist

Before delivering, verify:
- [ ] No unexplained acronyms or technical terms
- [ ] Numbers are contextual ("12% more than last year", not just "12%")
- [ ] Uncertainty framed positively ("we're confident within this range")
- [ ] Exactly ONE clear recommended action
- [ ] A smart 12-year-old would understand it
