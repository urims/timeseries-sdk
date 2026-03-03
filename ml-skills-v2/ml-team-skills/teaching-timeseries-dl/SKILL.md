---
name: teaching-timeseries-dl
description: Teaches ML concepts, time series theory, and deep learning principles clearly and progressively to team members at any level. Adapts explanations from beginner-friendly analogies to advanced mathematical formulations based on the learner's background. Use when someone needs to learn, review, or build intuition about ML, DL, or time series concepts. Trigger on "explain how X works", "teach me about", "I don't understand", "what is [concept]", "how does attention work in forecasting", "walk me through", "tutorial on", "learning path for", "what's the intuition behind", or any educational request about ML, DL, or time series concepts. Do not use for making architecture decisions — use `designing-ts-architectures`. Do not use for explaining specific model outputs to reviewers — use `explaining-technically`.
---

# Teaching Timeseries DL

## Teaching Philosophy

- **Spiral learning**: Introduce concepts simply, then revisit with more depth
- **Concrete before abstract**: Ground every theory in a forecasting example first
- **Build on what they know**: Connect new ideas to the learner's existing mental models
- **One concept per explanation**: Don't overwhelm — each response focuses on one core idea

## Level Detection

Gauge the learner's level from context clues, then adapt:

| Level | Signals | Approach |
|-------|---------|----------|
| **Beginner** | "What is a neural network?", no ML vocabulary | Use analogies, avoid math, visual descriptions |
| **Intermediate** | Uses terms like "loss", "epoch", "overfitting" | Explain mechanisms, introduce formulas with intuition |
| **Advanced** | Discusses gradients, attention, specific architectures | Go deep — math, trade-offs, recent papers, edge cases |

If unsure, start at intermediate and adjust based on follow-up questions.

## Explanation Structure

For any concept, follow this progression:

1. **One-sentence definition** in plain language
2. **Concrete example** using time series forecasting (e.g., "Imagine you're predicting weekly ice cream sales...")
3. **How it works** at the appropriate depth for the learner's level
4. **Why it matters** for the specific task at hand
5. **Common misconceptions** or gotchas
6. **Connection to related concepts** they might explore next

## Topic Index

These are the most common teaching requests. For each, prepare to explain at all three levels:

- Stationarity and differencing
- Autocorrelation and partial autocorrelation
- Seasonality (single and multiple)
- Attention mechanisms in time series context
- Encoder-decoder architectures
- Quantile regression and probabilistic forecasting
- Backpropagation through time
- Overfitting vs. underfitting (with time series specific nuances)
- Cross-validation for time series (why random splits fail)
- Transfer learning and foundation models for forecasting

## Quality Checklist

- [ ] Explanation matches the detected learner level
- [ ] At least one concrete forecasting example included
- [ ] No unnecessary complexity (Occam's explanation)
- [ ] Related concepts are signposted for further exploration
- [ ] Misconceptions are addressed proactively
