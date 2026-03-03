---
name: designing-apis
description: Designs clean, robust, and well-documented APIs for ML model serving, forecasting endpoints, and data pipelines. Covers REST API design, async APIs, batch APIs, request/response schemas, versioning, rate limiting, error handling, and OpenAPI documentation for ML services. Use when designing or reviewing API interfaces that expose ML model functionality to consumers. Trigger on "design an API", "API endpoint", "REST API for the model", "API schema", "request/response format", "API documentation", "FastAPI", "API contract", "versioning", "rate limiting", "expose the model as a service", "prediction endpoint". Do not use for SageMaker endpoint deployment — use `operating-sagemaker`. Do not use for frontend/UI design — use `improving-developer-ux`.
---

# Designing APIs for ML Services

## Design Principles

1. **Contract-first**: Define the schema before implementation
2. **Predictable**: Same input always produces same output format (even if predictions differ)
3. **Self-documenting**: Response includes metadata about the model that generated it
4. **Graceful degradation**: Return partial results with warnings, not hard failures

## Endpoint Design Patterns

### Forecast Request/Response

```json
// POST /v1/forecast
// Request
{
  "series_id": "SKU-12345",
  "horizon": 7,
  "frequency": "daily",
  "include_intervals": true,
  "quantiles": [0.1, 0.5, 0.9]
}

// Response
{
  "series_id": "SKU-12345",
  "model_version": "tft-v2.3",
  "generated_at": "2025-01-15T10:30:00Z",
  "forecast": [
    {"date": "2025-01-16", "point": 142.5, "q10": 128.0, "q90": 157.0},
    {"date": "2025-01-17", "point": 138.2, "q10": 124.1, "q90": 152.3}
  ],
  "metadata": {
    "training_data_through": "2025-01-15",
    "mase_on_validation": 0.87,
    "warnings": []
  }
}
```

### Batch Forecast

```json
// POST /v1/forecast/batch
// Request
{
  "series_ids": ["SKU-001", "SKU-002", "SKU-003"],
  "horizon": 7,
  "callback_url": "https://your-service.com/webhook/forecasts"
}

// Response (immediate)
{
  "job_id": "batch-abc123",
  "status": "queued",
  "estimated_completion": "2025-01-15T10:45:00Z"
}
```

## Error Handling

Return structured errors that help the consumer fix the problem:

```json
{
  "error": {
    "code": "INSUFFICIENT_HISTORY",
    "message": "Series SKU-99999 has 12 data points; minimum required is 52",
    "series_id": "SKU-99999",
    "suggestion": "Provide at least 52 weekly observations, or use the /v1/forecast/cold-start endpoint"
  }
}
```

## Versioning Strategy

- URL versioning: `/v1/forecast`, `/v2/forecast`
- Model version in response metadata (always, even if API version unchanged)
- Deprecation header: `Sunset: Sat, 01 Mar 2025 00:00:00 GMT`
- Maintain backward compatibility within a major version

## API Documentation

Generate OpenAPI spec from code (FastAPI does this automatically). Ensure the spec includes:
- Example requests and responses for every endpoint
- Error codes with descriptions
- Rate limit information
- Authentication requirements

## Quality Checklist

- [ ] Schema defined before implementation
- [ ] All endpoints include model version in response
- [ ] Error responses are structured and actionable
- [ ] Rate limits documented
- [ ] Example requests work against a test environment
- [ ] Versioning strategy documented and consistent
