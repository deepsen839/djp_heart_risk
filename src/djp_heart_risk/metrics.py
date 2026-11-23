"""Prometheus metrics for API operations; never label metrics with input features."""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

REQUESTS = Counter("djp_api_requests_total", "API requests", ["method", "endpoint", "status"])
LATENCY = Histogram("djp_api_request_duration_seconds", "API request latency", ["endpoint"])
PREDICTIONS = Counter("djp_model_predictions_total", "Model predictions completed")
MODEL_LOADED = Gauge("djp_model_loaded", "Whether the model artifact is available")

def metrics_payload():
    return generate_latest(), CONTENT_TYPE_LATEST
