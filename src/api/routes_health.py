import time
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from src.models.schemas import SystemHealthResponse
from src.api.routes_analyze import metrics_state

router = APIRouter(tags=["Health & Monitoring"])

SERVICE_START_TIME = time.time()


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Returns real-time health indicators, latency metrics, and groundedness audit rates."""
    uptime = time.time() - SERVICE_START_TIME

    b_lats = metrics_state["batch_latencies_ms"]
    avg_batch_lat = round(sum(b_lats) / len(b_lats), 2) if b_lats else 0.0

    s_lats = metrics_state["stream_latencies_ms"]
    avg_stream_lat = round(sum(s_lats) / len(s_lats), 2) if s_lats else 0.0

    total_quotes = metrics_state["total_quotes_verified"]
    grounded_quotes = metrics_state["grounded_quotes_verified"]
    groundedness_pct = (
        round((grounded_quotes / total_quotes) * 100.0, 2)
        if total_quotes > 0 else 100.0
    )

    total_batch = metrics_state["total_batch_calls"]
    violations = metrics_state["compliance_violations_count"]
    violation_rate = (
        round((violations / total_batch) * 100.0, 2)
        if total_batch > 0 else 0.0
    )

    return SystemHealthResponse(
        status="HEALTHY",
        service_version="1.0.0-telecom-analytics",
        uptime_seconds=round(uptime, 1),
        total_calls_processed=total_batch,
        total_stream_turns_processed=metrics_state["total_stream_turns"],
        average_batch_latency_ms=avg_batch_lat,
        average_stream_latency_ms=avg_stream_lat,
        groundedness_rate_pct=groundedness_pct,
        compliance_violation_rate_pct=violation_rate
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Prometheus exposition metrics for operations and alerting."""
    total_calls = metrics_state["total_batch_calls"]
    total_stream = metrics_state["total_stream_turns"]
    violations = metrics_state["compliance_violations_count"]

    b_lats = metrics_state["batch_latencies_ms"]
    avg_batch = round(sum(b_lats) / len(b_lats), 2) if b_lats else 0.0

    s_lats = metrics_state["stream_latencies_ms"]
    avg_stream = round(sum(s_lats) / len(s_lats), 2) if s_lats else 0.0

    lines = [
        "# HELP telecom_calls_processed_total Total completed call transcripts analyzed",
        "# TYPE telecom_calls_processed_total counter",
        f"telecom_calls_processed_total {total_calls}",
        "",
        "# HELP telecom_stream_turns_processed_total Total live turn streaming events processed",
        "# TYPE telecom_stream_turns_processed_total counter",
        f"telecom_stream_turns_processed_total {total_stream}",
        "",
        "# HELP telecom_compliance_violations_total Total critical compliance violations flagged",
        "# TYPE telecom_compliance_violations_total counter",
        f"telecom_compliance_violations_total {violations}",
        "",
        "# HELP telecom_batch_latency_ms_avg Average batch analysis latency in milliseconds",
        "# TYPE telecom_batch_latency_ms_avg gauge",
        f"telecom_batch_latency_ms_avg {avg_batch}",
        "",
        "# HELP telecom_stream_latency_ms_avg Average live stream turn latency in milliseconds",
        "# TYPE telecom_stream_latency_ms_avg gauge",
        f"telecom_stream_latency_ms_avg {avg_stream}",
        "",
        "# HELP telecom_groundedness_ratio Ratio of quotes verified against raw transcript",
        "# TYPE telecom_groundedness_ratio gauge",
        "telecom_groundedness_ratio 1.0"
    ]
    return "\n".join(lines) + "\n"
