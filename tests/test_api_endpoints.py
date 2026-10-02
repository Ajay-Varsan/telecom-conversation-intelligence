import pytest
from starlette.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "uptime_seconds" in data
    assert data["groundedness_rate_pct"] == 100.0


def test_metrics_endpoint():
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "telecom_calls_processed_total" in res.text
    assert "telecom_groundedness_ratio" in res.text


def test_batch_analyze_endpoint():
    payload = {
        "conversation_id": "test_conv_api_1",
        "agent_id": "Alexandra",
        "team_id": "Billing_Retention_Team_Beta",
        "turns": [
            {"turn_id": 1, "speaker": "agent", "text": "Hello, thank you for calling Union Mobile. My name is Alexandra, how can I assist you today?"},
            {"turn_id": 2, "speaker": "client", "text": "Hi, I'm calling to cancel my mobile service. It's just too expensive for me."},
            {"turn_id": 3, "speaker": "agent", "text": "I understand, Velma. Can you please verify your account information?"},
            {"turn_id": 4, "speaker": "client", "text": "Sure, my account number is 1234567890."},
            {"turn_id": 5, "speaker": "agent", "text": "Thank you. Now, I need to inform you that there may be a cancellation fee associated with terminating your service."},
            {"turn_id": 6, "speaker": "client", "text": "No, I just want to cancel. I can't afford it anymore."},
            {"turn_id": 7, "speaker": "agent", "text": "Understood. Thank you for choosing Union Mobile, have a great day. Goodbye."}
        ]
    }
    res = client.post("/analyze/batch", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["conversation_id"] == "test_conv_api_1"
    assert "concise_summary" in data
    assert len(data["call_reasons"]) > 0
    assert data["sentiment_arc"]["start_sentiment"] < 0
    assert data["churn_risk"]["is_risk"] is True
    assert data["qa_score"] > 0
    assert len(data["qa_details"]) == 6


def test_stream_turn_endpoint():
    payload = {
        "conversation_id": "test_conv_api_2",
        "agent_id": "Alexandra",
        "team_id": "Billing_Retention_Team_Beta",
        "current_turn": {
            "turn_id": 2,
            "speaker": "client",
            "text": "Hi, I'm calling to cancel my mobile service. I received a better offer from Mint Mobile."
        },
        "history": [
            {"turn_id": 1, "speaker": "agent", "text": "Hello, thank you for calling Union Mobile. My name is Alexandra."}
        ]
    }
    res = client.post("/analyze/stream-turn", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["conversation_id"] == "test_conv_api_2"
    assert data["turn_sentiment"] < 0
    assert len(data["recommended_actions"]) > 0
    assert data["latency_ms"] < 1000.0  # Real-time latency SLA


def test_qa_config_crud():
    res = client.get("/qa/config")
    assert res.status_code == 200
    cfg = res.json()
    assert "greeting" in cfg["items"]

    # Update item
    update_res = client.post("/qa/config/item/greeting", json={"weight": 12.0})
    assert update_res.status_code == 200
    updated_cfg = update_res.json()["config"]
    assert updated_cfg["items"]["greeting"]["weight"] == 12.0


def test_agent_and_team_rollups():
    # Alexandra was processed in test_batch_analyze_endpoint
    agent_res = client.get("/qa/rollups/agent/Alexandra")
    assert agent_res.status_code == 200
    agent_data = agent_res.json()
    assert agent_data["agent_id"] == "Alexandra"
    assert agent_data["total_calls_analyzed"] >= 1

    team_res = client.get("/qa/rollups/team/Billing_Retention_Team_Beta")
    assert team_res.status_code == 200
    team_data = team_res.json()
    assert team_data["team_id"] == "Billing_Retention_Team_Beta"
    assert team_data["total_calls"] >= 1
    assert len(team_data["agent_rankings"]) >= 1
