import pytest
from src.models.schemas import Turn, Speaker
from src.analytics.sentiment_analyzer import SentimentAnalyzer
from src.analytics.reason_classifier import ReasonClassifier
from src.analytics.churn_detector import ChurnAndResolutionDetector
from src.analytics.live_assist import LiveAssistEngine, LiveTurnInput


@pytest.fixture
def sample_turns():
    return [
        Turn(turn_id=1, speaker=Speaker.AGENT, text="Hello, thank you for calling Union Mobile. My name is Justin."),
        Turn(turn_id=2, speaker=Speaker.CLIENT, text="Hi Justin, I'm calling to cancel my mobile service. It's just too expensive."),
        Turn(turn_id=3, speaker=Speaker.AGENT, text="I understand, Vada. Can you please provide your account PIN?"),
        Turn(turn_id=4, speaker=Speaker.CLIENT, text="Sure, my account PIN is 1234."),
        Turn(turn_id=5, speaker=Speaker.AGENT, text="Thank you. Before we proceed, you'll need to pay any balance and return equipment."),
        Turn(turn_id=6, speaker=Speaker.CLIENT, text="Okay, thanks for your help, Justin."),
        Turn(turn_id=7, speaker=Speaker.AGENT, text="You're welcome. Have a great day and thank you for choosing Union Mobile.")
    ]


def test_sentiment_scoring(sample_turns):
    analyzer = SentimentAnalyzer()
    
    # Negative client turn
    neg_score = analyzer.score_turn_text("I'm calling to cancel, it's just too expensive.")
    assert neg_score < 0
    assert analyzer.get_label(neg_score) == "negative"

    # Positive client turn
    pos_score = analyzer.score_turn_text("Thank you so much, that sounds great and very helpful!")
    assert pos_score > 0
    assert analyzer.get_label(pos_score) == "positive"

    # Sentiment Arc
    arc = analyzer.compute_sentiment_arc(sample_turns)
    assert arc.start_sentiment < 0
    assert len(arc.turn_sentiments) > 0


def test_reason_classification(sample_turns):
    classifier = ReasonClassifier()
    reasons = classifier.classify(sample_turns)
    labels = [r.label for r in reasons]

    assert "Service Cancellation / Churn Threat" in labels
    assert "Pricing & Cost Dissatisfaction" in labels
    
    # Check evidence turn attribution
    cancel_reason = next(r for r in reasons if r.label == "Service Cancellation / Churn Threat")
    assert 2 in cancel_reason.evidence_turns


def test_churn_and_resolution(sample_turns):
    detector = ChurnAndResolutionDetector()
    churn = detector.evaluate_churn_risk(sample_turns)
    resolution = detector.evaluate_resolution(sample_turns)

    assert churn.is_risk is True
    assert churn.risk_score > 0.4
    assert "Explicit request to cancel service" in churn.drivers

    assert resolution.status in ["RESOLVED", "PENDING_CUSTOMER_ACTION"]


def test_live_turn_streaming(sample_turns):
    engine = LiveAssistEngine()
    
    # Turn 2: Customer asks to cancel
    live_input = LiveTurnInput(
        conversation_id="conv_test_1",
        agent_id="Justin",
        team_id="Retention_Team_Alpha",
        current_turn=sample_turns[1],
        history=[sample_turns[0]]
    )
    res = engine.process_turn(live_input)

    assert res.turn_sentiment < 0
    assert len(res.recommended_actions) > 0
    # Should recommend verification or retention exploration
    action_types = [a.action_type for a in res.recommended_actions]
    assert any("VERIFICATION" in a or "RETENTION" in a for a in action_types)
    assert res.latency_ms >= 0.0
