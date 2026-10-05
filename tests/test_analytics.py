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


def test_dialogue_state_tracker_phase_awareness():
    from src.analytics.dialogue_state_tracker import DialogueStateTracker

    tracker = DialogueStateTracker()

    turns = [
        Turn(turn_id=1, speaker=Speaker.AGENT, text="Thank you for calling Union Mobile. My name is Julia."),
        Turn(turn_id=2, speaker=Speaker.CLIENT, text="Hi Julia, I want to cancel my account."),
        Turn(turn_id=3, speaker=Speaker.AGENT, text="I can help with that, could you please verify your account PIN?"),
        Turn(turn_id=4, speaker=Speaker.CLIENT, text="Sure, my PIN is 4821."),
        Turn(turn_id=5, speaker=Speaker.AGENT, text="Thanks, I've located your account. You are on our 5GB plan."),
        Turn(turn_id=6, speaker=Speaker.CLIENT, text="Mint Mobile has a cheaper deal with a free phone so I want to switch."),
        Turn(turn_id=7, speaker=Speaker.AGENT, text="What if I offer you our loyalty discount of $10 off?"),
        Turn(turn_id=8, speaker=Speaker.CLIENT, text="No thanks, please cancel it."),
    ]

    # After turn 2: unauthenticated cancellation request
    state_t2 = tracker.track_state(turns[:2])
    assert state_t2.is_authenticated is False
    assert state_t2.current_phase == "AUTHENTICATION"

    # After turn 4: authenticated
    state_t4 = tracker.track_state(turns[:4])
    assert state_t4.is_authenticated is True

    # After turn 6: competitor detected, negotiation phase
    state_t6 = tracker.track_state(turns[:6])
    assert state_t6.is_authenticated is True
    assert state_t6.competitor_detected == "Mint Mobile"
    assert state_t6.current_phase == "NEGOTIATION"

    # After turn 8: cancellation confirmed
    state_t8 = tracker.track_state(turns)
    assert state_t8.cancellation_confirmed is True
    assert state_t8.retention_accepted is False


def test_live_assist_unauthenticated_cancellation_intervention():
    from src.analytics.live_assist import LiveAssistEngine, LiveTurnInput

    engine = LiveAssistEngine()

    turns = [
        Turn(turn_id=1, speaker=Speaker.CLIENT, text="Hello, I'm calling to cancel my mobile service with Union Mobile."),
        Turn(turn_id=2, speaker=Speaker.AGENT, text="Hi Tresa, sorry to hear that you're considering canceling your service. Can you tell me a little bit more about why you're looking to cancel?"),
        Turn(turn_id=3, speaker=Speaker.CLIENT, text="Well, I just don't have good coverage in my area."),
        Turn(turn_id=4, speaker=Speaker.AGENT, text="I understand. However, I can certainly assist you with the cancellation process."),
        Turn(turn_id=5, speaker=Speaker.CLIENT, text="That's fine. Can you just cancel my service now?"),
        Turn(turn_id=6, speaker=Speaker.AGENT, text="Of course, Tresa. Before we proceed, I just want to make sure that you're aware that canceling your service will mean that you'll no longer be able to use your phone number."),
    ]

    # Turn 5: Client asks to cancel, unauthenticated -> CPNI required
    res5 = engine.process_turn(LiveTurnInput(conversation_id="conv_tresa", current_turn=turns[4], history=turns[:4]))
    assert res5.recommended_actions[0].action_type == "VERIFICATION_REQUIRED"
    assert res5.recommended_actions[0].urgency == "critical"

    # Turn 6: Agent proceeds with cancellation without authenticating -> MUST flag violation and HALT
    res6 = engine.process_turn(LiveTurnInput(conversation_id="conv_tresa", current_turn=turns[5], history=turns[:5]))
    assert len(res6.compliance_alerts) > 0
    assert "CRITICAL COMPLIANCE VIOLATION" in res6.compliance_alerts[0]
    assert res6.recommended_actions[0].action_type == "CRITICAL_AUTHENTICATION_INTERVENTION"
    assert res6.recommended_actions[0].urgency == "critical"


def test_masked_pin_and_account_confirmation_suppresses_cpni():
    from src.analytics.live_assist import LiveAssistEngine, LiveTurnInput

    engine = LiveAssistEngine()

    transcript = [
        Turn(turn_id=1, speaker=Speaker.CLIENT, text="I want to cancel my account and switch to Mint Mobile."),
        Turn(turn_id=2, speaker=Speaker.AGENT, text="Can you please confirm your account PIN for me?"),
        Turn(turn_id=3, speaker=Speaker.CLIENT, text="Sure, it's *******."),
        Turn(turn_id=4, speaker=Speaker.AGENT, text="Great, thank you. Alright, I've confirmed your account. Is there anything specific you'd like to know about the transition process?"),
        Turn(turn_id=5, speaker=Speaker.CLIENT, text="Mint Mobile offered me a free phone."),
        Turn(turn_id=6, speaker=Speaker.AGENT, text="I understand.")
    ]

    # Process Turn 6 (Agent response after customer authenticated and mentioned Mint Mobile)
    res6 = engine.process_turn(LiveTurnInput(
        conversation_id="conv_masked_pin",
        current_turn=transcript[5],
        history=transcript[:5]
    ))

    # CPNI verification MUST NOT be requested
    action_types = [a.action_type for a in res6.recommended_actions]
    assert "VERIFICATION_REQUIRED" not in action_types
    assert "CRITICAL_AUTHENTICATION_INTERVENTION" not in action_types

    # Should offer competitive rebuttal for Mint Mobile
    assert any("COMPETITIVE_REBUTTAL" in a or "RETENTION" in a for a in action_types)



