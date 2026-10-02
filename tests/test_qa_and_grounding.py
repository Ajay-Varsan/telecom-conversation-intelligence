import pytest
from src.models.schemas import Turn, Speaker, QAChecklistConfig
from src.qa.grounding_guardrail import GroundingGuardrail
from src.qa.evaluator import QAEvaluator


@pytest.fixture
def compliant_turns():
    return [
        Turn(turn_id=1, speaker=Speaker.AGENT, text="Hello, thank you for calling Union Mobile. My name is Justin, how can I assist you today?"),
        Turn(turn_id=2, speaker=Speaker.CLIENT, text="Hi Justin, I'm calling to cancel my mobile service. It's just too expensive."),
        Turn(turn_id=3, speaker=Speaker.AGENT, text="I understand your frustration, Vada. Before we proceed, I'll need to verify your account PIN."),
        Turn(turn_id=4, speaker=Speaker.CLIENT, text="Sure, my account PIN is 1234."),
        Turn(turn_id=5, speaker=Speaker.AGENT, text="Thank you. Please be aware that there is a cancellation fee and you must return any equipment."),
        Turn(turn_id=6, speaker=Speaker.CLIENT, text="Okay, that sounds fine. Thanks for your help."),
        Turn(turn_id=7, speaker=Speaker.AGENT, text="You're welcome. Is there anything else I can assist you with? Thank you for choosing Union Mobile, have a great day!")
    ]


@pytest.fixture
def non_compliant_turns():
    return [
        Turn(turn_id=1, speaker=Speaker.AGENT, text="Hello, thank you for calling Union Mobile. My name is Julia."),
        Turn(turn_id=2, speaker=Speaker.CLIENT, text="Hi, I want to cancel my mobile service."),
        # Agent skips identity verification and immediately offers free iPhone without terms
        Turn(turn_id=3, speaker=Speaker.AGENT, text="How about we give you a brand new iPhone with a new line? It's on us. Would you like that?"),
        Turn(turn_id=4, speaker=Speaker.CLIENT, text="Okay, I'll take it."),
        Turn(turn_id=5, speaker=Speaker.AGENT, text="Great, thanks. Bye.")
    ]


def test_grounding_guardrail_verbatim_verification(compliant_turns):
    guardrail = GroundingGuardrail()

    # Exact verbatim quote
    quote = "Before we proceed, I'll need to verify your account PIN."
    is_grounded, turn_id, snippet = guardrail.verify_quote(quote, compliant_turns, Speaker.AGENT)
    assert is_grounded is True
    assert turn_id == 3

    # Hallucinated quote that does not exist
    fake_quote = "I promise we will credit your bill with two hundred dollars immediately."
    is_grounded_fake, turn_fake, _ = guardrail.verify_quote(fake_quote, compliant_turns, Speaker.AGENT)
    assert is_grounded_fake is False
    assert turn_fake is None


def test_qa_evaluator_compliant_call(compliant_turns):
    config = QAChecklistConfig()
    evaluator = QAEvaluator(config)

    score, passed, critical_violation, details = evaluator.evaluate(compliant_turns)

    assert score >= 85.0
    assert passed is True
    assert critical_violation is False

    # Check that all items passed
    for item in details:
        assert item.passed is True
        assert item.is_grounded is True
        if item.quoted_evidence:
            assert len(item.evidence_turn_indices) > 0


def test_qa_evaluator_catches_critical_violations(non_compliant_turns):
    config = QAChecklistConfig()
    evaluator = QAEvaluator(config)

    score, passed, critical_violation, details = evaluator.evaluate(non_compliant_turns)

    assert passed is False
    assert critical_violation is True

    # Check that ID verification failed
    id_item = next(d for d in details if d.item_id == "identity_verification")
    assert id_item.passed is False
    assert id_item.severity == "CRITICAL"

    # Check that prohibited promise was caught
    promise_item = next(d for d in details if d.item_id == "no_prohibited_promises")
    assert promise_item.passed is False
    assert promise_item.severity == "CRITICAL"
    assert "It's on us" in promise_item.quoted_evidence[0]
