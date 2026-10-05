from dataclasses import dataclass, field
from typing import List, Optional
import re
from src.models.schemas import Turn, Speaker


@dataclass
class DialogueState:
    """Stateful representation of an active customer contact center dialogue."""
    current_phase: str = "GREETING"
    # Phases: GREETING, AUTHENTICATION, DISCOVERY, NEGOTIATION, DISCLOSURE, RESOLUTION, CLOSING

    # Authentication & Security
    is_authenticated: bool = False
    auth_turn_id: Optional[int] = None

    # Intent & Competitor Context
    primary_intent: str = "GENERAL"
    # Intents: CANCELLATION, BILLING_DISPUTE, NETWORK_DEGRADATION, PLAN_CHANGE, GENERAL
    competitor_detected: Optional[str] = None
    competitor_offer_mentioned: bool = False

    # Retention Lifecycle
    retention_offered: bool = False
    retention_accepted: bool = False
    cancellation_confirmed: bool = False

    # Compliance Tracking
    mandated_disclosures_read: bool = False
    prohibited_promises_flagged: bool = False

    # Emotional State
    customer_escalated: bool = False
    customer_state_description: str = "In Progress"


class DialogueStateTracker:
    """Tracks chronological dialogue transitions across turns to eliminate stateless heuristic errors.
    
    Resolves:
    1. CPNI Re-trigger Bug: Suppresses authentication alerts once account is opened/verified.
    2. Competitor Praise Inversion: Routes competitor promotions to competitive rebuttal actions.
    3. Premature Closure & Disclosure Gaps: Enforces disclosure checklists upon final decision.
    """

    COMPETITORS = ["mint mobile", "t-mobile", "verizon", "at&t", "cricket", "boost mobile", "xfinity"]

    def track_state(self, turns: List[Turn]) -> DialogueState:
        state = DialogueState()

        if not turns:
            return state

        agent_requested_pin = False

        for t in turns:
            text = t.text.lower()
            speaker = t.speaker

            # 1. Track Authentication / CPNI State
            if not state.is_authenticated:
                # Check if agent requested PIN/verification
                if speaker == Speaker.AGENT and any(k in text for k in [
                    "pin", "verify", "verification", "security code", "social security", "last 4", "billing address"
                ]):
                    agent_requested_pin = True

                # Direct pin/verification language or masked PIN tokens
                has_direct_pin = any(k in text for k in [
                    "pin is", "my pin", "4 digits", "verify your account", "security pin",
                    "pin number is", "account pin"
                ])
                has_masked_pin = bool(re.search(r"(\*{3,}|\[redacted\]|\[pin\])", text))
                has_numeric_pin = bool(re.search(r"\b\d{4,6}\b", text)) and ("pin" in text or agent_requested_pin)

                # Client responding to PIN request
                if speaker == Speaker.CLIENT and (
                    has_direct_pin
                    or has_masked_pin
                    or has_numeric_pin
                    or (agent_requested_pin and any(k in text for k in ["sure, it's", "sure it's", "it is", "it's", "here you go"]))
                ):
                    state.is_authenticated = True
                    state.auth_turn_id = t.turn_id

                # Agent account lookup confirmations
                elif speaker == Speaker.AGENT and any(k in text for k in [
                    "located your account", "pulled up your account", "pulling up your account",
                    "looking at your account", "found your account", "currently on our",
                    "thank you for verifying", "got your account up", "confirmed your account",
                    "confirm your account", "verified your account", "verify your account",
                    "account is verified", "account has been verified", "account is confirmed",
                    "thank you for confirming", "thank you for providing that", "thanks for verifying",
                    "thanks for confirming", "authenticated your account", "access your account",
                    "got you verified", "into your account", "confirmed your details"
                ]):
                    state.is_authenticated = True
                    state.auth_turn_id = t.turn_id

            # 2. Track Primary Intent
            if speaker == Speaker.CLIENT:
                if any(k in text for k in ["cancel", "port out", "close my account", "switch carriers"]):
                    state.primary_intent = "CANCELLATION"
                elif any(k in text for k in ["coverage", "signal", "dropped call", "slow data", "poor reception", "reception"]):
                    if state.primary_intent == "GENERAL":
                        state.primary_intent = "NETWORK_DEGRADATION"
                elif any(k in text for k in ["overcharged", "bill is high", "unexpected charge", "charge on my bill"]):
                    if state.primary_intent == "GENERAL":
                        state.primary_intent = "BILLING_DISPUTE"

            # 3. Track Competitor Threats
            for comp in self.COMPETITORS:
                if comp in text:
                    state.competitor_detected = comp.title()
                    if any(k in text for k in ["free phone", "better offer", "better deal", "$15", "cheaper"]):
                        state.competitor_offer_mentioned = True
                    break

            # 4. Track Retention & Resolution
            if speaker == Speaker.AGENT:
                if any(k in text for k in ["loyalty discount", "$10 off", "2gb plan", "credit", "switch you to our", "promotional discount"]):
                    state.retention_offered = True

            if speaker == Speaker.CLIENT:
                if any(k in text for k in ["just cancel it", "go ahead and cancel", "process the cancellation", "i've made up my mind", "no thanks, i've made", "please cancel it"]):
                    state.cancellation_confirmed = True
                    state.retention_accepted = False
                elif state.retention_offered and any(k in text for k in ["take the offer", "take that offer", "switch me over to that", "sounds good", "i'll stay", "that's a deal", "it's a deal", "sounds like a deal", "deal!"]):
                    state.retention_accepted = True
                    state.cancellation_confirmed = False

            # 5. Track Disclosures & Compliance
            if speaker == Speaker.AGENT:
                if any(k in text for k in ["cancellation fee", "14 days", "return", "equipment", "installment agreement", "24-month"]):
                    state.mandated_disclosures_read = True
                if ("it's on us" in text or "brand new iphone" in text) and "installment" not in text and "credit" not in text:
                    state.prohibited_promises_flagged = True

            # 6. Track Escalation
            if speaker == Speaker.CLIENT:
                if any(k in text for k in ["frustrat", "ridiculous", "can't believe", "unacceptable", "terrible", "awful", "angry", "manager"]):
                    state.customer_escalated = True

        # Phase determination based on tracked state and turn count
        last_turn = turns[-1]
        last_text = last_turn.text.lower()

        if not state.is_authenticated and state.primary_intent == "CANCELLATION":
            state.current_phase = "AUTHENTICATION"
        elif len(turns) <= 1 and not state.cancellation_confirmed:
            state.current_phase = "GREETING"
        elif state.retention_accepted:
            state.current_phase = "RESOLUTION"
        elif state.cancellation_confirmed:
            if state.mandated_disclosures_read:
                state.current_phase = "CLOSING"
            else:
                state.current_phase = "DISCLOSURE"
        elif state.retention_offered or state.competitor_detected:
            state.current_phase = "NEGOTIATION"
        elif any(k in last_text for k in ["anything else", "thank you for calling", "have a great day"]):
            state.current_phase = "CLOSING"
        else:
            state.current_phase = "DISCOVERY"

        # Customer state description synthesis
        if state.retention_accepted:
            state.customer_state_description = "Retained / Offer Accepted"
        elif state.cancellation_confirmed:
            state.customer_state_description = "Churned / Cancellation Confirmed"
        elif state.competitor_detected:
            state.customer_state_description = f"Churn Threat / Porting to {state.competitor_detected}"
        elif state.customer_escalated:
            state.customer_state_description = "Escalated / High Frustration"
        elif not state.is_authenticated and state.primary_intent == "CANCELLATION":
            state.customer_state_description = "Seeking Cancellation / Unverified"
        elif state.is_authenticated and state.primary_intent == "CANCELLATION":
            state.customer_state_description = "Cancellation Exploration / Authenticated"
        elif state.primary_intent == "NETWORK_DEGRADATION":
            state.customer_state_description = "Network Degradation Complaint"
        else:
            state.customer_state_description = "Active Engagement"

        return state
