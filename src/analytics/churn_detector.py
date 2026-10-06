import os
import re
import joblib
from typing import List, Tuple, Optional
from src.models.schemas import Turn, ChurnRisk, ResolutionStatus, Speaker


class ChurnAndResolutionDetector:
    """Evaluates churn risk probability, competitor triggers, and call resolution status using trained statistical ML with rule fallback."""

    COMPETITORS = ["mint mobile", "t-mobile", "verizon", "at&t", "cricket", "boost mobile", "spectrum"]

    def __init__(self, model_path: str = "models/telecom_churn_model.joblib"):
        self.ml_pipeline = None
        self.feature_names = None
        self.coefficients = None
        if os.path.exists(model_path):
            try:
                artifact = joblib.load(model_path)
                if isinstance(artifact, dict) and "pipeline" in artifact:
                    self.ml_pipeline = artifact["pipeline"]
                    self.feature_names = artifact.get("feature_names")
                    self.coefficients = artifact.get("coefficients")
                else:
                    self.ml_pipeline = artifact
            except Exception:
                self.ml_pipeline = None

    def evaluate_churn_risk(self, turns: List[Turn]) -> ChurnRisk:
        if not turns:
            return ChurnRisk(
                is_risk=False,
                risk_score=0.0,
                risk_level="LOW",
                drivers=[],
                competitor_mentioned=None
            )

        all_text = " ".join(t.text for t in turns).lower()
        agent_turns = [t for t in turns if t.speaker == Speaker.AGENT]
        client_turns = [t for t in turns if t.speaker == Speaker.CLIENT]
        agent_text = " ".join(t.text for t in agent_turns).lower()
        client_text = " ".join(t.text for t in client_turns).strip() if client_turns else all_text
        lower_client = client_text.lower()

        drivers = []
        competitor_found: Optional[str] = None

        # 1. Competitor Identification
        for comp in self.COMPETITORS:
            if comp in lower_client:
                competitor_found = comp.title()
                drivers.append(f"Competitor offer mentioned: {competitor_found}")
                break

        # 2. Key Driver Attribution
        has_cancel_intent = bool(re.search(r"\b(cancel|terminat\w*|close my (account|service)|disconnect|port out|port my number)\b", lower_client))
        if has_cancel_intent:
            if "don't want to cancel" not in lower_client and "not looking to cancel" not in lower_client:
                drivers.append("Explicit request to cancel service")

        if re.search(r"\b(dropped calls?|poor reception|slow data|spotty coverage|unreliable|no signal)\b", lower_client):
            drivers.append("Persistent network connectivity / quality dissatisfaction")

        if re.search(r"\b(too expensive|can't afford|cheaper|bill is high|costly|price increase)\b", lower_client):
            drivers.append("Service affordability / cost complaints")

        # 3. Conversational State Analysis: Transfer vs Cancellation vs Genuine Retention
        agent_transferred = bool(re.search(
            r"\b(transfer you to|transfer to our (technical support|cancellation|billing|account management)|transfer your call)\b",
            agent_text
        ))

        customer_doubted_offer = bool(re.search(
            r"\b(not sure if that will solve|don't know if that will help|not sure that will help|not sure if that will work|doesn't solve|won't solve|still want to cancel|rather just cancel)\b",
            lower_client
        ))

        agent_confirmed_cancellation = bool(re.search(
            r"\b(i've processed the cancellation|processed your cancellation|canceled your (mobile )?service|placed a request to cancel your service|we've canceled your|processed the cancellation request)\b",
            agent_text
        )) and not any(k in agent_text for k in ["unable to cancel", "cannot cancel", "can't cancel", "without proper verification", "unable to proceed"])

        explicit_retention_accept = [
            r"\b(i'll take the offer|take that offer|i will take the offer)\b",
            r"\b(i think i'd like to take advantage of that|take advantage of that|take advantage of the deal)\b",
            r"\b(switch me over( to that)?|apply that to (my )?account|go ahead and apply)\b",
            r"\b(i will stay|i'll stay|keep my service)\b",
            r"\b(let's go with that plan|sign me up for that|i'll go with the)\b",
            r"\b(that sounds like a (good|great) deal.*(let's do|sign me|apply))\b"
        ]
        customer_explicitly_accepted = any(re.search(p, lower_client) for p in explicit_retention_accept)
        agent_completed_order = bool(re.search(
            r"\b(process the order for the (new )?(iphone|phone|device)|placed the order|order has been placed|applied the (discount|promo|credit)|switched your plan to|enrolled you in)\b",
            agent_text
        ))

        # True retention: customer accepted or agent finalized order, AND customer did not doubt, AND agent did not transfer or cancel
        accepted_retention = (customer_explicitly_accepted or agent_completed_order) and not customer_doubted_offer and not agent_transferred and not agent_confirmed_cancellation

        if accepted_retention:
            drivers.append("Customer accepted retention offer / plan adjustment / device upgrade")
            if "Explicit request to cancel service" in drivers:
                drivers.remove("Explicit request to cancel service")
                drivers.insert(0, "Initial Churn Intent: Explicit request to cancel service (Successfully Mitigated)")

        if agent_confirmed_cancellation:
            drivers.append("Service cancellation processed / confirmed")

        if agent_transferred and has_cancel_intent and not accepted_retention:
            if customer_doubted_offer:
                drivers.append("Retention proposal declined/doubted ('not sure if that will solve problem')")
            drivers.append("Customer transferred to specialized support queue without on-call resolution")

        unauthenticated_or_disconnected = bool(re.search(
            r"\b(unable to verify|cannot cancel your service without|call got disconnected|got disconnected)\b",
            agent_text
        )) or "( response" in lower_client

        if unauthenticated_or_disconnected and has_cancel_intent and not accepted_retention and not agent_confirmed_cancellation:
            drivers.append("Call disconnected / authentication declined prior to cancellation")

        # 4. Probability Estimation (Trained ML Model with Dialogue Outcome Calibration)
        final_score = None
        if self.ml_pipeline is not None:
            try:
                ml_prob = float(self.ml_pipeline.predict_proba([client_text])[0][1])
                if agent_confirmed_cancellation:
                    final_score = max(0.95, round(ml_prob, 2))
                elif accepted_retention:
                    final_score = min(0.20, round(ml_prob * 0.2, 2))
                elif agent_transferred and has_cancel_intent:
                    final_score = max(0.72, min(0.88, round(ml_prob, 2)))
                elif unauthenticated_or_disconnected and has_cancel_intent:
                    final_score = max(0.80, min(0.95, round(ml_prob, 2)))
                else:
                    final_score = round(ml_prob, 2)
            except Exception:
                final_score = None

        # Fallback to rule-based scoring if ML is unavailable or errored
        if final_score is None:
            risk_score = 0.0
            if has_cancel_intent:
                risk_score += 0.55
            if competitor_found:
                risk_score += 0.25
            if any("network" in d.lower() for d in drivers):
                risk_score += 0.20
            if any("affordability" in d.lower() for d in drivers):
                risk_score += 0.15
            if accepted_retention:
                risk_score = max(0.15, risk_score - 0.45)
            if agent_confirmed_cancellation:
                risk_score = min(1.0, risk_score + 0.35)
            if agent_transferred and has_cancel_intent:
                risk_score = max(0.70, risk_score)
            final_score = min(1.0, max(0.0, round(risk_score, 2)))

        final_score = min(1.0, max(0.0, round(final_score, 2)))

        # Risk level categorization
        if final_score >= 0.75:
            level = "CRITICAL"
        elif final_score >= 0.50:
            level = "HIGH"
        elif final_score >= 0.25:
            level = "MEDIUM"
        else:
            level = "LOW"

        is_risk = final_score >= 0.35

        return ChurnRisk(
            is_risk=is_risk,
            risk_score=final_score,
            risk_level=level,
            drivers=drivers,
            competitor_mentioned=competitor_found
        )

    def evaluate_resolution(self, turns: List[Turn]) -> ResolutionStatus:
        if not turns:
            return ResolutionStatus(status="UNRESOLVED", confidence=0.5, explanation="No conversation turns available.")

        all_agent_text = " ".join(t.text for t in turns if t.speaker == Speaker.AGENT).lower()
        all_client_text = " ".join(t.text for t in turns if t.speaker == Speaker.CLIENT).lower()
        last_turns = turns[-5:]
        last_agent_text = " ".join(t.text for t in last_turns if t.speaker == Speaker.AGENT).lower()
        last_client_text = " ".join(t.text for t in last_turns if t.speaker == Speaker.CLIENT).lower()

        # 1. Escalation / Departmental Transfer
        if re.search(r"\b(transfer you to|transfer to our|transfer your call|escalat\w*|engineering team|supervisor|tier 2)\b", all_agent_text):
            queue = "Technical Support" if "technical support" in all_agent_text else (
                "Cancellation Department" if "cancellation department" in all_agent_text else (
                    "Billing / Account Management" if ("billing" in all_agent_text or "account management" in all_agent_text) else "Specialized Support"
                )
            )
            return ResolutionStatus(
                status="ESCALATED",
                confidence=0.92,
                explanation=f"Customer transferred to {queue} queue; issue in-progress and unfinalized on current call."
            )

        # 2. Authentication Failure / Refusal
        if re.search(r"\b(unable to verify|cannot cancel your service without|unable to proceed with(out)? proper verification|unable to locate your account)\b", all_agent_text):
            return ResolutionStatus(
                status="UNRESOLVED",
                confidence=0.92,
                explanation="Call concluded without account transaction due to customer identity authentication failure."
            )

        # 3. Call Disconnection (unless customer reconnected and finished satisfactorily)
        is_disconnected = bool(re.search(r"\b(call got disconnected|got disconnected|are you still on the line)\b", all_agent_text)) or "( response" in all_client_text
        reconnected_and_resolved = bool(re.search(r"\b(all set|mistake on my end|thank you again|no, i'm all set)\b", last_client_text)) and not bool(re.search(r"\b(are you still on the line|call got disconnected)\b", last_agent_text))
        if is_disconnected and not reconnected_and_resolved:
            return ResolutionStatus(
                status="UNRESOLVED",
                confidence=0.90,
                explanation="Call disconnected unexpectedly prior to transaction completion."
            )

        # 4. Confirmed Cancellation Execution
        if re.search(r"\b(i've processed the cancellation|processed your cancellation|canceled your (mobile )?service|placed a request to cancel your service|we've canceled your|processed the cancellation request)\b", all_agent_text):
            return ResolutionStatus(
                status="RESOLVED",
                confidence=0.95,
                explanation="Caller's service cancellation request was fully processed and confirmed by the agent."
            )

        # 5. Confirmed Retention / Upgrade Execution
        if re.search(r"\b(process the order for the (new )?(iphone|phone|device)|placed the order|order has been placed|applied the (discount|promo|credit)|switched your plan to|enrolled you in)\b", all_agent_text):
            return ResolutionStatus(
                status="RESOLVED",
                confidence=0.94,
                explanation="Retention offer or plan adjustment was successfully processed and confirmed on account."
            )

        # 6. Standard Satisfactory Inquiry Closure
        if re.search(r"\b(no, that's all|thanks for your help|thank you\. goodbye|all set)\b", last_client_text):
            return ResolutionStatus(
                status="RESOLVED",
                confidence=0.88,
                explanation="Customer confirmed no further assistance needed and call closed satisfactorily."
            )

        return ResolutionStatus(
            status="PENDING_CUSTOMER_ACTION",
            confidence=0.75,
            explanation="Call ended without explicit resolution or further customer action is required."
        )
