import re
from typing import List, Tuple, Optional
from src.models.schemas import Turn, ChurnRisk, ResolutionStatus, Speaker


class ChurnAndResolutionDetector:
    """Evaluates churn risk probability, competitor triggers, and call resolution status."""

    COMPETITORS = ["mint mobile", "t-mobile", "verizon", "at&t", "cricket", "boost mobile", "spectrum"]

    def evaluate_churn_risk(self, turns: List[Turn]) -> ChurnRisk:
        risk_score = 0.0
        drivers = []
        competitor_found: Optional[str] = None

        all_text = " ".join(t.text for t in turns).lower()
        client_text = " ".join(t.text for t in turns if t.speaker == Speaker.CLIENT).lower()

        # Check explicit cancellation request
        if re.search(r"\b(cancel my (mobile )?service|terminate my service|close my account)\b", client_text):
            risk_score += 0.55
            drivers.append("Explicit request to cancel service")

        # Check competitor mention
        for comp in self.COMPETITORS:
            if comp in client_text:
                risk_score += 0.25
                competitor_found = comp.title()
                drivers.append(f"Competitor offer mentioned: {competitor_found}")
                break

        # Check service quality complaints
        if re.search(r"\b(dropped calls?|poor reception|slow data|spotty coverage|unreliable)\b", client_text):
            risk_score += 0.20
            drivers.append("Persistent network connectivity / quality dissatisfaction")

        # Check price complaints
        if re.search(r"\b(too expensive|can't afford|cheaper)\b", client_text):
            risk_score += 0.15
            drivers.append("Service affordability / cost complaints")

        # Check customer acceptance of retention offer (mitigating factor)
        accepted_retention = False
        if re.search(r"\b(i'll take the offer|sounds good\. i'll take|switch me over to that plan|that sounds great)\b", client_text):
            accepted_retention = True
            risk_score = max(0.15, risk_score - 0.45)
            drivers.append("Customer tentatively accepted retention offer / plan adjustment")

        # Check final confirmed cancellation (amplifying factor)
        if re.search(r"\b(go ahead and cancel|i've canceled your service|process the cancellation|cancellation process)\b", all_text) and not accepted_retention:
            risk_score = min(1.0, risk_score + 0.25)
            drivers.append("Service cancellation processed / confirmed")

        final_score = min(1.0, max(0.0, round(risk_score, 2)))

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

        last_turns = turns[-5:]
        last_agent_text = " ".join(t.text for t in last_turns if t.speaker == Speaker.AGENT).lower()
        last_client_text = " ".join(t.text for t in last_turns if t.speaker == Speaker.CLIENT).lower()
        full_client_text = " ".join(t.text for t in turns if t.speaker == Speaker.CLIENT).lower()

        # Check escalation
        if re.search(r"\b(escalat\w*|engineering team|supervisor|tier 2)\b", last_agent_text):
            return ResolutionStatus(
                status="ESCALATED",
                confidence=0.92,
                explanation="Issue escalated to specialized engineering / supervisor queue for back-office resolution."
            )

        # Check identity verification failure / hangup
        if re.search(r"\bunable to (verify|proceed with(out)? proper verification|locate your account)\b", last_agent_text):
            if "( response" in last_client_text or "still on the line" in last_agent_text or "ridiculous" in full_client_text:
                return ResolutionStatus(
                    status="UNRESOLVED",
                    confidence=0.90,
                    explanation="Call disconnected or unresolved due to customer identification authentication failure."
                )

        # Check successful completion / closure
        if re.search(r"\b(process a refund|cancel your service|confirmation email|all set|work something out|take the offer)\b", last_agent_text + " " + last_client_text):
            return ResolutionStatus(
                status="RESOLVED",
                confidence=0.94,
                explanation="Caller's primary request (plan change, retention deal, or cancellation workflow) was fully executed."
            )

        # Check standard resolution signoffs
        if re.search(r"\b(no, that's all|thanks for your help|thank you\. goodbye)\b", last_client_text):
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
