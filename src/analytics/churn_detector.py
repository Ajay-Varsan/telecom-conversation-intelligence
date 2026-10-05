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
        client_turns = [t for t in turns if t.speaker == Speaker.CLIENT]
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
        if re.search(r"\b(cancel|terminat|close my account|disconnect|port out|port my number)\b", lower_client):
            if "don't want to cancel" not in lower_client and "not looking to cancel" not in lower_client:
                drivers.append("Explicit request to cancel service")

        if re.search(r"\b(dropped calls?|poor reception|slow data|spotty coverage|unreliable|no signal)\b", lower_client):
            drivers.append("Persistent network connectivity / quality dissatisfaction")

        if re.search(r"\b(too expensive|can't afford|cheaper|bill is high|costly|price increase)\b", lower_client):
            drivers.append("Service affordability / cost complaints")

        accepted_retention = False
        if re.search(r"\b(i'll take the offer|take that offer|sounds good\. i'll take|switch me over to that plan|that sounds great|i will stay|i'll stay|keep my service)\b", lower_client):
            accepted_retention = True
            drivers.append("Customer tentatively accepted retention offer / plan adjustment")

        if re.search(r"\b(go ahead and cancel|i've canceled your service|process the cancellation|cancellation process)\b", all_text) and not accepted_retention:
            drivers.append("Service cancellation processed / confirmed")

        # 3. Probability Estimation (Trained ML Model vs Rule Fallback)
        final_score = None
        if self.ml_pipeline is not None:
            try:
                ml_prob = float(self.ml_pipeline.predict_proba([client_text])[0][1])
                if accepted_retention:
                    final_score = min(0.20, ml_prob)
                else:
                    final_score = round(ml_prob, 2)
            except Exception:
                final_score = None

        # Fallback to rule-based scoring if ML is unavailable or errored
        if final_score is None:
            risk_score = 0.0
            if re.search(r"\b(cancel my (mobile )?service|terminate my service|close my account)\b", lower_client):
                risk_score += 0.55
            if competitor_found:
                risk_score += 0.25
            if any("network" in d.lower() for d in drivers):
                risk_score += 0.20
            if any("affordability" in d.lower() for d in drivers):
                risk_score += 0.15
            if accepted_retention:
                risk_score = max(0.15, risk_score - 0.45)
            if any("confirmed" in d.lower() for d in drivers) and not accepted_retention:
                risk_score = min(1.0, risk_score + 0.25)
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
