import time
import re
from typing import List, Tuple
from src.models.schemas import (
    Turn,
    Speaker,
    LiveTurnInput,
    LiveAssistResponse,
    LiveActionRecommendation
)
from src.analytics.sentiment_analyzer import SentimentAnalyzer


class LiveAssistEngine:
    """Provides real-time next-best-actions, compliance alerts, and running sentiment per turn."""

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()

    def process_turn(self, input_data: LiveTurnInput) -> LiveAssistResponse:
        start_time = time.perf_counter()

        current = input_data.current_turn
        history = input_data.history or []
        all_turns = history + [current]

        # 1. Turn Sentiment & Contextual Customer Experience
        turn_sentiment = self.sentiment_analyzer.score_turn_text(current.text)
        sentiment_label = self.sentiment_analyzer.get_label(turn_sentiment)

        # Compute cumulative customer experience sentiment & relationship status across the call
        cust_score, cust_label, cust_state = self.sentiment_analyzer.compute_cumulative_customer_sentiment(all_turns)

        # Calculate running sentiment trend strictly across customer turns
        client_turns = [t for t in all_turns if t.speaker == Speaker.CLIENT]
        client_sentiments = [
            self.sentiment_analyzer.score_turn_text(t.text)
            for t in client_turns
        ]

        if len(client_sentiments) >= 2:
            recent_delta = client_sentiments[-1] - client_sentiments[0]
            if recent_delta >= 0.25:
                trend = "improving"
            elif recent_delta <= -0.25:
                trend = "deteriorating"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # 2. Live Compliance Checks & Next Best Actions
        compliance_alerts: List[str] = []
        recommended_actions: List[LiveActionRecommendation] = []

        curr_text = current.text.lower()
        full_text = " ".join(t.text for t in all_turns).lower()

        # If agent spoke:
        if current.speaker == Speaker.AGENT:
            if ("it's on us" in curr_text or "brand new iphone" in curr_text) and "credit" not in curr_text:
                compliance_alerts.append(
                    "COMPLIANCE VIOLATION: Unauthorized 'free' device promise without mandatory installment agreement disclosure."
                )
                recommended_actions.append(LiveActionRecommendation(
                    action_type="CORRECTIVE_DISCLOSURE",
                    title="Rectify Promotional Terms",
                    recommended_script="To clarify, this device offer requires a 24-month bill credit agreement and standard line activation fee.",
                    urgency="critical",
                    trigger_reason="Agent offered free hardware without qualifying disclosures."
                ))

            if "cancel your service" in curr_text and "fee" not in full_text and "balance" not in full_text:
                compliance_alerts.append(
                    "COMPLIANCE WARNING: Processing cancellation without disclosing early termination fees or return obligations."
                )
                recommended_actions.append(LiveActionRecommendation(
                    action_type="MANDATORY_DISCLOSURE",
                    title="Read Cancellation Disclosure Script",
                    recommended_script="Before final confirmation, please be aware that any leased equipment must be returned within 14 days to avoid non-return fees.",
                    urgency="high",
                    trigger_reason="Cancellation workflow active without fee/equipment disclosure."
                ))

        # If client spoke:
        if current.speaker == Speaker.CLIENT:
            # Client wants to cancel
            if "cancel" in curr_text or "too expensive" in curr_text:
                # Check if identity was verified yet
                if "verify" not in full_text and "account pin" not in full_text and "credit card" not in full_text:
                    recommended_actions.append(LiveActionRecommendation(
                        action_type="VERIFICATION_REQUIRED",
                        title="Execute Identity Authentication (CPNI)",
                        recommended_script="I understand your request and will be glad to assist. First, for your account security, could you please verify your account PIN or the last 4 digits of your card?",
                        urgency="critical",
                        trigger_reason="Federal CPNI Requirement: Identity verification must precede account cancellation."
                    ))
                else:
                    recommended_actions.append(LiveActionRecommendation(
                        action_type="RETENTION_EXPLORATION",
                        title="Explore Cost-Effective Plan Options",
                        recommended_script="I completely understand cost is important. Before closing the account, may I check if switching you to our 2GB plan with unlimited talk & text ($10 less) meets your needs?",
                        urgency="high",
                        trigger_reason="Customer expressed cost dissatisfaction; retention opportunity available."
                    ))

            # Client expresses service, coverage, or signal dissatisfaction
            if any(k in curr_text for k in ["coverage", "signal", "dropped call", "poor reception", "spotty", "slow data", "connectivity", "no service", "trouble getting"]):
                recommended_actions.append(LiveActionRecommendation(
                    action_type="EMPATHY_AND_DIAGNOSTICS",
                    title="Acknowledge Frustration & Gather Device/Location",
                    recommended_script="I am very sorry for the frustration with coverage and signal issues. Could you confirm your device model and the cross-streets or zip code where you experience this?",
                    urgency="high",
                    trigger_reason="Customer reported persistent network degradation or coverage problems."
                ))

            # Client frustrated / de-escalation
            if any(k in curr_text for k in ["frustrat", "ridiculous", "can't believe", "unacceptable", "terrible", "awful", "angry"]):
                recommended_actions.append(LiveActionRecommendation(
                    action_type="DE_ESCALATION",
                    title="De-escalation & Active Listening",
                    recommended_script="I sincerely apologize for the inconvenience and understand this has been frustrating. I am here to help you get this resolved right now.",
                    urgency="critical",
                    trigger_reason="Customer emotional state escalated to high frustration."
                ))

            # Client mentions competitor
            if "mint mobile" in curr_text or "t-mobile" in curr_text or "verizon" in curr_text:
                recommended_actions.append(LiveActionRecommendation(
                    action_type="COMPETITIVE_REBUTTAL",
                    title="Competitor Comparison & Retention Credit",
                    recommended_script="I understand other carriers have attractive introductory promotions. We can offer you our loyalty monthly credit and network priority guarantee if you remain with us.",
                    urgency="medium",
                    trigger_reason="Competitor switching threat detected."
                ))

            # Client agrees or says thank you
            if "take the offer" in curr_text or "sounds good" in curr_text:
                recommended_actions.append(LiveActionRecommendation(
                    action_type="CONFIRMATION_AND_WRAPUP",
                    title="Confirm Promotion Details & Next Steps",
                    recommended_script="Excellent! I have applied that plan change to your account effective immediately. You'll receive a confirmation email shortly. Is there anything else today?",
                    urgency="low",
                    trigger_reason="Customer accepted offer; proceed to clean wrap-up."
                ))

        # Default action if none triggered
        if not recommended_actions:
            recommended_actions.append(LiveActionRecommendation(
                action_type="ACTIVE_LISTENING",
                title="Active Listening & Professional Engagement",
                recommended_script="I see. Please continue, and I will be glad to help resolve this for you.",
                urgency="low",
                trigger_reason="Standard conversational flow (no urgent compliance or retention trigger)."
            ))

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return LiveAssistResponse(
            conversation_id=input_data.conversation_id,
            turn_id=current.turn_id,
            turn_speaker=current.speaker,
            turn_sentiment=turn_sentiment,
            sentiment_label=sentiment_label,
            customer_sentiment=cust_score,
            customer_sentiment_label=cust_label,
            customer_state=cust_state,
            running_sentiment_trend=trend,
            compliance_alerts=compliance_alerts,
            recommended_actions=recommended_actions,
            latency_ms=latency_ms
        )
