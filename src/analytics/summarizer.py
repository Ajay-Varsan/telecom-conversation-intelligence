import re
from typing import List, Tuple
from src.models.schemas import Turn, Speaker, ChurnRisk, ResolutionStatus


class ConversationSummarizer:
    """Generates concise executive summaries and actionable post-call follow-ups."""

    def summarize(
        self,
        conversation_id: str,
        turns: List[Turn],
        reasons: List[str],
        churn: ChurnRisk,
        resolution: ResolutionStatus
    ) -> Tuple[str, List[str]]:
        if not turns:
            return "Empty conversation transcript.", []

        client_turns = [t for t in turns if t.speaker == Speaker.CLIENT]
        agent_turns = [t for t in turns if t.speaker == Speaker.AGENT]

        first_client_issue = client_turns[0].text if client_turns else "inquiry"
        all_agent_text = " ".join(t.text for t in agent_turns).lower()
        all_client_text = " ".join(t.text for t in client_turns).lower()

        # Build Intent
        primary_reason = reasons[0] if reasons else "General Service"
        summary_parts = []
        summary_parts.append(f"Customer initiated contact regarding {primary_reason.lower()} ('{first_client_issue[:90]}...').")

        # Build Action
        actions_taken = []
        if "unable to verify" in all_agent_text:
            actions_taken.append("agent attempted identity verification which could not be validated")
        elif "verify your identity" in all_agent_text or "account pin" in all_agent_text:
            actions_taken.append("agent authenticated customer identity")

        if "iphone" in all_agent_text or "deal that might change your mind" in all_agent_text or "2gb plan" in all_agent_text:
            actions_taken.append("agent presented retention options/promotional plan adjustments")

        if "cancellation fee" in all_agent_text or "outstanding balance" in all_agent_text:
            actions_taken.append("disclosed required cancellation fees and account balance policies")

        if "escalate" in all_agent_text or "engineering team" in all_agent_text:
            actions_taken.append("escalated network ticket to engineering")

        if "cancel your service" in all_agent_text and "go ahead" in all_agent_text:
            actions_taken.append("processed service termination workflow")

        if actions_taken:
            summary_parts.append(f"Agent response: {'; '.join(actions_taken)}.")
        else:
            summary_parts.append("Agent provided general troubleshooting and customer support.")

        # Build Outcome
        summary_parts.append(f"Outcome: {resolution.status} - {resolution.explanation}")

        concise_summary = " ".join(summary_parts)

        # Build Follow-up actions
        follow_ups: List[str] = []

        if "return any equipment" in all_agent_text or "equipment that you may have" in all_agent_text:
            follow_ups.append("Dispatch equipment return prepaid shipping box & return instructions to customer's address on file.")

        if "refund" in all_agent_text or "process a refund" in all_agent_text:
            follow_ups.append("Execute billing credit / prorated refund to customer's original payment method within 3 business days.")

        if resolution.status == "ESCALATED" or "engineering team" in all_agent_text:
            follow_ups.append("Track Engineering Tier-2 Ticket for localized cell tower congestion and notify customer via SMS.")

        if resolution.status == "UNRESOLVED" or "unable to verify" in all_agent_text:
            follow_ups.append("Initiate proactive supervisor callback with CPNI two-factor identity verification link.")

        if churn.risk_level in ["HIGH", "CRITICAL"] and resolution.status != "RESOLVED":
            follow_ups.append("Enqueue account into automated win-back retention campaign after 48 hours.")

        if "deal that might change your mind" in all_agent_text or "on us" in all_agent_text:
            follow_ups.append("Audit retention promo applied by agent for compliance with authorized discount matrix.")

        if not follow_ups:
            follow_ups.append("Log call notes in CRM and close customer interaction ticket.")

        return concise_summary, follow_ups
