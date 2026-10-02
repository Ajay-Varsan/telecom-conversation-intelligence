import re
from typing import List, Dict, Any, Tuple
from src.models.schemas import (
    Turn,
    Speaker,
    QAChecklistConfig,
    QAScoreResult
)
from src.qa.grounding_guardrail import GroundingGuardrail


class QAEvaluator:
    """Evaluates conversation against configurable QA checklist with grounded evidence."""

    def __init__(self, config: QAChecklistConfig):
        self.config = config
        self.guardrail = GroundingGuardrail()

    def evaluate(self, turns: List[Turn]) -> Tuple[float, bool, bool, List[QAScoreResult]]:
        """Evaluates all QA criteria.
        Returns:
            (total_score, overall_passed, critical_compliance_violation, item_results)
        """
        agent_turns = [t for t in turns if t.speaker == Speaker.AGENT]
        client_turns = [t for t in turns if t.speaker == Speaker.CLIENT]

        results: List[QAScoreResult] = []

        # 1. Greeting
        results.append(self._eval_greeting(agent_turns))

        # 2. Identity Verification
        results.append(self._eval_identity_verification(agent_turns, client_turns, turns))

        # 3. Empathy
        results.append(self._eval_empathy(agent_turns, client_turns))

        # 4. Correct Disclosure
        results.append(self._eval_disclosure(agent_turns, client_turns))

        # 5. No Prohibited Promises
        results.append(self._eval_no_prohibited_promises(agent_turns))

        # 6. Proper Closure
        results.append(self._eval_closure(agent_turns))

        # Compute weighted aggregate score
        total_weight = sum(item.weight for item in results)
        weighted_sum = sum(item.score * (item.weight / (total_weight or 1.0)) for item in results)
        final_score = round(weighted_sum, 1)

        # Critical compliance check: if any is_critical item is marked as violation
        critical_violation = any(
            r.is_violation and r.severity in ["HIGH", "CRITICAL"] for r in results
        )

        overall_passed = (final_score >= 75.0) and not critical_violation

        return final_score, overall_passed, critical_violation, results

    def _eval_greeting(self, agent_turns: List[Turn]) -> QAScoreResult:
        cfg = self.config.items.get("greeting", {})
        weight = cfg.get("weight", 10.0)

        if not agent_turns:
            return QAScoreResult(
                item_id="greeting",
                name="Professional Greeting & Name Introduction",
                category="SERVICE_QUALITY",
                passed=False,
                score=0.0,
                weight=weight,
                explanation="No agent turns recorded.",
                is_violation=False,
                severity="NONE"
            )

        first_turn = agent_turns[0]
        text_lower = first_turn.text.lower()

        has_hello = bool(re.search(r"\b(hello|hi|good (morning|afternoon|evening)|thank you for calling)\b", text_lower))
        has_name = bool(re.search(r"\b(my name is|i am|this is)\b", text_lower))
        has_brand = bool(re.search(r"\b(union mobile|mobile|customer support)\b", text_lower))

        score = 0.0
        if has_hello:
            score += 40.0
        if has_name:
            score += 40.0
        if has_brand:
            score += 20.0

        passed = score >= 80.0
        quotes = [first_turn.text] if has_hello or has_name else []
        grounded, matched_ids = self.guardrail.verify_all_quotes(quotes, agent_turns, Speaker.AGENT)

        return QAScoreResult(
            item_id="greeting",
            name="Professional Greeting & Name Introduction",
            category="SERVICE_QUALITY",
            passed=passed,
            score=score,
            weight=weight,
            quoted_evidence=quotes,
            evidence_turn_indices=matched_ids,
            is_grounded=grounded,
            explanation="Agent introduced self and company brand cordially." if passed else "Agent greeting missed full name introduction or company branding.",
            is_violation=not passed,
            severity="LOW" if not passed else "NONE"
        )

    def _eval_identity_verification(
        self, agent_turns: List[Turn], client_turns: List[Turn], all_turns: List[Turn]
    ) -> QAScoreResult:
        cfg = self.config.items.get("identity_verification", {})
        weight = cfg.get("weight", 25.0)

        all_client_text = " ".join(t.text for t in client_turns).lower()
        requires_verification = any(k in all_client_text for k in ["cancel", "terminate", "switch", "plan", "phone", "account"])

        quotes = []
        matched_turns = []

        for t in agent_turns:
            tl = t.text.lower()
            if any(k in tl for k in ["verify your identity", "account pin", "credit card on file", "verify your account", "driver's license", "identification"]):
                quotes.append(t.text)
                matched_turns.append(t.turn_id)

        if quotes:
            grounded, validated_ids = self.guardrail.verify_all_quotes(quotes, all_turns, Speaker.AGENT)
            return QAScoreResult(
                item_id="identity_verification",
                name="CPNI & Customer Identity Verification",
                category="COMPLIANCE",
                passed=True,
                score=100.0,
                weight=weight,
                quoted_evidence=quotes[:2],
                evidence_turn_indices=validated_ids[:2],
                is_grounded=grounded,
                explanation="Agent successfully initiated mandatory CPNI identity verification prior to handling account.",
                is_violation=False,
                severity="NONE"
            )
        else:
            if requires_verification:
                return QAScoreResult(
                    item_id="identity_verification",
                    name="CPNI & Customer Identity Verification",
                    category="COMPLIANCE",
                    passed=False,
                    score=0.0,
                    weight=weight,
                    quoted_evidence=[],
                    evidence_turn_indices=[],
                    is_grounded=True,
                    explanation="CRITICAL COMPLIANCE FAILURE: Agent discussed account cancellation or retention without authenticating customer identity.",
                    is_violation=True,
                    severity="CRITICAL"
                )
            else:
                return QAScoreResult(
                    item_id="identity_verification",
                    name="CPNI & Customer Identity Verification",
                    category="COMPLIANCE",
                    passed=True,
                    score=90.0,
                    weight=weight,
                    quoted_evidence=[],
                    evidence_turn_indices=[],
                    is_grounded=True,
                    explanation="Call was general inquiry not requiring privileged CPNI authentication.",
                    is_violation=False,
                    severity="NONE"
                )

    def _eval_empathy(self, agent_turns: List[Turn], client_turns: List[Turn]) -> QAScoreResult:
        cfg = self.config.items.get("empathy", {})
        weight = cfg.get("weight", 15.0)

        # Check if customer expressed problem or frustration
        problem_expressed = any(
            re.search(r"\b(dropped calls?|poor reception|slow data|expensive|frustrat\w*|ridiculous|cancel)\b", t.text, re.IGNORECASE)
            for t in client_turns
        )

        quotes = []
        for t in agent_turns:
            tl = t.text.lower()
            if any(k in tl for k in [
                "sorry to hear that", "i understand your frustration", "apologize for the inconvenience",
                "i understand how frustrating", "i apologize", "let me see what i can do to help"
            ]):
                quotes.append(t.text)

        if quotes:
            grounded, matched_ids = self.guardrail.verify_all_quotes(quotes, agent_turns, Speaker.AGENT)
            return QAScoreResult(
                item_id="empathy",
                name="Empathy & Frustration Acknowledgment",
                category="SERVICE_QUALITY",
                passed=True,
                score=100.0,
                weight=weight,
                quoted_evidence=quotes[:2],
                evidence_turn_indices=matched_ids[:2],
                is_grounded=grounded,
                explanation="Agent actively acknowledged customer dissatisfaction with empathetic phrasing.",
                is_violation=False,
                severity="NONE"
            )
        elif not problem_expressed:
            return QAScoreResult(
                item_id="empathy",
                name="Empathy & Frustration Acknowledgment",
                category="SERVICE_QUALITY",
                passed=True,
                score=85.0,
                weight=weight,
                quoted_evidence=[],
                evidence_turn_indices=[],
                is_grounded=True,
                explanation="No explicit dissatisfaction expressed by caller; neutral courteous handling.",
                is_violation=False,
                severity="NONE"
            )
        else:
            return QAScoreResult(
                item_id="empathy",
                name="Empathy & Frustration Acknowledgment",
                category="SERVICE_QUALITY",
                passed=False,
                score=30.0,
                weight=weight,
                quoted_evidence=[],
                evidence_turn_indices=[],
                is_grounded=True,
                explanation="Customer conveyed dissatisfaction or intent to cancel, but agent failed to offer an apology or empathetic statement.",
                is_violation=True,
                severity="MEDIUM"
            )

    def _eval_disclosure(self, agent_turns: List[Turn], client_turns: List[Turn]) -> QAScoreResult:
        cfg = self.config.items.get("correct_disclosure", {})
        weight = cfg.get("weight", 20.0)

        all_text = " ".join(t.text for t in agent_turns + client_turns).lower()
        is_cancellation_or_promo = any(k in all_text for k in ["cancel", "terminate", "iphone", "plan", "refund", "return"])

        quotes = []
        for t in agent_turns:
            tl = t.text.lower()
            if any(k in tl for k in [
                "cancellation fee", "return any equipment", "pay off any outstanding balance",
                "activation fee", "return instructions", "terms and conditions", "per month"
            ]):
                quotes.append(t.text)

        if quotes:
            grounded, matched_ids = self.guardrail.verify_all_quotes(quotes, agent_turns, Speaker.AGENT)
            return QAScoreResult(
                item_id="correct_disclosure",
                name="Mandatory Terms, Fees & Disclosures",
                category="COMPLIANCE",
                passed=True,
                score=100.0,
                weight=weight,
                quoted_evidence=quotes[:2],
                evidence_turn_indices=matched_ids[:2],
                is_grounded=grounded,
                explanation="Agent complied with mandatory disclosure requirements regarding fees, balances, and equipment returns.",
                is_violation=False,
                severity="NONE"
            )
        elif is_cancellation_or_promo:
            return QAScoreResult(
                item_id="correct_disclosure",
                name="Mandatory Terms, Fees & Disclosures",
                category="COMPLIANCE",
                passed=False,
                score=0.0,
                weight=weight,
                quoted_evidence=[],
                evidence_turn_indices=[],
                is_grounded=True,
                explanation="COMPLIANCE VIOLATION: Agent proceeded with cancellation or promotional deal without disclosing fee schedule or return policies.",
                is_violation=True,
                severity="HIGH"
            )
        else:
            return QAScoreResult(
                item_id="correct_disclosure",
                name="Mandatory Terms, Fees & Disclosures",
                category="COMPLIANCE",
                passed=True,
                score=100.0,
                weight=weight,
                quoted_evidence=[],
                evidence_turn_indices=[],
                is_grounded=True,
                explanation="No fee or policy disclosure triggered for this interaction type.",
                is_violation=False,
                severity="NONE"
            )

    def _eval_no_prohibited_promises(self, agent_turns: List[Turn]) -> QAScoreResult:
        cfg = self.config.items.get("no_prohibited_promises", {})
        weight = cfg.get("weight", 20.0)

        prohibited_quotes = []
        for t in agent_turns:
            tl = t.text.lower()
            # Catch unauthorized free hardware promises without disclosure: "brand new iPhone... It's on us."
            if ("it's on us" in tl or "completely free" in tl or "guarantee you will never have" in tl) and ("bill credit" not in tl and "24-month" not in tl):
                prohibited_quotes.append(t.text)

        if prohibited_quotes:
            grounded, matched_ids = self.guardrail.verify_all_quotes(prohibited_quotes, agent_turns, Speaker.AGENT)
            return QAScoreResult(
                item_id="no_prohibited_promises",
                name="Zero Prohibited / Deceptive Promises",
                category="COMPLIANCE",
                passed=False,
                score=0.0,
                weight=weight,
                quoted_evidence=prohibited_quotes,
                evidence_turn_indices=matched_ids,
                is_grounded=grounded,
                explanation="CRITICAL COMPLIANCE VIOLATION: Agent offered prohibited or unauthorized promises ('It's on us' / free hardware without qualifying contract terms).",
                is_violation=True,
                severity="CRITICAL"
            )
        else:
            return QAScoreResult(
                item_id="no_prohibited_promises",
                name="Zero Prohibited / Deceptive Promises",
                category="COMPLIANCE",
                passed=True,
                score=100.0,
                weight=weight,
                quoted_evidence=[],
                evidence_turn_indices=[],
                is_grounded=True,
                explanation="No prohibited promises or misleading promotional statements detected.",
                is_violation=False,
                severity="NONE"
            )

    def _eval_closure(self, agent_turns: List[Turn]) -> QAScoreResult:
        cfg = self.config.items.get("proper_closure", {})
        weight = cfg.get("weight", 10.0)

        if not agent_turns:
            return QAScoreResult(
                item_id="proper_closure",
                name="Comprehensive Closure & Survey / Branding",
                category="SERVICE_QUALITY",
                passed=False,
                score=0.0,
                weight=weight,
                explanation="No agent turns recorded.",
                is_violation=False,
                severity="NONE"
            )

        last_turns = agent_turns[-3:]
        quotes = []
        score = 0.0

        for t in last_turns:
            tl = t.text.lower()
            if "anything else i can assist" in tl or "further questions" in tl or "anything else today" in tl:
                score += 40.0
                quotes.append(t.text)
            if "thank you for choosing" in tl or "thanks for choosing" in tl:
                score += 35.0
                if t.text not in quotes:
                    quotes.append(t.text)
            if "have a great day" in tl or "goodbye" in tl or "take care" in tl:
                score += 25.0
                if t.text not in quotes:
                    quotes.append(t.text)

        passed = score >= 60.0
        grounded, matched_ids = self.guardrail.verify_all_quotes(quotes, agent_turns, Speaker.AGENT)

        return QAScoreResult(
            item_id="proper_closure",
            name="Comprehensive Closure & Survey / Branding",
            category="SERVICE_QUALITY",
            passed=passed,
            score=min(100.0, score),
            weight=weight,
            quoted_evidence=quotes[:2],
            evidence_turn_indices=matched_ids[:2],
            is_grounded=grounded,
            explanation="Agent executed proper closing etiquette with assistance offer and branding." if passed else "Agent closed interaction abruptly without standard closure checklist items.",
            is_violation=not passed,
            severity="LOW" if not passed else "NONE"
        )
