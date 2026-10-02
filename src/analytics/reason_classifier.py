import re
from typing import List, Dict, Any
from src.models.schemas import Turn, CallReason, Speaker

CALL_REASON_RULES = [
    {
        "label": "Service Cancellation / Churn Threat",
        "keywords": [r"\bcancel\b", r"\bterminat\w*\b", r"\bclose my (account|service)\b", r"\bdisconnect\b"],
        "speaker": Speaker.CLIENT,
        "base_confidence": 0.95,
        "explanation": "Customer explicitly stated their intent to cancel or terminate telecom service."
    },
    {
        "label": "Network Coverage & Call Quality",
        "keywords": [r"\bdropped calls?\b", r"\bpoor reception\b", r"\bspotty\b", r"\bslow data\b", r"\bconnectivity\b", r"\bcoverage\b", r"\bno signal\b"],
        "speaker": Speaker.CLIENT,
        "base_confidence": 0.92,
        "explanation": "Customer reported network anomalies, dropped calls, or degraded data speed."
    },
    {
        "label": "Pricing & Cost Dissatisfaction",
        "keywords": [r"\btoo expensive\b", r"\bcost(ly)?\b", r"\bcheaper\b", r"\bafford\b", r"\bprice\b", r"\bbill(ing)?\b"],
        "speaker": Speaker.CLIENT,
        "base_confidence": 0.88,
        "explanation": "Customer cited service costs or financial constraints as a primary issue."
    },
    {
        "label": "Competitor Switching",
        "keywords": [r"\bmint mobile\b", r"\bt-mobile\b", r"\bverizon\b", r"\bat&t\b", r"\bbetter offer\b", r"\bswitch(ing)? to\b"],
        "speaker": Speaker.CLIENT,
        "base_confidence": 0.94,
        "explanation": "Customer mentioned receiving a competitive offer or planning to port out to another carrier."
    },
    {
        "label": "Plan Modification & Downgrade",
        "keywords": [r"\bswitch (you|me) over\b", r"\b2gb plan\b", r"\b5gb plan\b", r"\bcheaper plan\b", r"\bdata plan\b", r"\bnew line\b"],
        "speaker": None,  # Either speaker
        "base_confidence": 0.85,
        "explanation": "Discussion regarding switching plans, adjusting data allowance, or adding lines."
    },
    {
        "label": "Device & Equipment Return",
        "keywords": [r"\breturn (any )?equipment\b", r"\bhotspot\b", r"\bconfirmation email\b", r"\brefund\b", r"\breturn instructions\b"],
        "speaker": None,
        "base_confidence": 0.86,
        "explanation": "Inquiries or instructions on returning leased telecom devices, routers, or processing refunds."
    },
    {
        "label": "Identity Verification Difficulty",
        "keywords": [r"\bunable to verify\b", r"\btrouble locating your account\b", r"\bdriver's license\b", r"\baccount pin\b.*try again\b"],
        "speaker": Speaker.AGENT,
        "base_confidence": 0.90,
        "explanation": "Difficulties or repeated failures encountered during customer authentication."
    }
]


class ReasonClassifier:
    """Multi-label call reason classifier with turn evidence attribution."""

    def classify(self, turns: List[Turn]) -> List[CallReason]:
        detected_reasons: List[CallReason] = []

        for rule in CALL_REASON_RULES:
            matched_turn_ids: List[int] = []
            pattern_hits = 0

            for t in turns:
                if rule["speaker"] is not None and t.speaker != rule["speaker"]:
                    continue

                for pattern in rule["keywords"]:
                    if re.search(pattern, t.text, re.IGNORECASE):
                        if t.turn_id not in matched_turn_ids:
                            matched_turn_ids.append(t.turn_id)
                        pattern_hits += 1

            if matched_turn_ids:
                conf = min(0.99, rule["base_confidence"] + (0.02 * (len(matched_turn_ids) - 1)))
                detected_reasons.append(CallReason(
                    label=rule["label"],
                    confidence=round(conf, 2),
                    evidence_turns=matched_turn_ids[:5],
                    explanation=rule["explanation"]
                ))

        # Default fallback if no specific rule fired
        if not detected_reasons:
            detected_reasons.append(CallReason(
                label="General Customer Inquiry",
                confidence=0.70,
                evidence_turns=[turns[0].turn_id] if turns else [],
                explanation="Standard general service assistance or account query."
            ))

        # Sort by confidence descending
        detected_reasons.sort(key=lambda r: r.confidence, reverse=True)
        return detected_reasons
