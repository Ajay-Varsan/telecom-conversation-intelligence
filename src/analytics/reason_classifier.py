import os
import re
import joblib
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

REASON_EXPLANATIONS = {
    "Service Cancellation / Churn Threat": "Customer explicitly stated their intent to cancel or terminate telecom service.",
    "Network Coverage & Call Quality": "Customer reported network anomalies, dropped calls, or degraded data speed.",
    "Pricing & Cost Dissatisfaction": "Customer cited service costs, billing discrepancies, or financial constraints.",
    "Competitor Switching": "Customer mentioned receiving a competitive offer or planning to port out to another carrier.",
    "Plan Modification & Downgrade": "Discussion regarding switching plans, adjusting data allowance, or adding lines.",
    "Device & Equipment Return": "Inquiries or instructions on returning leased telecom devices, routers, or processing refunds.",
    "Identity Verification Difficulty": "Difficulties or repeated failures encountered during customer authentication."
}

REASON_SPEAKER_CONSTRAINTS = {
    "Service Cancellation / Churn Threat": Speaker.CLIENT,
    "Network Coverage & Call Quality": Speaker.CLIENT,
    "Pricing & Cost Dissatisfaction": Speaker.CLIENT,
    "Competitor Switching": Speaker.CLIENT,
    "Plan Modification & Downgrade": None,
    "Device & Equipment Return": None,
    "Identity Verification Difficulty": None
}


class ReasonClassifier:
    """Multi-label call reason classifier combining trained TF-IDF OneVsRest model with rule fallback."""

    def __init__(self, model_path: str = "models/telecom_call_reason_model.joblib"):
        self.ml_pipeline = None
        self.ml_labels = []
        if os.path.exists(model_path):
            try:
                artifact = joblib.load(model_path)
                if isinstance(artifact, dict) and "pipeline" in artifact:
                    self.ml_pipeline = artifact["pipeline"]
                    self.ml_labels = artifact["labels"]
                else:
                    self.ml_pipeline = artifact
            except Exception:
                self.ml_pipeline = None

    def classify(self, turns: List[Turn]) -> List[CallReason]:
        if not turns:
            return [CallReason(
                label="General Customer Inquiry",
                confidence=0.70,
                evidence_turns=[],
                explanation="Standard general service assistance or account query."
            )]

        detected_map: Dict[str, Dict[str, Any]] = {}

        # 1. ML Inference via Trained Pipeline if available
        if self.ml_pipeline is not None and self.ml_labels:
            texts = [t.text for t in turns]
            try:
                probs_matrix = self.ml_pipeline.predict_proba(texts)
                for t_idx, t in enumerate(turns):
                    probs = probs_matrix[t_idx]
                    for lbl_idx, label in enumerate(self.ml_labels):
                        prob = float(probs[lbl_idx])
                        req_speaker = REASON_SPEAKER_CONSTRAINTS.get(label)
                        if req_speaker is not None and t.speaker != req_speaker:
                            continue

                        if prob >= 0.35:
                            if label not in detected_map:
                                detected_map[label] = {
                                    "confidence": prob,
                                    "evidence_turns": [t.turn_id],
                                    "explanation": REASON_EXPLANATIONS.get(label, f"ML model detected {label}.")
                                }
                            else:
                                detected_map[label]["confidence"] = max(detected_map[label]["confidence"], prob)
                                if t.turn_id not in detected_map[label]["evidence_turns"]:
                                    detected_map[label]["evidence_turns"].append(t.turn_id)
            except Exception:
                pass

        # 2. Deterministic keyword reinforcement / fallback
        for rule in CALL_REASON_RULES:
            matched_turn_ids: List[int] = []
            for t in turns:
                if rule["speaker"] is not None and t.speaker != rule["speaker"]:
                    continue
                for pattern in rule["keywords"]:
                    if re.search(pattern, t.text, re.IGNORECASE):
                        if t.turn_id not in matched_turn_ids:
                            matched_turn_ids.append(t.turn_id)
                        break

            if matched_turn_ids:
                label = rule["label"]
                base_conf = rule["base_confidence"]
                calc_conf = min(0.99, base_conf + (0.02 * (len(matched_turn_ids) - 1)))
                if label not in detected_map:
                    detected_map[label] = {
                        "confidence": calc_conf,
                        "evidence_turns": matched_turn_ids[:5],
                        "explanation": rule["explanation"]
                    }
                else:
                    detected_map[label]["confidence"] = max(detected_map[label]["confidence"], calc_conf)
                    for tid in matched_turn_ids:
                        if tid not in detected_map[label]["evidence_turns"]:
                            detected_map[label]["evidence_turns"].append(tid)

        # 3. Assemble and sort results
        detected_reasons: List[CallReason] = [
            CallReason(
                label=lbl,
                confidence=round(data["confidence"], 2),
                evidence_turns=data["evidence_turns"][:5],
                explanation=data["explanation"]
            )
            for lbl, data in detected_map.items()
        ]

        if not detected_reasons:
            detected_reasons.append(CallReason(
                label="General Customer Inquiry",
                confidence=0.70,
                evidence_turns=[turns[0].turn_id] if turns else [],
                explanation="Standard general service assistance or account query."
            ))

        detected_reasons.sort(key=lambda r: r.confidence, reverse=True)
        return detected_reasons
