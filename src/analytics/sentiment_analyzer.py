import os
import re
import joblib
from typing import List, Tuple
from src.models.schemas import Turn, SentimentArc, Speaker

TELECOM_POSITIVE_TERMS = {
    "thank you": 0.85, "thanks": 0.75, "great": 0.85, "glad": 0.75, "appreciate": 0.8,
    "helpful": 0.75, "wonderful": 0.9, "excellent": 0.9, "perfect": 0.9, "sounds good": 0.8,
    "sounds easy": 0.75, "take the offer": 0.85, "resolved": 0.8, "satisfied": 0.85,
    "good deal": 0.85, "switch me over": 0.8, "good": 0.45, "better": 0.5, "happy": 0.8, "pleasure": 0.7
}

TELECOM_NEGATIVE_TERMS = {
    "cancel": -0.75, "frustrat": -0.85, "ridiculous": -0.9, "expensive": -0.65,
    "dropped call": -0.85, "poor reception": -0.85, "slow data": -0.7, "spotty": -0.7,
    "unreliable": -0.85, "terrible": -0.95, "awful": -0.95, "horrible": -0.95,
    "not working": -0.8, "unable": -0.5, "complaint": -0.8, "angry": -0.9,
    "switch to": -0.65, "rip off": -0.95, "worst": -0.95, "disappointed": -0.8,
    "hate": -0.9, "unacceptable": -0.85, "disconnect": -0.6, "useless": -0.85,
    "trouble": -0.7, "problem": -0.6, "issue": -0.5, "poor": -0.7, "bad": -0.7,
    "no coverage": -0.85, "no signal": -0.85, "just cancel": -0.85
}

NEGATION_TOKENS = {
    "not", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "cant", "can't", "cannot", "never", "hardly", "barely",
    "without", "lacks", "lacking"
}

CLOSING_IDIOMS = [
    r"\bno,?\s+that'?s\s+all\b",
    r"\bno\s+thanks?\b",
    r"\bno\s+further\b",
    r"\bno\s+other\b",
    r"\bthat'?s\s+all\b"
]


class SentimentAnalyzer:
    """Production-grade sentiment analysis combining trained statistical ML pipeline with deterministic fallback."""

    def __init__(self, model_path: str = "models/telecom_sentiment_model.joblib"):
        self.ml_pipeline = None
        if os.path.exists(model_path):
            try:
                self.ml_pipeline = joblib.load(model_path)
            except Exception:
                self.ml_pipeline = None

    def _score_clause(self, clause: str) -> Tuple[float, float]:
        lower = clause.lower()
        clean = re.sub(r"[^a-z0-9\s\']", " ", lower)
        tokens = clean.split()

        pos_score = 0.0
        neg_score = 0.0

        # 1. Negative terms scanning
        for phrase, weight in TELECOM_NEGATIVE_TERMS.items():
            if phrase in lower:
                neg_score += abs(weight)

        # 2. Positive terms scanning with clause-bounded negation window
        for phrase, weight in TELECOM_POSITIVE_TERMS.items():
            p_tokens = phrase.split()
            p_len = len(p_tokens)
            for i in range(len(tokens) - p_len + 1):
                if tokens[i:i + p_len] == p_tokens:
                    # Check preceding tokens within this clause
                    window_start = max(0, i - 3)
                    preceding = tokens[window_start:i]
                    if any(neg in preceding for neg in NEGATION_TOKENS):
                        # Negated positive term flips to strong negative!
                        neg_score += abs(weight) * 1.3
                    else:
                        pos_score += weight

        return pos_score, neg_score

    def score_turn_text(self, text: str) -> float:
        if not text:
            return 0.0
        lower = text.lower()

        # Check explicit emotion tag in synthetic transcripts e.g. "(frustrated)"
        base_bias = 0.0
        if "(frustrated)" in lower or "(angry)" in lower:
            base_bias = -0.4
        elif "(happy)" in lower or "(relieved)" in lower:
            base_bias = 0.4

        # Primary Inference: Trained Telecom ML Model
        if self.ml_pipeline is not None:
            try:
                probs = self.ml_pipeline.predict_proba([text])[0]
                classes = list(self.ml_pipeline.classes_)
                p_neg = probs[classes.index("negative")]
                p_pos = probs[classes.index("positive")]
                p_neu = probs[classes.index("neutral")]

                # Calibrated polarity score [-1.0, 1.0]
                polarity = (p_pos - p_neg) / (1.0 - (0.5 * p_neu) + 1e-5)
                polarity += base_bias
                return max(-1.0, min(1.0, round(polarity, 3)))
            except Exception:
                pass

        # Fallback Engine: Deterministic Clause & Negation Rules
        cleaned_text = lower
        for pattern in CLOSING_IDIOMS:
            cleaned_text = re.sub(pattern, "wrapup", cleaned_text)

        # Split text into clauses by punctuation boundaries (. ! ? ; ,) so negations don't bleed across sentences
        clauses = re.split(r"[.!?;\n]+", cleaned_text)

        total_pos = 0.0
        total_neg = 0.0

        for c in clauses:
            c = c.strip()
            if not c:
                continue
            p, n = self._score_clause(c)

            # Target-Aware Competitor Sentiment Inversion:
            # If customer praises a competitor ("better offer from Mint Mobile", "free phone from Mint Mobile"),
            # that positive praise is a strong churn threat for Union Mobile!
            is_competitor_clause = any(comp in c for comp in ["mint mobile", "t-mobile", "verizon", "at&t", "cricket"])
            if is_competitor_clause and p > 0:
                n += p * 1.5
                p = 0.0

            total_pos += p
            total_neg += n

        total = total_pos + total_neg
        if total == 0:
            return max(-1.0, min(1.0, base_bias))

        raw_sentiment = (total_pos - total_neg) / (total + 0.5)
        raw_sentiment += base_bias
        return max(-1.0, min(1.0, round(raw_sentiment, 3)))

    def compute_cumulative_customer_sentiment(self, turns: List[Turn]) -> Tuple[float, str, str]:
        """Calculates cumulative customer experience sentiment across the entire call.
        Returns:
            (cumulative_score, cumulative_label, relationship_status)
        """
        client_turns = [t for t in turns if t.speaker == Speaker.CLIENT]
        if not client_turns:
            return 0.0, "neutral", "In Progress"

        scores = [self.score_turn_text(t.text) for t in client_turns]
        all_client_text = " ".join(t.text for t in client_turns).lower()
        all_text = " ".join(t.text for t in turns).lower()

        # Recency-weighted average
        weights = [1.0 + (0.2 * i) for i in range(len(scores))]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        base_cumulative = weighted_sum / sum(weights)

        # Contextual relationship state overrides
        is_canceled = any(k in all_text for k in ["placed a request to cancel", "i've canceled your service", "cancel your service", "just cancel it", "process the cancellation"])
        is_upgraded = any(k in all_client_text for k in ["switch me over to the premium", "take the offer", "switch me over to that plan", "sounds like a good deal"])
        is_switching_competitor = any(comp in all_client_text for comp in ["mint mobile", "t-mobile", "verizon", "at&t"]) and any(k in all_client_text for k in ["cancel", "switch", "better offer"])

        if is_upgraded:
            status = "Retained / Plan Upgraded"
            final_score = max(0.45, min(1.0, base_cumulative + 0.35))
        elif is_switching_competitor and not is_upgraded:
            competitor_name = "Mint Mobile" if "mint mobile" in all_client_text else "Competitor"
            status = f"Churn Threat / Porting to {competitor_name}"
            # Heavily penalize competitor churn intent
            final_score = min(-0.65, base_cumulative - 0.35)
        elif is_canceled and not is_upgraded:
            status = "Churned / Service Cancelled"
            final_score = min(-0.50, base_cumulative - 0.25)
        elif base_cumulative < -0.2:
            status = "Dissatisfied / Seeking Cancellation"
            final_score = base_cumulative
        elif base_cumulative > 0.2:
            status = "Satisfied / Positive Inquiry"
            final_score = base_cumulative
        else:
            status = "Neutral Interaction"
            final_score = base_cumulative

        final_score = max(-1.0, min(1.0, round(final_score, 3)))
        label = self.get_label(final_score)
        return final_score, label, status

    def get_label(self, score: float) -> str:
        if score > 0.15:
            return "positive"
        if score < -0.15:
            return "negative"
        return "neutral"

    def analyze_turns(self, turns: List[Turn]) -> List[Turn]:
        """Enrich turns with sentiment score and label."""
        enriched = []
        for t in turns:
            score = self.score_turn_text(t.text)
            enriched.append(Turn(
                turn_id=t.turn_id,
                speaker=t.speaker,
                text=t.text,
                timestamp=t.timestamp,
                sentiment=score,
                sentiment_label=self.get_label(score)
            ))
        return enriched

    def compute_sentiment_arc(self, turns: List[Turn]) -> SentimentArc:
        """Compute the Start -> Middle -> End customer sentiment arc."""
        # Focus primarily on client sentiment, or all turns if client turns are sparse
        client_turns = [t for t in turns if t.speaker == Speaker.CLIENT]
        target_turns = client_turns if len(client_turns) >= 2 else turns

        if not target_turns:
            return SentimentArc(
                start_sentiment=0.0,
                middle_sentiment=0.0,
                end_sentiment=0.0,
                start_label="neutral",
                end_label="neutral",
                trajectory="neutral",
                turn_sentiments=[]
            )

        sentiments = [self.score_turn_text(t.text) for t in target_turns]
        n = len(sentiments)

        if n == 1:
            s = sentiments[0]
            lbl = self.get_label(s)
            return SentimentArc(
                start_sentiment=s, middle_sentiment=s, end_sentiment=s,
                start_label=lbl, end_label=lbl, trajectory="neutral",
                turn_sentiments=sentiments
            )

        third = max(1, n // 3)
        start_slice = sentiments[:third]
        end_slice = sentiments[-third:]
        mid_slice = sentiments[third:-third] if n > 2 * third else sentiments[third - 1: -third + 1]
        if not mid_slice:
            mid_slice = [sentiments[n // 2]]

        start_avg = round(sum(start_slice) / len(start_slice), 3)
        mid_avg = round(sum(mid_slice) / len(mid_slice), 3)
        end_avg = round(sum(end_slice) / len(end_slice), 3)

        start_lbl = self.get_label(start_avg)
        end_lbl = self.get_label(end_avg)

        # Trajectory determination
        delta = end_avg - start_avg
        if delta >= 0.35 and start_avg < 0:
            trajectory = "positive_recovery"
        elif delta <= -0.35:
            trajectory = "negative_escalation"
        elif start_avg < -0.2 and end_avg < -0.2:
            trajectory = "consistently_negative"
        elif start_avg > 0.2 and end_avg > 0.2:
            trajectory = "consistently_positive"
        else:
            trajectory = "neutral"

        return SentimentArc(
            start_sentiment=start_avg,
            middle_sentiment=mid_avg,
            end_sentiment=end_avg,
            start_label=start_lbl,
            end_label=end_lbl,
            trajectory=trajectory,
            turn_sentiments=sentiments
        )
