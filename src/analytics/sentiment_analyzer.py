import re
from typing import List, Tuple
from src.models.schemas import Turn, SentimentArc, Speaker

TELECOM_POSITIVE_TERMS = {
    "thank you": 0.85, "thanks": 0.75, "great": 0.85, "glad": 0.75, "appreciate": 0.8,
    "helpful": 0.75, "wonderful": 0.9, "excellent": 0.9, "perfect": 0.9, "sounds good": 0.8,
    "sounds easy": 0.75, "take the offer": 0.75, "resolved": 0.8, "satisfied": 0.85,
    "good": 0.45, "better": 0.5, "happy": 0.8, "pleasure": 0.7
}

TELECOM_NEGATIVE_TERMS = {
    "cancel": -0.7, "frustrat": -0.85, "ridiculous": -0.9, "expensive": -0.65,
    "dropped call": -0.85, "poor reception": -0.85, "slow data": -0.7, "spotty": -0.7,
    "unreliable": -0.85, "terrible": -0.95, "awful": -0.95, "horrible": -0.95,
    "not working": -0.8, "unable": -0.5, "complaint": -0.8, "angry": -0.9,
    "switch to": -0.65, "rip off": -0.95, "worst": -0.95, "disappointed": -0.8,
    "hate": -0.9, "unacceptable": -0.85, "disconnect": -0.6, "useless": -0.85,
    "trouble": -0.7, "problem": -0.6, "issue": -0.5, "poor": -0.7, "bad": -0.7,
    "no coverage": -0.85, "no signal": -0.85, "signal problem": -0.8
}

NEGATION_TOKENS = {
    "not", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "cant", "can't", "cannot", "no", "never", "hardly", "barely",
    "unable", "without", "lacks", "lacking", "trouble"
}


class SentimentAnalyzer:
    """Production-grade sentiment analysis for telecom contact centers with negation awareness."""

    def score_turn_text(self, text: str) -> float:
        if not text:
            return 0.0
        lower = text.lower()
        clean = re.sub(r"[^a-z0-9\s\']", " ", lower)
        tokens = clean.split()

        # Check explicit emotion tag in synthetic transcripts e.g. "(frustrated)"
        base_bias = 0.0
        if "(frustrated)" in lower or "(angry)" in lower:
            base_bias = -0.4
        elif "(happy)" in lower or "(relieved)" in lower:
            base_bias = 0.4

        pos_score = 0.0
        neg_score = 0.0

        # 1. Negative terms scanning
        for phrase, weight in TELECOM_NEGATIVE_TERMS.items():
            if phrase in lower:
                neg_score += abs(weight)

        # 2. Positive terms scanning with negation window check
        for phrase, weight in TELECOM_POSITIVE_TERMS.items():
            p_tokens = phrase.split()
            p_len = len(p_tokens)
            for i in range(len(tokens) - p_len + 1):
                if tokens[i:i + p_len] == p_tokens:
                    # Check 3 preceding words for negation (e.g. "not getting good", "don't have good", "not happy")
                    window_start = max(0, i - 3)
                    preceding = tokens[window_start:i]
                    if any(neg in preceding for neg in NEGATION_TOKENS):
                        # Negated positive term flips to strong negative!
                        neg_score += abs(weight) * 1.3
                    else:
                        pos_score += weight

        total = pos_score + neg_score
        if total == 0:
            return max(-1.0, min(1.0, base_bias))

        raw_sentiment = (pos_score - neg_score) / (total + 0.5)
        raw_sentiment += base_bias
        return max(-1.0, min(1.0, round(raw_sentiment, 3)))

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
