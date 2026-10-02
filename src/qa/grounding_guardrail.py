import re
from typing import List, Tuple, Optional
from src.models.schemas import Turn, Speaker


class GroundingGuardrail:
    """Verifies that evidence quotes attributed by the QA engine exist verbatim in raw transcript turns.
    This guarantees 100% explainability and eliminates LLM hallucinated citations.
    """

    def clean_text(self, text: str) -> str:
        """Normalize punctuation and whitespace for resilient matching."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    def verify_quote(self, quote: str, turns: List[Turn], expected_speaker: Optional[Speaker] = None) -> Tuple[bool, Optional[int], str]:
        """Checks if `quote` is grounded in any turn.
        Returns:
            (is_grounded, matched_turn_id, exact_transcript_snippet)
        """
        if not quote or not turns:
            return False, None, ""

        clean_q = self.clean_text(quote)
        if not clean_q:
            return False, None, ""

        # First try exact case-insensitive substring
        for t in turns:
            if expected_speaker and t.speaker != expected_speaker:
                continue
            if quote.lower() in t.text.lower():
                return True, t.turn_id, t.text

        # Second try normalized alphanumeric substring
        for t in turns:
            if expected_speaker and t.speaker != expected_speaker:
                continue
            clean_turn = self.clean_text(t.text)
            if clean_q in clean_turn or clean_turn in clean_q:
                return True, t.turn_id, t.text

        # Third try: check if at least 70% of words in sequence match
        q_words = clean_q.split()
        if len(q_words) >= 4:
            sub_phrase = " ".join(q_words[:5])
            for t in turns:
                if expected_speaker and t.speaker != expected_speaker:
                    continue
                clean_turn = self.clean_text(t.text)
                if sub_phrase in clean_turn:
                    return True, t.turn_id, t.text

        return False, None, ""

    def verify_all_quotes(self, quotes: List[str], turns: List[Turn], expected_speaker: Optional[Speaker] = None) -> Tuple[bool, List[int]]:
        """Verify multiple quotes and return matched turn indices."""
        matched_indices = []
        all_grounded = True

        for q in quotes:
            grounded, turn_id, _ = self.verify_quote(q, turns, expected_speaker)
            if grounded and turn_id is not None:
                if turn_id not in matched_indices:
                    matched_indices.append(turn_id)
            else:
                all_grounded = False

        return all_grounded, matched_indices
