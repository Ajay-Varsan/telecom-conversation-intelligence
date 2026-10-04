import re
from typing import List

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

NEGATION_WORDS = {
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

def score_clause(clause: str) -> tuple[float, float]:
    lower = clause.lower()
    clean = re.sub(r"[^a-z0-9\s\']", " ", lower)
    tokens = clean.split()
    
    pos = 0.0
    neg = 0.0
    
    # 1. Negative terms
    for phrase, wt in TELECOM_NEGATIVE_TERMS.items():
        if phrase in lower:
            neg += abs(wt)
            
    # 2. Positive terms
    for phrase, wt in TELECOM_POSITIVE_TERMS.items():
        p_tokens = phrase.split()
        p_len = len(p_tokens)
        for i in range(len(tokens) - p_len + 1):
            if tokens[i:i + p_len] == p_tokens:
                # Check preceding words in this clause
                window_start = max(0, i - 3)
                preceding = tokens[window_start:i]
                if any(n in preceding for n in NEGATION_WORDS):
                    neg += abs(wt) * 1.3
                else:
                    pos += wt
    return pos, neg

def score_turn(text: str) -> float:
    if not text:
        return 0.0
        
    lower = text.lower()
    
    # Handle "No, that's all. Thank you" idiom: replace "No, that's all" with neutral wrapup so "no" doesn't contaminate
    cleaned_text = lower
    for pattern in CLOSING_IDIOMS:
        cleaned_text = re.sub(pattern, "wrapup", cleaned_text)
        
    # Split into clauses by sentence terminators: . ! ? ;
    clauses = re.split(r"[.!?;\n]+", cleaned_text)
    total_pos = 0.0
    total_neg = 0.0
    
    for c in clauses:
        c = c.strip()
        if not c:
            continue
        p, n = score_clause(c)
        total_pos += p
        total_neg += n
        
    total = total_pos + total_neg
    if total == 0:
        return 0.0
    return round((total_pos - total_neg) / (total + 0.5), 3)

# Test 1: Image 1 Turn 14
t14 = "No, that's all. Thank you so much for your help, Janyce."
print("Image 1 Turn 14 Score:", score_turn(t14))

# Test 2: Image 2 Turn 9
t9 = "Okay, great. Thank you."
print("Image 2 Turn 9 Turn Score:", score_turn(t9))

# Test 3: Cumulative Customer Sentiment for Image 2 (Cancellation call)
image2_client_turns = [
    "Well, I just don't have good coverage in my area. I've been having trouble getting signal and it's really frustrating.",
    "That's fine. Can you just cancel my service now?",
    "Yes, I understand. Just cancel it.",
    "Okay, great. Thank you."
]
turn_scores = [score_turn(t) for t in image2_client_turns]
print("Image 2 Turn-by-Turn Client Scores:", turn_scores)

# Cumulative running sentiment calculation:
# Recency-weighted average of client's turns
weights = [1.0, 1.2, 1.5, 1.0] # wrap-up turn has lower anchor weight than cancellation demands
weighted_sum = sum(s * w for s, w in zip(turn_scores, weights))
cumulative = weighted_sum / sum(weights)
print("Image 2 Cumulative Customer Sentiment across the call:", round(cumulative, 3))
