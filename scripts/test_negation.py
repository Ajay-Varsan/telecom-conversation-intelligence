import re

TELECOM_POSITIVE_TERMS = {
    "thank you": 0.8, "thanks": 0.7, "great": 0.85, "glad": 0.75, "appreciate": 0.8,
    "helpful": 0.75, "wonderful": 0.9, "excellent": 0.9, "perfect": 0.9, "sounds good": 0.75,
    "sounds easy": 0.7, "take the offer": 0.65, "resolved": 0.8, "satisfied": 0.85,
    "good": 0.5, "better": 0.5, "happy": 0.8, "pleasure": 0.7
}

TELECOM_NEGATIVE_TERMS = {
    "cancel": -0.7, "frustrat": -0.85, "ridiculous": -0.9, "expensive": -0.6,
    "dropped call": -0.8, "poor reception": -0.85, "slow data": -0.7, "spotty": -0.7,
    "unreliable": -0.85, "terrible": -0.9, "awful": -0.9, "horrible": -0.95,
    "not working": -0.8, "unable": -0.5, "complaint": -0.8, "angry": -0.9,
    "switch to": -0.6, "rip off": -0.95, "worst": -0.9, "disappointed": -0.8,
    "hate": -0.9, "unacceptable": -0.85, "disconnect": -0.6, "useless": -0.85,
    "trouble": -0.65, "problem": -0.6, "issue": -0.5, "poor": -0.7, "bad": -0.7,
    "no coverage": -0.8, "no signal": -0.85
}

NEGATIONS = {
    "not", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "cant", "can't", "cannot", "no", "never", "hardly", "barely",
    "unable", "without", "lacks", "lacking"
}

def score(text):
    lower = text.lower()
    clean = re.sub(r"[^a-z0-9\s\']", " ", lower)
    tokens = clean.split()
    
    pos_score = 0.0
    neg_score = 0.0
    
    # Check negative terms
    for phrase, wt in TELECOM_NEGATIVE_TERMS.items():
        if phrase in lower:
            neg_score += abs(wt)
            
    # Check positive terms with negation window
    for phrase, wt in TELECOM_POSITIVE_TERMS.items():
        p_tokens = phrase.split()
        for i in range(len(tokens) - len(p_tokens) + 1):
            if tokens[i:i+len(p_tokens)] == p_tokens:
                # Check 3 preceding words for negation
                window_start = max(0, i - 3)
                preceding = tokens[window_start:i]
                if any(neg in preceding for neg in NEGATIONS):
                    # Inverted! 'not happy' or 'not good' is strongly negative
                    neg_score += abs(wt) * 1.3
                else:
                    pos_score += wt

    total = pos_score + neg_score
    if total == 0:
        return 0.0
    return round((pos_score - neg_score) / (total + 0.5), 3)

t1 = "Well, I just don't have good coverage in my area. I've been having trouble getting signal and it's really frustrating."
t2 = "Hi Janyce, I'm calling to cancel my mobile service. I'm not getting good coverage in my area and I'm just not happy with the service."
print("T1 (Turn 3):", score(t1))
print("T2 (Turn 2):", score(t2))
