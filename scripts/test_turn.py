import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.analytics.sentiment_analyzer import SentimentAnalyzer, TELECOM_POSITIVE_TERMS, TELECOM_NEGATIVE_TERMS, NEGATION_TOKENS
import re

sa = SentimentAnalyzer()
t14 = "No, that's all. Thank you so much for your help, Janyce."
lower = t14.lower()
clean = re.sub(r"[^a-z0-9\s\']", " ", lower)
tokens = clean.split()
print("Tokens:", tokens)

for phrase, weight in TELECOM_POSITIVE_TERMS.items():
    p_tokens = phrase.split()
    p_len = len(p_tokens)
    for i in range(len(tokens) - p_len + 1):
        if tokens[i:i + p_len] == p_tokens:
            window_start = max(0, i - 3)
            preceding = tokens[window_start:i]
            print(f"Match positive '{phrase}' at {i}. Preceding tokens: {preceding}")
            if any(neg in preceding for neg in NEGATION_TOKENS):
                print(f"  -> FLIPPED by negation token: {[n for n in preceding if n in NEGATION_TOKENS]}")

