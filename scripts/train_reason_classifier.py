import os
import time
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, hamming_loss
from typing import List
import joblib

REASON_LABELS = [
    "Service Cancellation / Churn Threat",
    "Network Coverage & Call Quality",
    "Pricing & Cost Dissatisfaction",
    "Competitor Switching",
    "Plan Modification & Downgrade",
    "Device & Equipment Return",
    "Identity Verification Difficulty"
]

def generate_multi_labels(text: str, speaker: str) -> List[int]:
    """Generates weak supervision multi-label binary vector for a turn."""
    if not isinstance(text, str):
        return [0] * len(REASON_LABELS)
    t = text.lower()
    spk = str(speaker).lower()

    # 1. Service Cancellation / Churn Threat
    is_cancel = 1 if (
        any(k in t for k in ["cancel", "canceling", "cancelled", "cancellation", "close my account", "disconnect my service", "terminate my", "end my service", "port my number"])
        and "considering" not in t and "don't want to cancel" not in t
    ) else 0

    # 2. Network Coverage & Call Quality
    is_network = 1 if any(k in t for k in [
        "dropped call", "dropping call", "calls keep dropping", "poor reception", "spotty",
        "slow data", "data is slow", "trouble getting signal", "no signal", "poor coverage",
        "bad coverage", "connectivity", "no service", "coverage in my area", "can't connect"
    ]) else 0

    # 3. Pricing & Cost Dissatisfaction
    is_pricing = 1 if any(k in t for k in [
        "too expensive", "expensive", "cost too much", "bill is high", "overcharged",
        "hidden fee", "afford", "price increases", "cheaper", "monthly rate", "bill keeps increasing"
    ]) else 0

    # 4. Competitor Switching
    is_competitor = 1 if (
        any(c in t for c in ["mint mobile", "t-mobile", "verizon", "at&t", "cricket", "boost mobile"])
        or ("better offer" in t and "another" in t)
        or "switch carriers" in t
    ) else 0

    # 5. Plan Modification & Downgrade
    is_plan = 1 if any(k in t for k in [
        "switch me over", "change my plan", "2gb plan", "5gb plan", "unlimited plus",
        "loyalty discount", "$10 off", "$15 off", "cheaper plan", "downgrade", "data allowance"
    ]) else 0

    # 6. Device & Equipment Return
    is_device = 1 if any(k in t for k in [
        "return equipment", "return the device", "return instructions", "14 days to return",
        "router", "modem", "hotspot", "leased equipment", "non-return fee", "brand new iphone", "it's on us"
    ]) else 0

    # 7. Identity Verification Difficulty
    is_auth = 1 if (
        any(k in t for k in ["unable to verify", "trouble locating your account", "can't find your account", "pin is incorrect", "try your pin again"])
        or ("pin" in t and any(k in t for k in ["verify", "verification", "security code", "last 4"]))
    ) else 0

    return [is_cancel, is_network, is_pricing, is_competitor, is_plan, is_device, is_auth]

def train_reason_classifier():
    print("=" * 65)
    print("TRAINING MULTI-LABEL TELECOM CALL REASON CLASSIFIER")
    print("=" * 65)

    corpus_path = "telecom-conversation-corpus/telecom_corpus_supplimental.csv"
    if not os.path.exists(corpus_path):
        print(f"Error: Dataset not found at {corpus_path}")
        return

    print("\n[1/5] Ingesting dataset from corpus...")
    df = pd.read_csv(corpus_path)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    print(f"Loaded {len(df):,} turns from corpus.")

    print("\n[2/5] Generating Multi-Label Binary Ground Truth...")
    label_matrix = np.array([
        generate_multi_labels(row["text"], row.get("speaker", ""))
        for _, row in df.iterrows()
    ])

    print("Label Occurrence Across Corpus:")
    for idx, name in enumerate(REASON_LABELS):
        count = int(np.sum(label_matrix[:, idx]))
        pct = (count / len(df)) * 100
        print(f"  {idx+1}. {name:<38}: {count:>6,} ({pct:.1f}%)")

    # Filter to turns with at least one reason, plus a sample of neutral turns for negative training
    has_label = np.sum(label_matrix, axis=1) > 0
    df_labeled = df[has_label].copy()
    y_labeled = label_matrix[has_label]

    # Sample equal number of negative/neutral turns
    df_neutral = df[~has_label].sample(n=min(len(df_labeled), int(np.sum(~has_label))), random_state=42)
    y_neutral = label_matrix[~has_label][:len(df_neutral)]

    X_all = pd.concat([df_labeled["text"], df_neutral["text"]], ignore_index=True)
    y_all = np.vstack([y_labeled, y_neutral])

    print(f"\nFinal Training Pool: {len(X_all):,} utterances ({len(df_labeled):,} active, {len(df_neutral):,} neutral baseline)")

    print("\n[3/5] Stratified Train/Test Split (80% Train, 20% Held-Out Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42
    )
    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples : {len(X_test):,}")

    print("\n[4/5] Training TF-IDF (1-3 N-grams) + OneVsRest Logistic Regression Pipeline...")
    start_train = time.perf_counter()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=30000,
            sublinear_tf=True,
            min_df=2
        )),
        ("clf", OneVsRestClassifier(LogisticRegression(
            C=3.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs"
        )))
    ])

    pipeline.fit(X_train, y_train)
    train_time = round(time.perf_counter() - start_train, 2)
    print(f"Model successfully trained in {train_time} seconds!")

    print("\n[5/5] Evaluating on Held-Out Test Set...")
    start_eval = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    eval_time = (time.perf_counter() - start_eval) * 1000
    avg_latency = eval_time / len(X_test)

    h_loss = hamming_loss(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    micro_f1 = f1_score(y_test, y_pred, average="micro")

    print(f"\nHamming Loss (Error Rate): {h_loss:.4f} (Accuracy ~ {(1 - h_loss)*100:.2f}%)")
    print(f"Micro-averaged F1 Score : {micro_f1 * 100:.2f}%")
    print(f"Macro-averaged F1 Score : {macro_f1 * 100:.2f}%")
    print(f"Average Inference Latency: {avg_latency:.3f} ms per utterance")

    print("\nDetailed Multi-Label Classification Report:")
    print(classification_report(y_test, y_pred, target_names=REASON_LABELS, digits=4))

    # Save model artifact and label list
    os.makedirs("models", exist_ok=True)
    model_path = "models/telecom_call_reason_model.joblib"
    artifact = {
        "pipeline": pipeline,
        "labels": REASON_LABELS
    }
    joblib.dump(artifact, model_path)
    print(f"\nModel artifact saved to: {model_path}")

    # Benchmark on real conversational edge cases
    print("\n" + "=" * 65)
    print("BENCHMARKING MULTI-LABEL CLASSIFIER ON REAL TEST UTTERANCES")
    print("=" * 65)

    test_samples = [
        "Hi Julia, I'm calling to cancel my mobile service with Union Mobile.",
        "Well, I just don't have good coverage in my area. I've been having trouble getting signal and it's really frustrating.",
        "Mint Mobile is offering me a better deal and a free phone, so I really just want to switch.",
        "My monthly bill is way too expensive, and I cannot afford these extra charges.",
        "Could you switch me over to that cheaper 2GB plan with the $10 loyalty credit?",
        "Do I need to return any equipment or leased routers to avoid the $25 fee?",
        "I am having trouble locating your account, could you verify your PIN again?"
    ]

    for text in test_samples:
        probs = pipeline.predict_proba([text])[0]
        active = [(REASON_LABELS[i], round(float(probs[i]), 3)) for i in range(len(REASON_LABELS)) if probs[i] >= 0.35]
        active.sort(key=lambda x: x[1], reverse=True)
        print(f"\nUtterance: \"{text}\"")
        if active:
            for lbl, conf in active:
                print(f"  -> [{int(conf*100)}% Conf] {lbl}")
        else:
            print("  -> No dominant intent (General / Politeness)")

if __name__ == "__main__":
    train_reason_classifier()
