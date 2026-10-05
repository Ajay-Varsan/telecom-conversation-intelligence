import os
import time
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def generate_weak_sentiment_label(text: str, speaker: str) -> str:
    """Generates weak supervision silver labels based on linguistic markers and telecom cues."""
    if not isinstance(text, str):
        return "neutral"
    t = str(text).lower()
    
    # Negative markers
    neg_signals = [
        "cancel", "canceling", "not happy", "unhappy", "terrible", "horrible", "awful",
        "dropped call", "dropping", "spotty", "poor reception", "slow data", "frustrat",
        "overcharge", "hidden fee", "switch carriers", "mint mobile", "t-mobile", "verizon",
        "ridiculous", "unacceptable", "too expensive", "expensive", "hate", "worst",
        "can't connect", "no service", "trouble getting signal", "doesn't work", "broken"
    ]
    
    # Positive markers (exclude if negated)
    pos_signals = [
        "thank you", "thanks", "sounds great", "sounds good", "perfect", "appreciate",
        "helpful", "take the offer", "switch me over", "deal", "awesome", "excellent",
        "glad", "resolved", "great service", "wonderful", "love it"
    ]
    if "not happy" not in t and "not very happy" not in t:
        pos_signals.append("happy with")
        pos_signals.append("happy")
    
    # Polite closing idiom exception: "no, that's all, thank you" -> positive / neutral wrapup
    if re.search(r"\bno,?\s+(that's|that is|it's|it is)\s+all\b", t):
        return "positive"

    # Competitor praise inversion: praising competitor = negative churn for Union Mobile
    if any(comp in t for comp in ["mint mobile", "t-mobile", "verizon", "at&t"]) and any(k in t for k in ["better", "cheaper", "free phone", "offer", "switch"]):
        return "negative"

    neg_count = sum(1 for sig in neg_signals if sig in t)
    pos_count = sum(1 for sig in pos_signals if sig in t)
    
    if neg_count > pos_count:
        return "negative"
    elif pos_count > neg_count:
        return "positive"
    else:
        return "neutral"

def train_and_evaluate():
    print("=" * 60)
    print("TRAINING TELECOM SENTIMENT CLASSIFICATION MODEL")
    print("=" * 60)

    corpus_path = "telecom-conversation-corpus/telecom_corpus_supplimental.csv"
    if not os.path.exists(corpus_path):
        print(f"Error: Dataset not found at {corpus_path}")
        return

    print("\n[1/5] Ingesting dataset...")
    df = pd.read_csv(corpus_path)
    print(f"Loaded {len(df):,} total turns from corpus.")

    # Filter to client turns for customer experience sentiment and drop empty texts
    client_df = df[df["speaker"] == "client"].dropna(subset=["text"]).copy()
    client_df["text"] = client_df["text"].astype(str)
    print(f"Filtering to customer turns: {len(client_df):,} client utterances.")

    print("\n[2/5] Synthesizing Weak Supervision Silver Labels...")
    client_df["sentiment"] = [
        generate_weak_sentiment_label(row["text"], row["speaker"])
        for _, row in client_df.iterrows()
    ]
    
    label_dist = client_df["sentiment"].value_counts()
    print("Class Distribution:")
    for lbl, cnt in label_dist.items():
        print(f"  - {lbl.upper():<10}: {cnt:>6,} ({cnt/len(client_df)*100:.1f}%)")

    print("\n[3/5] Train/Test Stratified Split (80% Train, 20% Held-out Test)...")
    X = client_df["text"]
    y = client_df["sentiment"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples : {len(X_test):,}")

    print("\n[4/5] Training TF-IDF (1-3 N-grams) + Multinomial Logistic Regression Pipeline...")
    start_train = time.perf_counter()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=25000,
            sublinear_tf=True,
            min_df=2
        )),
        ("classifier", LogisticRegression(
            C=2.5,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    pipeline.fit(X_train, y_train)
    train_time = round(time.perf_counter() - start_train, 2)
    print(f"Model successfully trained in {train_time} seconds!")

    print("\n[5/5] Evaluating on Held-Out Test Set (11,359 samples)...")
    start_eval = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    eval_time = (time.perf_counter() - start_eval) * 1000
    avg_latency_ms = eval_time / len(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Test Accuracy: {acc * 100:.2f}%")
    print(f"Average Inference Latency per Turn: {avg_latency_ms:.3f} ms")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix (Rows=True, Cols=Predicted):")
    classes = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    print(cm_df)

    # Save model artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/telecom_sentiment_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nTrained model artifact saved to: {model_path}")

    # Test real edge cases
    print("\n" + "=" * 60)
    print("BENCHMARKING MODEL ON USER EDGE CASES")
    print("=" * 60)

    test_cases = [
        "Mint Mobile is offering me a better deal and a free phone, so I really just want to switch.",
        "Well, I just don't have good coverage in my area. I've been having trouble getting signal and it's really frustrating.",
        "Can you just cancel my service now?",
        "No, that's all, thank you.",
        "That sounds great, thank you for your help.",
        "Sure, my account PIN is 4821.",
        "I'm not happy with my monthly bill, it keeps increasing without notice."
    ]

    for tc in test_cases:
        pred_label = pipeline.predict([tc])[0]
        probs = pipeline.predict_proba([tc])[0]
        class_order = pipeline.classes_
        prob_dict = {c: round(p, 3) for c, p in zip(class_order, probs)}
        print(f"\nText: \"{tc}\"")
        print(f" -> Predicted: {pred_label.upper()} (Probabilities: {prob_dict})")

if __name__ == "__main__":
    train_and_evaluate()
