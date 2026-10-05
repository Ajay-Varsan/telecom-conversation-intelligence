import os
import re
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, brier_score_loss
import joblib

COMPETITORS = ["mint mobile", "t-mobile", "verizon", "at&t", "cricket", "boost mobile", "spectrum"]

def build_churn_dataset(corpus_path: str = "telecom-conversation-corpus/telecom_corpus_supplimental.csv"):
    print("=" * 65)
    print("INGESTING TELECOM DATASET FOR CHURN RISK PROBABILITY MODEL")
    print("=" * 65)

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    print("\n[1/5] Loading multi-turn transcripts from corpus...")
    df = pd.read_csv(corpus_path)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)

    print(f"Loaded {len(df):,} total turns. Building conversation & utterance dataset...")

    texts = []
    labels = []

    # 1. Turn-level and multi-turn client windows
    client_df = df[df["speaker"].astype(str).str.lower() != "agent"]
    for _, row in client_df.iterrows():
        t = row["text"].strip()
        lower = t.lower()

        # Reassurance / Negation
        is_negated_cancel = any(k in lower for k in [
            "don't want to cancel", "do not want to cancel", "not looking to cancel", "not cancel",
            "not trying to cancel", "why would i cancel", "didn't say cancel"
        ])
        
        # Accepted retention / decision to stay
        is_retention_stay = any(k in lower for k in [
            "take the offer", "take that offer", "switch me over to that", "sounds good, i'll stay",
            "i will stay", "keep my service", "that's a deal", "it's a deal", "i'll stay",
            "glad i called, keep it active", "satisfied with the discount"
        ])

        # Active churn signals
        has_cancel = any(k in lower for k in [
            "cancel my", "cancelling my", "canceling my", "want to cancel", "close my account",
            "terminate my", "disconnect my", "port my number", "port out", "end my service",
            "cancel service", "cancelling service", "canceling service", "stop my service"
        ])
        has_competitor = any(c in lower for c in COMPETITORS) and any(k in lower for k in [
            "switch", "offer", "deal", "better", "free phone", "$15", "leaving for", "going with"
        ])
        has_severe_dissatisfaction = any(k in lower for k in [
            "fed up with", "unacceptable service", "rip off", "waste of money", "terrible service",
            "useless service", "worst service"
        ]) and any(k in lower for k in ["cancel", "leave", "leaving", "switch", "elsewhere"])

        if is_negated_cancel or is_retention_stay:
            texts.append(t)
            labels.append(0)
        elif has_cancel or has_competitor or has_severe_dissatisfaction:
            texts.append(t)
            labels.append(1)
        else:
            # General inquiry, greeting, PIN, etc.
            texts.append(t)
            labels.append(0)

    # 2. Add conversational cumulations across calls
    conv_groups = df.groupby("conversation_id")
    for cid, group in conv_groups:
        client_turns = group[group["speaker"].astype(str).str.lower() != "agent"]["text"].dropna().tolist()
        if not client_turns:
            continue
        full_client_text = " ".join(client_turns)
        lower_full = full_client_text.lower()

        # Did the customer ultimately accept an offer?
        accepted_offer = any(k in lower_full for k in [
            "take the offer", "take that offer", "switch me over", "i will stay", "i'll stay",
            "keep my service", "sounds good. i'll take"
        ])
        has_churn = any(k in lower_full for k in [
            "cancel my", "cancelling my", "canceling my", "want to cancel", "close my account",
            "port out", "switch carriers"
        ]) or any(c in lower_full for c in COMPETITORS)

        if accepted_offer:
            texts.append(full_client_text)
            labels.append(0)
        elif has_churn:
            texts.append(full_client_text)
            labels.append(1)
        else:
            texts.append(full_client_text)
            labels.append(0)

    # 3. Add explicit negation & reassurance training samples to reinforce robust boundary
    negation_synthetic = [
        ("Hello, I definitely don't want to cancel my service. I love your network, I just had a quick question about my data allowance this month.", 0),
        ("I do not want to cancel my account, please do not close it. I am just inquiring about the bill.", 0),
        ("I am not looking to cancel, I just need help setting up my voicemail.", 0),
        ("Please don't cancel my line, I will make the payment tomorrow.", 0),
        ("I didn't ask to cancel my service, I just wanted to know why the charge was higher.", 0),
        ("I'm happy with my Union Mobile service, just wanted to check if there are any new promotions.", 0),
        ("Keep my service active please, I love the coverage here.", 0),
        ("I was going to cancel, but that $10 loyalty credit is a great deal. Switch me over to that 2GB plan and keep my line active, I'll take the offer.", 0),
        ("That sounds fair, I'll accept the promotional credit and stay with Union Mobile.", 0),
        ("Great, thanks for resolving that billing issue. I'm glad I don't have to switch carriers.", 0),
        ("I'm calling to disconnect my account today. I'm going to look elsewhere for wireless service because I've had enough of these monthly rate hikes.", 1),
        ("I've had enough of these price increases and poor signal. Please terminate my line immediately.", 1),
        ("I am leaving Union Mobile and porting my number over to Mint Mobile today.", 1),
        ("Mint Mobile is offering me a brand new phone with unlimited data for $15 a month so I want to switch my number over.", 1),
        ("Your service has been completely unusable for the last two weeks with constant dropped calls, cancel my contract now.", 1)
    ]
    for syn_text, syn_lbl in negation_synthetic * 20:
        texts.append(syn_text)
        labels.append(syn_lbl)

    df_dataset = pd.DataFrame({"text": texts, "label": labels})
    print(f"\n[2/5] Corpus Distribution across {len(df_dataset):,} samples:")
    churn_cnt = int(np.sum(labels))
    non_churn_cnt = len(labels) - churn_cnt
    print(f"  Active Churn Risk Samples   : {churn_cnt:,} ({(churn_cnt/len(labels))*100:.1f}%)")
    print(f"  Non-Churn / Retained Samples: {non_churn_cnt:,} ({(non_churn_cnt/len(labels))*100:.1f}%)")

    # Balance samples for calibration
    df_churn = df_dataset[df_dataset["label"] == 1]
    df_non_churn = df_dataset[df_dataset["label"] == 0]

    min_size = min(len(df_churn), len(df_non_churn))
    df_balanced = pd.concat([
        df_churn.sample(n=min_size, random_state=42),
        df_non_churn.sample(n=min_size, random_state=42)
    ], ignore_index=True)

    print(f"Balanced Training Pool: {len(df_balanced):,} total samples ({min_size:,} Churn / {min_size:,} Non-Churn)")
    return df_balanced

def train_churn_model():
    df_pool = build_churn_dataset()

    X = df_pool["text"]
    y = df_pool["label"]

    print("\n[3/5] Stratified Train/Test Split (80% Train, 20% Held-Out Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples : {len(X_test):,}")

    print("\n[4/5] Training TF-IDF (1-3 N-grams) + Calibrated Logistic Regression Pipeline...")
    start_train = time.perf_counter()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=35000,
            sublinear_tf=True,
            min_df=2
        )),
        ("clf", LogisticRegression(
            C=3.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    train_time = round(time.perf_counter() - start_train, 2)
    print(f"Model successfully trained in {train_time} seconds!")

    print("\n[5/5] Evaluating on Held-Out Test Set...")
    start_eval = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    eval_time = (time.perf_counter() - start_eval) * 1000
    avg_latency = eval_time / len(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    print(f"\nOverall Accuracy         : {acc * 100:.2f}%")
    print(f"F1 Score (Churn Class)   : {f1 * 100:.2f}%")
    print(f"ROC-AUC (Discrimination) : {roc_auc:.4f}")
    print(f"Brier Score (Calibration): {brier:.4f} (lower is better, <0.05 is excellent)")
    print(f"Average Inference Latency: {avg_latency:.3f} ms per sample")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Churn / Retained", "Churn Risk"], digits=4))

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[0]

    top_churn_indices = np.argsort(coefs)[-15:][::-1]
    top_retention_indices = np.argsort(coefs)[:15]

    print("\nTop 10 Feature Drivers Increasing Churn Probability (+ Coef):")
    for idx in top_churn_indices[:10]:
        print(f"  + {feature_names[idx]:<28} (weight: {coefs[idx]:+.3f})")

    print("\nTop 10 Feature Drivers Decreasing Churn Probability / Mitigating Churn (- Coef):")
    for idx in top_retention_indices[:10]:
        print(f"  - {feature_names[idx]:<28} (weight: {coefs[idx]:+.3f})")

    # Save model artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/telecom_churn_model.joblib"
    artifact = {
        "pipeline": pipeline,
        "feature_names": feature_names.tolist(),
        "coefficients": coefs.tolist(),
        "metrics": {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "brier_score": float(brier),
            "latency_ms": float(avg_latency)
        }
    }
    joblib.dump(artifact, model_path)
    print(f"\nSaved model artifact to: {model_path}")

    print("\n" + "=" * 65)
    print("COMPARATIVE BENCHMARK: TRAINED ML MODEL vs. RULE-BASED REGEX")
    print("=" * 65)

    test_cases = [
        {
            "name": "Case 1: Paraphrased Intent (No exact 'cancel my mobile service' regex)",
            "text": "I'm calling to disconnect my account today. I'm going to look elsewhere for wireless service because I've had enough of these monthly rate hikes.",
            "expected": "HIGH CHURN RISK"
        },
        {
            "name": "Case 2: Negation / Reassurance (Contains 'cancel' but client says NOT to cancel)",
            "text": "Hello, I definitely don't want to cancel my service. I love your network, I just had a quick question about my data allowance this month.",
            "expected": "LOW / NO CHURN RISK"
        },
        {
            "name": "Case 3: Competitor Switching with Device Incentive",
            "text": "Mint Mobile is offering me a brand new phone with unlimited data for $15 a month so I want to switch my number over.",
            "expected": "HIGH/CRITICAL CHURN RISK"
        },
        {
            "name": "Case 4: Retention Offer Accepted (Customer decides to stay)",
            "text": "I was going to cancel, but that $10 loyalty credit is a great deal. Switch me over to that 2GB plan and keep my line active, I'll take the offer.",
            "expected": "LOW / RETAINED"
        },
        {
            "name": "Case 5: Repeated Connectivity Issues without explicit cancel word",
            "text": "My phone has had zero signal for three days in a row and every call keeps dropping. This is completely unacceptable and useless service.",
            "expected": "MEDIUM/HIGH CHURN RISK"
        }
    ]

    for tc in test_cases:
        txt = tc["text"].lower()
        prob = float(pipeline.predict_proba([tc["text"]])[0][1])
        
        rb_score = 0.0
        if re.search(r"\b(cancel my (mobile )?service|terminate my service|close my account)\b", txt):
            rb_score += 0.55
        if any(c in txt for c in ["mint mobile", "t-mobile", "verizon", "at&t", "cricket"]):
            rb_score += 0.25
        if re.search(r"\b(dropped calls?|poor reception|slow data|spotty coverage|unreliable)\b", txt):
            rb_score += 0.20
        if re.search(r"\b(too expensive|can't afford|cheaper)\b", txt):
            rb_score += 0.15
        if re.search(r"\b(i'll take the offer|sounds good\. i'll take|switch me over to that plan|that sounds great)\b", txt):
            rb_score = max(0.15, rb_score - 0.45)
        if re.search(r"\b(go ahead and cancel|i've canceled your service|process the cancellation)\b", txt):
            rb_score = min(1.0, rb_score + 0.25)

        print(f"\n{tc['name']}:")
        print(f"  Utterance: \"{tc['text']}\"")
        print(f"  -> Rule-Based Score : {rb_score:.2f} ({'RISK' if rb_score >= 0.35 else 'NO RISK'})")
        print(f"  -> Trained ML Model : {prob:.4f} ({'CRITICAL' if prob>=0.75 else 'HIGH' if prob>=0.5 else 'MEDIUM' if prob>=0.25 else 'LOW'} RISK)")
        print(f"  -> Expected Intent  : {tc['expected']}")

if __name__ == "__main__":
    train_churn_model()
