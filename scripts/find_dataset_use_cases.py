import pandas as pd
import re
import joblib

def find_cases():
    art = joblib.load("models/telecom_churn_model.joblib")
    pipe = art["pipeline"]

    df = pd.read_csv("telecom-conversation-corpus/telecom_corpus_supplimental.csv")
    convs = df.groupby("conversation_id")

    divergent_cases = []

    for cid, group in convs:
        client_turns = group[group["speaker"].astype(str).str.lower() != "agent"]["text"].dropna().tolist()
        if not client_turns:
            continue
        client_text = " ".join(client_turns)
        txt = client_text.lower()
        
        # Rule-based calculation
        rb_score = 0.0
        if re.search(r"\b(cancel my (mobile )?service|terminate my service|close my account)\b", txt):
            rb_score += 0.55
        if any(c in txt for c in ["mint mobile", "t-mobile", "verizon", "at&t", "cricket", "boost mobile"]):
            rb_score += 0.25
        if re.search(r"\b(dropped calls?|poor reception|slow data|spotty coverage|unreliable)\b", txt):
            rb_score += 0.20
        if re.search(r"\b(too expensive|can't afford|cheaper)\b", txt):
            rb_score += 0.15
        if re.search(r"\b(i'll take the offer|sounds good\. i'll take|switch me over to that plan|that sounds great)\b", txt):
            rb_score = max(0.15, rb_score - 0.45)
        
        rb_score = min(1.0, max(0.0, round(rb_score, 2)))
        ml_score = round(float(pipe.predict_proba([client_text])[0][1]), 2)
        
        # Check for meaningful divergence (Rule-based missed or under-scored vs ML)
        if (ml_score >= 0.70 and rb_score <= 0.25) or (ml_score <= 0.25 and rb_score >= 0.50):
            divergent_cases.append({
                "cid": cid,
                "client_text": client_text,
                "client_turns": client_turns,
                "rb_score": rb_score,
                "ml_score": ml_score,
                "reason": "False Negative by Rule-Based" if ml_score > rb_score else "False Positive by Rule-Based"
            })

    print(f"Total strongly divergent cases found: {len(divergent_cases)}")
    for i, c in enumerate(divergent_cases[:6]):
        print(f"\n{'='*65}")
        print(f"CASE {i+1} [Conversation ID: {c['cid']}]")
        print(f"Outcome Divergence: {c['reason']}")
        print(f"  -> Rule-Based Score : {c['rb_score']:.2f} ({'RISK' if c['rb_score'] >= 0.35 else 'NO RISK'})")
        print(f"  -> Trained ML Model : {c['ml_score']:.2f} ({'CRITICAL' if c['ml_score']>=0.75 else 'HIGH' if c['ml_score']>=0.5 else 'LOW'} RISK)")
        print(f"Customer Dialogue Snippet:")
        for t in c["client_turns"][:4]:
            print(f"  Customer: \"{t}\"")

if __name__ == "__main__":
    find_cases()
