import pandas as pd

df = pd.read_csv('telecom-conversation-corpus/telecom_corpus_supplimental.csv', nrows=1000)
print(f"Loaded {len(df)} rows.")
print("Unique conversations in 1k rows:", df['conversation_id'].nunique())

conv_ids = df['conversation_id'].unique()[:5]
for idx, cid in enumerate(conv_ids, 1):
    sub = df[df['conversation_id'] == cid]
    print(f"\n==========================================")
    print(f"CONVERSATION #{idx}: {cid} ({len(sub)} turns)")
    print(f"==========================================")
    for _, row in sub.iterrows():
        speaker = str(row['speaker']).upper()
        text = str(row['text'])
        dt = str(row['date_time'])
        print(f"[{speaker}] ({dt}): {text}")
