import pandas as pd

df = pd.read_csv("telecom-conversation-corpus/telecom_corpus_supplimental.csv")
print(f"Total turns: {len(df)}")
print(f"Unique conversations: {df['conversation_id'].nunique()}")
print("\nSpeaker breakdown:")
print(df["speaker"].value_counts())
print("\nFirst 5 turns:")
for _, row in df.head(5).iterrows():
    print(f"[{row['speaker']}]: {row['text']}")
