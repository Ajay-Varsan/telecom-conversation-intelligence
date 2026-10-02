import pandas as pd

df = pd.read_csv('telecom-conversation-corpus/telecom_corpus_supplimental.csv', nrows=25000)
first_turns = df[df['speaker'] == 'client'].groupby('conversation_id').first()['text']

print(f"Sample of 20 client opening statements:")
for i, txt in enumerate(first_turns.head(20), 1):
    print(f"{i}. {txt[:120]}")
