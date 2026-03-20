import pandas as pd
from sklearn.model_selection import train_test_split

df_train = pd.read_json("data/raw/train_backgrounds_suff.jsonl", lines=True)
df_test  = pd.read_json("data/raw/test_backgrounds_suff.jsonl",  lines=True)

# Exactly from construct_label() in train_outcome_predictor.py
LABEL2ID = {"Insufficient": 0, "Upheld": 1, "Overturned": 2}
ID2LABEL = {0: "Insufficient", 1: "Upheld", 2: "Overturned"}

def make_label(row):
    if row['sufficiency_id'] == 0:
        return "Insufficient"
    return row['decision']   # "Upheld" or "Overturned"

df_train['label'] = df_train.apply(make_label, axis=1)
df_test['label']  = df_test.apply(make_label, axis=1)

print("Train 3-class distribution:")
print(df_train['label'].value_counts())
print("\nProportions:")
print(df_train['label'].value_counts(normalize=True).round(4))

df_10k, _ = train_test_split(
    df_train,
    train_size=10_000,
    stratify=df_train['label'],
    random_state=42
)

print("10k label distribution:")
print(df_10k['label'].value_counts())
# Expect ~5,900 Upheld / ~4,100 Overturned / ~200 Insufficient

df_10k.to_csv("data/processed/imr_train_10k.csv", index=False)
df_test.to_csv("data/processed/imr_test.csv",     index=False)
print("Saved.")
