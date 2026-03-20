from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# THE labeled benchmark — confirmed by TPAFS download_adjudications_hf.py
train_ds = load_dataset("Persius/imr-appeals", split="train")
test_ds  = load_dataset("Persius/imr-appeals", split="test")

df_train = train_ds.to_pandas()
df_test  = test_ds.to_pandas()

print("Columns:", df_train.columns.tolist())
# → ['text', 'decision', 'appeal_type', 'full_text', 'sufficiency_id']

print(f"Train: {len(df_train):,} rows")   # → ~64,067
print(f"Test:  {len(df_test):,} rows")    # → ~9,920

print("\ndecision values:")
print(df_train['decision'].value_counts())

print("\nsufficiency_id values:")
print(df_train['sufficiency_id'].value_counts())

print("\nappeal_type values:")
print(df_train['appeal_type'].value_counts())

# Save raw splits
df_train.to_json("data/raw/train_backgrounds_suff.jsonl", orient="records", lines=True)
df_test.to_json("data/raw/test_backgrounds_suff.jsonl",  orient="records", lines=True)
print("\nSaved raw train and test JSONL files.")
