from typing import Any

from datasets import load_dataset
import pandas as pd


def has_kb_tag(tags: Any) -> bool:
    """Return True if 'kb' is present in the tags sequence.

    HuggingFace ``to_pandas()`` yields numpy arrays (not Python lists),
    so we coerce to list rather than checking ``isinstance(tags, list)``.
    """
    try:
        return "kb" in list(tags)
    except TypeError:
        return False

# The KB configs confirmed by download_corpus_hf.py
# For your 2,551-doc working set: regulatory-guidance + kb-tagged docs

ds_rg  = load_dataset("Persius/hicric", "regulatory-guidance")   # 1,110 docs
ds_leg = load_dataset("Persius/hicric", "legal")                  # 1,348 docs
ds_cg  = load_dataset("Persius/hicric", "clinical-guidelines")    # 40,110 rows

df_rg  = ds_rg['train'].to_pandas()
df_leg = ds_leg['train'].to_pandas()
df_cg  = ds_cg['train'].to_pandas()

print("regulatory-guidance columns:", df_rg.columns.tolist())
print("Sample tags:", df_rg['tags'].iloc[0])

# Filter for documents with the 'kb' tag
df_leg_kb = df_leg[df_leg['tags'].apply(has_kb_tag)]
df_cg_kb  = df_cg[df_cg['tags'].apply(has_kb_tag)]

print(f"\nregulatory-guidance docs: {len(df_rg):,}")
print(f"legal docs with kb tag:   {len(df_leg_kb):,}")
print(f"clinical-guidelines with kb tag: {len(df_cg_kb):,}")

# Combine into working KB (record source config for downstream tag filtering)
df_rg = df_rg.assign(config="regulatory-guidance")
df_leg_kb = df_leg_kb.assign(config="legal")
df_cg_kb = df_cg_kb.assign(config="clinical-guidelines")
df_kb = pd.concat([df_rg, df_leg_kb, df_cg_kb], ignore_index=True)
print(f"\nTotal working KB: {len(df_kb):,} docs")  # expect ~2,544

df_kb.to_csv("data/raw/hicric_kb_working.csv", index=False)
print("Saved: data/raw/hicric_kb_working.csv")
