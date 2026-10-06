# CLAUDE.md — Agentic RAG Decision Support for Health Insurance Appeals

## Project Overview

This is a Masters research project (LJMU) implementing a three-agent RAG pipeline for
health insurance appeal outcome prediction. The system predicts whether an external appeal
will result in **Overturned**, **Upheld**, or **Insufficient** based on the case description
and retrieved policy evidence.

**Researcher:** Siniya Yousuf
**Degree:** MSc Data Science and AI, LJMU
**Title:** Agentic RAG Decision Support for Health Insurance Appeals

---

## Architecture — Three-Agent Sequential Pipeline

```
[Case Description (text)]
         │
         ▼
  ┌─────────────┐
  │   AGENT 1   │  Query Formulator
  │  FLAN-T5    │  Zero-shot → compact 1-2 sentence retrieval query
  │    base     │  + metadata tags (jurisdiction, insurance_type) forwarded
  └──────┬──────┘
         │ Structured Query
         ▼
  ┌─────────────┐
  │   AGENT 2   │  Evidence Retrieval & Summarisation
  │ FAISS +     │  1. Tag-filter KB to ~400-800 docs (jurisdiction + insurance_type)
  │ MiniLM-L6   │  2. Dense retrieval → top-5 passages (cosine similarity)
  └──────┬──────┘
         │ Evidence Summary (top-5 passages with doc IDs)
         ▼
  ┌─────────────┐
  │   AGENT 3   │  Decision Classifier
  │  LegalBERT  │  Fine-tuned on HICRIC benchmark
  │  + Softmax  │  Input: case text + evidence passages
  └──────┬──────┘
         │
         ▼
  [Decision: Overturned / Upheld / Insufficient + Justification]
```

---

## Datasets — VERIFIED FROM SOURCE CODE

### CRITICAL: Two separate HuggingFace datasets are used

#### 1. Labelled Benchmark — `Persius/imr-appeals`
- **Purpose:** Agent 1 input + Agent 3 training and evaluation
- **Total rows:** 73,987
- **Train split:** ~64,067 rows
- **Test split:** ~9,920 rows
- **Load with:** `load_dataset("Persius/imr-appeals", split="train")`

**Exact columns (verified from HuggingFace viewer):**
| Column | Type | Description |
|--------|------|-------------|
| `text` | string (60–2.6k chars) | Cleaned background span — **Agent 1 INPUT** |
| `decision` | string (2 values) | "Upheld" or "Overturned" |
| `appeal_type` | string (8 values) | Medical Necessity, Experimental, etc. |
| `full_text` | string (206–13.1k chars) | Full raw case text |
| `sufficiency_id` | int (0 or 1) | 0 = insufficient background, 1 = sufficient |

**3-class label construction (from TPAFS train_outcome_predictor.py):**
```python
ID2LABEL = {0: "Insufficient", 1: "Upheld", 2: "Overturned"}

def make_label(row):
    if row['sufficiency_id'] == 0:
        return "Insufficient"
    return row['decision']   # "Upheld" or "Overturned"

df['label'] = df.apply(make_label, axis=1)
```

**Expected 10k stratified subset distribution:**
- Upheld: ~5,900
- Overturned: ~4,100
- Insufficient: ~200

#### 2. Knowledge Base Corpus — `Persius/hicric`
- **Purpose:** Agent 2 FAISS index (retrieval corpus only — unlabelled)
- **Load with:** `load_dataset("Persius/hicric", "<config_name>")`

**Configs and row counts (verified):**
| Config | Rows | Use in Pipeline |
|--------|------|-----------------|
| `regulatory-guidance` | 1,110 | **Primary KB — include all** |
| `legal` | 1,348 | **Include if tagged 'kb'** |
| `clinical-guidelines` | 40,110 | **Include if tagged 'kb'** |
| `contract-coverage-rule-medical-policy` | 3,661 | Include if tagged 'kb' |
| `opinion-policy-summary` | 2,081 | Exclude — non-authoritative |
| `case-description` | 990,065 | Exclude — raw case text, not KB |

**Columns in all hicric configs:** `text`, `tags`, `date_accessed`, `source_url`, `source_md5`, `relative_path`

**Working KB filter — VERIFIED 2026-07 (supersedes the proposal's "~2,551 docs" claim):**
```python
# Include: all regulatory-guidance + rows with 'kb' tag from other configs.
# NOTE: to_pandas() yields numpy arrays, not lists — use list coercion:
df_kb_filtered = df[df['tags'].apply(lambda t: 'kb' in list(t))]
```

**Actual kb-tagged row counts (measured, not the proposal's estimate):**
| Config | kb-tagged rows | Unique source docs (source_md5) |
|--------|---------------|--------------------------------|
| `regulatory-guidance` (all rows) | 1,110 | ~1,105 |
| `legal` | 1,348 | ~340 |
| `clinical-guidelines` | 37,970 | 1 (single scraped source, pre-fragmented) |
| Working KB total (3 configs) | 40,428 | 1,445 |

**The proposal's "~2,551 doc KB" does NOT match any row-level count** — hicric rows
are heterogeneous chunks (legal rows average ~442KB; clinical rows are pre-split
fragments of one source). The working KB is 40,428 rows / ~166M words (40,418 after exact-text dedup), chunked to
1,121,403 MiniLM passages for the FAISS index (built 2026-07-16, fp32 torch backend,
IndexFlatIP, chunk=180 words / overlap=30). Thesis must report these corrected numbers.

**NOTE:** There is NO separate "kb" config — the `kb` tag lives inside the `tags` list field.

---

## Proposal vs Reality — Corrections to Apply in Thesis

| Proposal States | Actual Value | Notes |
|----------------|--------------|-------|
| Training corpus: 73,987 cases | Train: ~64,067, Test: ~9,920 | 73,987 = total dataset |
| Test set: 9,745 cases | ~9,920 cases | Use actual split size |
| HuggingFace: Persius/hicric | Labelled data: Persius/imr-appeals | Both datasets needed |
| KB: 2,551 documents | ~2,544 documents | Negligible difference |

---

## Software Stack

```
Python          3.11+
torch           2.2+
transformers    4.38+    (FLAN-T5-base, LegalBERT)
datasets        2.18+    (HuggingFace data loading)
sentence-transformers 3.0+  (all-MiniLM-L6-v2 embeddings)
faiss-cpu       1.8+     (FAISS IndexFlatIP — inner product similarity)
scikit-learn    1.5+     (stratified split, metrics, ECE)
pandas          2.0+
numpy           1.26+
wandb           0.17+    (experiment tracking)
accelerate      0.27+    (Trainer API support)
tqdm
```

**Models (all from HuggingFace Hub):**
- `google/flan-t5-base` — Agent 1 (zero-shot query formulation, ~250M params)
- `sentence-transformers/all-MiniLM-L6-v2` — Agent 2 (passage + query embedding, 22M params)
- `nlpaueb/legal-bert-base-uncased` — Agent 3 (fine-tuned classifier)
- `medicalai/ClinicalBERT` — Agent 3 alternative

---

## File Structure

```
agentic_rag_appeals/
├── CLAUDE.md                  ← this file
├── SKILLS.md                  ← subagent and skill definitions
├── .mcp.json                  ← MCP server configuration
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── train_backgrounds_suff.jsonl   ← from Persius/imr-appeals train
│   │   ├── test_backgrounds_suff.jsonl    ← from Persius/imr-appeals test
│   │   └── hicric_kb_working.csv          ← ~2,544 KB docs
│   └── processed/
│       ├── imr_train_10k.csv              ← stratified 10k subset (random_state=42)
│       └── imr_test.csv                   ← full test split with 3-class label
├── agents/
│   ├── agent1_query_formulator.py
│   ├── agent2_retrieval.py
│   └── agent3_classifier.py
├── indexes/
│   └── faiss_index/                       ← saved FAISS IndexFlatIP
├── evaluation/
│   └── results/                           ← macro-F1, ECE, confusion matrices
├── notebooks/
│   ├── 01_benchmark_exploration.ipynb
│   ├── 02_kb_exploration.ipynb
│   └── 03_pipeline_trace.ipynb
├── checkpoints/                           ← LegalBERT fine-tuned weights
└── src/
    └── modeling/                          ← reference: TPAFS/hicric baseline code
```

---

## 12-Week Plan Summary

| Phase | Weeks | Environment | Key Tasks |
|-------|-------|-------------|-----------|
| Phase 1: Data Prep | 1–2 | VS Code (CPU) | Load imr-appeals, build 10k split, build FAISS KB index |
| Phase 2: Agent 1 + 2 | 3–5 | VS Code (CPU) | FLAN-T5 query formulator, FAISS retrieval pipeline |
| Phase 3: Agent 3 | 5–8 | Google Colab (GPU) | LegalBERT fine-tuning, closed-book vs RAG comparison |
| Phase 4: Integration | 9–10 | Google Colab (GPU) | End-to-end pipeline, ablations, error analysis |
| Phase 5: Writing | 11–12 | Any | Thesis chapters, evaluation tables, citations |

**GPU requirement:** ~40 GPU-hours total. Use Google Colab Pro (T4 16GB) or Colab Pro+ (A100).
Fine-tuning LegalBERT on 10k cases: ~2–4 hours per run on T4.

---

## Key Design Decisions

1. **Zero-shot FLAN-T5** for Agent 1 — avoids NER annotation overhead
2. **Stratified 10k subset** — preserves class balance, trains in 2–4h vs days
3. **FAISS IndexFlatIP** — inner product = cosine similarity for unit-normalised vectors
4. **Tag-first filtering** — reduces KB from 2,544 to ~400–800 docs per query before dense retrieval
5. **Open-source only** — fully reproducible, no paid APIs
6. **fixed random_state=42** — must use consistently in all train_test_split calls
7. **Primary metric: macro-F1** — handles class imbalance; Insufficient class ~0.3% of data

---

## Evaluation Protocol

- **Primary:** Macro-F1 on full test split (~9,920 cases)
- **Secondary:** Per-class F1, Confusion matrix, ECE (calibration), Abstention accuracy
- **Ablation conditions:**
  - A3 only (no retrieval, no query formulation)
  - A2 + A3 (retrieval but raw case text as query)
  - Full pipeline A1 + A2 + A3
  - A1-null (FLAN-T5 skipped, raw text passed to FAISS)
- **Retrieval quality proxy:** macro-F1 delta (full vs retrieval-disabled)
- **Qualitative review:** 20–30 cases using 4-part rubric (Relevance, Faithfulness, Sufficiency, Clarity)

---

## Reference Repositories

- TPAFS baseline code: https://github.com/TPAFS/hicric
- Labelled benchmark: https://huggingface.co/datasets/Persius/imr-appeals
- KB corpus: https://huggingface.co/datasets/Persius/hicric
