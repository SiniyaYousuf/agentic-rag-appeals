# SKILLS.md — Agentic RAG Health Insurance Appeals Project

This file defines the subagent roles, skills, and task boundaries for AI-assisted
development of this project. Use this as a reference when delegating tasks to Claude
subagents or when working in Claude Code / Cowork.

---

## Subagent Definitions

Each subagent below has a specific scope. Always activate only the relevant subagent
for a task to prevent scope creep and keep outputs focused.

---

### Subagent 1: `data-prep-agent`

**Trigger:** Any task involving dataset loading, splitting, filtering, or saving.

**Scope:**
- Loading `Persius/imr-appeals` and `Persius/hicric` from HuggingFace
- Constructing the 3-class label from `decision` + `sufficiency_id`
- Creating the stratified 10,000-case training subset (random_state=42)
- Filtering KB documents (regulatory-guidance + kb-tagged docs)
- Saving JSONL/CSV files to `data/raw/` and `data/processed/`

**Must NOT do:**
- Model training or inference
- FAISS index building (that's `indexing-agent`)
- Any evaluation

**Key facts to remember:**
- Input dataset: `Persius/imr-appeals` (NOT `Persius/hicric`)
- Label column: derived — `sufficiency_id==0` → Insufficient, else use `decision`
- Text column for model input: `text` (cleaned background span)
- Training split: ~64,067 rows; test split: ~9,920 rows
- Always stratify on 3-class label, always use random_state=42

**Sample invocation prompt:**
> "Load Persius/imr-appeals, construct the 3-class label column, create a stratified
> 10,000-case training subset, and save to data/processed/imr_train_10k.csv"

---

### Subagent 2: `indexing-agent`

**Trigger:** Any task involving FAISS index creation, passage chunking, or embedding.

**Scope:**
- Loading KB CSV from `data/raw/hicric_kb_working.csv`
- Chunking documents into 256-token passages with 64-token overlap
- Embedding passages using `sentence-transformers/all-MiniLM-L6-v2`
- Building and saving FAISS IndexFlatIP to `indexes/faiss_index/`
- Storing metadata lookup table (tags, relative_path, doc_id) alongside index

**Must NOT do:**
- Any model fine-tuning
- Agent 1 or Agent 3 logic
- Loading the imr-appeals benchmark dataset

**Key facts to remember:**
- Use `IndexFlatIP` (inner product) — NOT `IndexFlatL2`. Vectors are unit-normalised so IP = cosine.
- Expected passage count: ~180,000–250,000 after chunking
- Index file size: ~1.5GB. Save with `faiss.write_index()`
- Metadata must be saved as a separate JSON/CSV alongside the index
- Tags field in hicric is a list — filter with `'kb' in tags`

**Sample invocation prompt:**
> "Chunk the KB documents in data/raw/hicric_kb_working.csv into 256-token passages
> with 64-token overlap, embed with all-MiniLM-L6-v2, and build a FAISS IndexFlatIP.
> Save index and metadata to indexes/faiss_index/"

---

### Subagent 3: `agent1-dev`

**Trigger:** Any task involving query formulation, FLAN-T5, or prompt engineering for Agent 1.

**Scope:**
- Implementing `agents/agent1_query_formulator.py`
- Designing and testing zero-shot FLAN-T5-base prompts
- Benchmarking query quality (FLAN-T5 vs raw text passthrough)
- Writing the Agent-1-null ablation condition

**Must NOT do:**
- Fine-tuning FLAN-T5 (it is used zero-shot only)
- Anything involving FAISS or retrieval (that is Agent 2)
- Agent 3 classification

**Key facts to remember:**
- Model: `google/flan-t5-base` (~250M params, runs on CPU for testing)
- Prompt template: "Summarise the denied medical procedure, diagnosis, and the key
  coverage question in this health insurance appeal in 1-2 sentences: [case description]"
- Input field: `text` column from imr-appeals
- Output: compact 1-2 sentence retrieval query + metadata tags forwarded to Agent 2
- Ablation: Agent-1-null passes raw `text` directly to Agent 2 without FLAN-T5

**Sample invocation prompt:**
> "Implement agent1_query_formulator.py using FLAN-T5-base zero-shot. Include
> an agent1_null() function that passes the raw text without reformulation."

---

### Subagent 4: `agent2-dev`

**Trigger:** Any task involving retrieval, FAISS search, tag filtering, or evidence summarisation.

**Scope:**
- Implementing `agents/agent2_retrieval.py`
- Two-stage retrieval: (1) tag-filter by jurisdiction/insurance_type, (2) dense cosine search
- Returning top-5 passages with doc IDs, tags, and similarity scores
- Extractive summarisation of retrieved passages

**Must NOT do:**
- Building the FAISS index (that's `indexing-agent`)
- Anything involving LegalBERT or classification (that's Agent 3)

**Key facts to remember:**
- Tag filtering reduces 2,544 KB docs to ~400–800 per query
- Embed query with the same model used for indexing: `all-MiniLM-L6-v2`
- Return top-10 by similarity, keep top-5 as evidence set
- Each evidence note must include: passage text + doc_id + category tag + similarity score
- These passage IDs are cited in Agent 3 output for auditability

**Sample invocation prompt:**
> "Implement agent2_retrieval.py. Load the saved FAISS index. Given a query string
> and metadata tags, filter the index, embed the query, retrieve top-5 passages,
> and return them with doc IDs and similarity scores."

---

### Subagent 5: `agent3-dev`

**Trigger:** Any task involving LegalBERT fine-tuning, classification, or softmax output.

**Scope:**
- Implementing `agents/agent3_classifier.py`
- Fine-tuning `nlpaueb/legal-bert-base-uncased` on the 10k training subset
- Building input template: `[case_text] EVIDENCE: [p1 (id)] | [p2 (id)] | ...`
- Training both variants: closed-book (no evidence) and RAG (with evidence)
- Logging all runs to Weights & Biases

**Must NOT do:**
- Anything involving data preparation or FAISS
- Running on CPU — always use Google Colab GPU for training

**Key facts to remember:**
- Must run on GPU (T4 16GB minimum, A100 preferred)
- Batch size 16 requires ~12GB VRAM
- Training time: 2–4 hours per run on T4
- Use Softmax (not Sigmoid) — classes are mutually exclusive
- Label mapping: {0: "Insufficient", 1: "Upheld", 2: "Overturned"}
- Report: macro-F1, per-class F1, confusion matrix, ECE

**Sample invocation prompt:**
> "Fine-tune LegalBERT on data/processed/imr_train_10k.csv for 3-class classification.
> Run two variants: closed-book (text only) and RAG (text + evidence passages).
> Log both runs to W&B and save checkpoints to checkpoints/"

---

### Subagent 6: `eval-agent`

**Trigger:** Any task involving metric calculation, ablation tables, error analysis, or results reporting.

**Scope:**
- Running the full pipeline on the test set (~9,920 cases)
- Computing macro-F1, per-class F1, ECE, abstention accuracy
- Generating confusion matrices
- Running all 4 ablation conditions and building comparison table
- Automated error categorisation (retrieval failures, query failures, decision failures)
- Qualitative review of 20–30 cases using the 4-part rubric

**Key facts to remember:**
- Primary metric: macro-F1 (not accuracy — data is imbalanced)
- ECE measures calibration of softmax probabilities
- Abstention: treat "Insufficient" class as uncertainty signal, report precision + recall
- Retrieval proxy: macro-F1 delta between full pipeline and retrieval-disabled
- Error categories: (1) retrieval failure = mean cosine < 0.25, (2) query failure = FLAN-T5
  output < 10 tokens, (3) decision failure = residual

---

## Core Skills Required

### Data Engineering Skills
- HuggingFace `datasets` library loading with split= and name= parameters
- Stratified train/test splits with `sklearn.model_selection.train_test_split`
- JSONL reading/writing with pandas
- Handling list-type columns (e.g., `tags` field) in pandas DataFrames

### Embedding and Indexing Skills
- SentenceTransformer model loading and batch encoding
- FAISS `IndexFlatIP` creation, population, serialisation (`write_index` / `read_index`)
- Passage chunking with token-level overlap using HuggingFace tokenizers
- Metadata lookup table construction and linking to FAISS index by position

### NLP / Transformers Skills
- HuggingFace `AutoModelForSequenceClassification` fine-tuning with `Trainer` API
- Zero-shot generation with FLAN-T5 using `pipeline("text2text-generation")`
- Tokenisation with truncation and padding for BERT-type models
- LegalBERT loading: `nlpaueb/legal-bert-base-uncased`

### Evaluation Skills
- `sklearn.metrics`: f1_score(average='macro'), confusion_matrix, classification_report
- ECE calculation using probability bins
- Weights & Biases run logging: `wandb.init()`, `wandb.log()`

### Environment Skills
- VS Code venv management for CPU-based development (Weeks 1–4)
- Google Colab + Google Drive mounting for GPU training (Weeks 5–10)
- GitHub push/pull workflow as bridge between VS Code and Colab

---

## Task-to-Environment Mapping

| Task | Environment | Why |
|------|-------------|-----|
| Data loading, splitting, saving | VS Code (CPU) | No GPU needed |
| FAISS index building (250k passages) | VS Code overnight or Colab | One-time, ~45min CPU |
| Agent 1 development and testing | VS Code (CPU) | FLAN-T5 inference on small batches |
| Agent 2 development and testing | VS Code (CPU) | FAISS search is CPU-friendly |
| LegalBERT fine-tuning | Google Colab (T4/A100) | 12GB VRAM required |
| Full pipeline evaluation | Google Colab (T4) | Batch inference on 9,920 cases |
| Ablation experiments | Google Colab (T4) | Multiple training runs |
| Error analysis scripts | VS Code (CPU) | Pure pandas/sklearn |
| Thesis writing | Any | |

---

## Recommended Development Workflow

1. Write and test code in VS Code (push to GitHub after each session)
2. In Colab: `!git clone https://github.com/YOUR_USERNAME/agentic-rag-appeals.git`
3. Mount Google Drive: `drive.mount('/content/drive')` — save checkpoints and index here
4. Pull latest code: `!git pull` before each Colab session
5. After GPU run: download results to `evaluation/results/` and commit to GitHub
