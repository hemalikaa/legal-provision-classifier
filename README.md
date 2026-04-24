# Legal Provision Classification & Contract Risk Analyzer

A system that classifies legal contract provisions into 100 categories and assesses deviation risk using five approaches from classical ML to fine-tuned BERT and RAG-based LLMs.

**Course:** IE 7374 — Generative AI | Northeastern University | Spring 2026

---

## Project Overview

This project addresses the research question: *To what extent can retrieval-augmented generation with open-source LLMs match or exceed fine-tuned domain-specific baselines for legal provision classification?*

We compare five approaches on the LEDGAR dataset (60,000 SEC contract provisions, 100 categories):

| Approach | Micro-F1 | Macro-F1 |
|---|---|---|
| LEGAL-BERT (published baseline) | ~0.87 | ~0.77 |
| TF-IDF + Logistic Regression | 0.8726 | 0.8112 |
| Zero-shot Llama-3.3-70B | 0.5150 | 0.4018 |
| Chain-of-Thought Llama-3.3-70B | 0.3850 | 0.3199 |
| RAG Few-shot Llama-3.3-70B | TBD | TBD |
| BERT-base Fine-tuned | 0.8715 | 0.7862 |

---

## Repository Structure

```
legal-provision-classifier/
├── config/
│   └── config.yaml          # All hyperparameters and paths
├── src/
│   ├── data_loader.py       # LEDGAR dataset loading + reference embeddings
│   ├── bert_classifier.py   # BERT fine-tuning and evaluation
│   ├── tfidf_baseline.py    # TF-IDF + Logistic Regression baseline
│   ├── zeroshot_classifier.py  # Zero-shot Llama via Groq
│   ├── cot_classifier.py    # Chain-of-Thought Llama via Groq
│   ├── rag_pipeline.py      # RAG few-shot + ChromaDB via Groq
│   ├── risk_detector.py     # Cosine similarity risk assessment
│   └── evaluate.py          # Unified evaluation + plots
├── app/
│   └── streamlit_app.py     # Interactive demo app
├── data/                    # Reference embeddings (auto-generated)
├── models/                  # Saved model weights (auto-generated)
├── results/                 # Evaluation outputs (auto-generated)
├── notebooks/               # Exploration notebooks
└── requirements.txt
```

---

## Setup

**1. Clone the repository:**
```bash
git clone https://github.com/hemalikaa/legal-provision-classifier.git
cd legal-provision-classifier
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Set your Groq API key:**
```bash
# Windows
$env:GROQ_API_KEY = "your_groq_api_key"

# Mac/Linux
export GROQ_API_KEY="your_groq_api_key"
```

---

## Running the Pipeline

Run in this order:

**Step 1 — Build reference embeddings (run once):**
```bash
python src/data_loader.py
```

**Step 2 — Train and evaluate TF-IDF baseline:**
```bash
python src/tfidf_baseline.py
```

**Step 3 — Run zero-shot evaluation:**
```bash
python src/zeroshot_classifier.py
```

**Step 4 — Run Chain-of-Thought evaluation:**
```bash
python src/cot_classifier.py
```

**Step 5 — Build RAG index and evaluate:**
```bash
python src/rag_pipeline.py
```

**Step 6 — Load BERT results and generate all plots:**
```bash
python src/evaluate.py
```

**Step 7 — Launch Streamlit demo:**
```bash
streamlit run app/streamlit_app.py
```

---

## Requirements

- Python 3.11+
- Groq API key (free tier at console.groq.com)
- ~4GB disk space for models and embeddings
- GPU recommended for BERT fine-tuning (CPU works but slower)

See `requirements.txt` for all dependencies with pinned versions.

---

## Dataset

**LEDGAR** via LexGLUE benchmark (Chalkidis et al., 2022)
- 60,000 contract provisions from SEC EDGAR filings
- 100 provision categories
- HuggingFace: `coastalcph/lex_glue`, config `ledgar`

Downloaded automatically when you run `src/data_loader.py`.

---

## Results

All evaluation results are saved to `results/` after running `evaluate.py`:
- `bert_results.json` — BERT test set metrics
- `tfidf_results.json` — TF-IDF test set metrics
- `zeroshot_results.json` — Zero-shot metrics
- `cot_results.json` — CoT metrics
- `rag_results.json` — RAG metrics
- `comparison_table.csv` — Full comparison table
- `plots/` — F1 comparison chart, per-class F1, risk distribution

---

## Team

- Hemalikaa Thirumavalavan
- Frank Amankwah

**Course:** IE 7374 Generative AI, Northeastern University Vancouver, Spring 2026

---

## Gen AI Usage

Claude (Anthropic) was used to assist with code structure, debugging, and documentation. All technical decisions, experimental design, and analysis were made by the team. All AI-generated code was reviewed and understood before use.
