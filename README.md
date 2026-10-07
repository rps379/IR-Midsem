# IR-Midsem
IR Midsem 2410110601
# Scientific Claim Verification using Information Retrieval

## 1. Project Overview

This project is a scientific claim verification system that combines classical Information Retrieval (IR) techniques with neural NLP methods.

The system accepts an arbitrary scientific claim as input and searches a scientific corpus for relevant documents and evidence sentences. It then reranks the retrieved evidence, applies Natural Language Inference (NLI), and produces one of three final verdicts:

- `SUPPORT`
- `CONTRADICT`
- `ABSTAIN`

The project uses the SciFact dataset for development and evaluation, while the live application accepts arbitrary user-entered scientific claims.

The main retrieval pipeline is:

```text
Scientific Claim
       |
       v
Text Preprocessing
       |
       v
TF-IDF Document Retrieval
       |
       v
Top 10 Documents
       |
       v
BM25 Sentence Retrieval
       |
       v
~150 Evidence Candidates
       |
       v
MS-MARCO Cross-Encoder
       |
       v
Top 20 Evidence Candidates
       |
       v
NLI Classification
       |
       v
SUPPORT / NEUTRAL / CONTRADICT
       |
       v
Verdict + Verification
       |
       v
SUPPORT / CONTRADICT / ABSTAIN



IR-Midsem/
│
├── data/
│   └── corpus.jsonl
│
├── src/
│   ├── __init__.py
│   ├── corpus.py
│   ├── preprocess.py
│   ├── inverted_index.py
│   ├── weighting.py
│   ├── retriever.py
│   ├── evidence.py
│   ├── reranker.py
│   ├── pairs.py
│   ├── nli.py
│   ├── verdict.py
│   ├── verifier.py
│   └── explain.py
│
├── testing/
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_inverted_index.py
│   ├── test_retriever.py
│   ├── test_pipeline.py
│   ├── test_pairs.py
│   ├── test_nli.py
│   ├── test_verifier.py
│   └── test_explain.py
│
├── app.py
├── requirements.txt
├── README.md
└── .venv/


Requirements:

torch
transformers
nltk
numpy
pytest