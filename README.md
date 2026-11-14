# Enterprise GraphRAG Evaluation Harness

An end-to-end Python project for LLM-based information extraction,
knowledge graph construction, hybrid document retrieval,
graph-enhanced retrieval-augmented generation (GraphRAG), and
quantitative evaluation.

## Key Features

- Structured entity and relationship extraction using LLMs.
- Pydantic validation of extraction results.
- Knowledge graph construction using NetworkX.
- Multi-hop graph traversal and relationship queries.
- TF-IDF and BM25 document retrieval.
- Reciprocal Rank Fusion (RRF) of TF-IDF and BM25 rankings.
- Multi-source breadth-first graph-context retrieval.
- Relationship provenance with source document IDs and sentences.
- Sentence Transformers embedding support.
- Evidence-grounded LLM answer generation.
- FastAPI REST endpoints.
- Entity and relationship precision, recall and F1 evaluation.
- Retrieval benchmarking using Precision@K, Recall@K, MRR and NDCG.
- LLM-as-a-Judge evaluation for faithfulness and answer relevance.
- Reproducible answer-quality reports with original evidence snapshots.
- Automated testing, linting, type checking and security scanning.

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.13 |
| LLM | OpenAI Python SDK |
| Validation | Pydantic v2 |
| Graph | NetworkX |
| Embeddings | Sentence Transformers, TF-IDF |
| Retrieval | TF-IDF, BM25, RRF |
| API | FastAPI, Uvicorn |
| Evaluation | Precision, Recall, F1, MRR, NDCG |
| Answer quality | LLM-as-a-Judge, faithfulness, relevance |
| Testing | Pytest |
| Code quality | Ruff, Mypy |
| Security | Bandit |
| CI | GitHub Actions |

## Project Structure

- `data/`: Source enterprise documents.
- `evals/`: Golden datasets and generated evaluation reports.
- `src/app.py`: FastAPI application.
- `src/config.py`: Application configuration.
- `src/data_processor.py`: Document validation and cleaning.
- `src/evaluator.py`: Entity and relationship extraction evaluation.
- `src/extractor.py`: LLM entity and relationship extraction.
- `src/graph_builder.py`: Knowledge graph construction.
- `src/graph_engine.py`: Graph querying and traversal.
- `src/graph_loader.py`: Exported graph loading.
- `src/retriever.py`: TF-IDF, embedding and BM25 retrieval.
- `src/pipeline.py`: GraphRAG answer generation.
- `src/retrieval_evaluator.py`: Retrieval evaluation metrics.
- `src/retrieval_comparison.py`: TF-IDF, BM25 and RRF comparison.
- `src/answer_evaluator.py`: LLM-as-a-Judge engine.
- `src/judge_prompts.py`: Faithfulness and relevance rubrics.
- `src/run_evaluation.py`: Extraction evaluation runner.
- `src/run_graph_extraction.py`: Graph extraction runner.
- `src/run_llm_evaluation.py`: Live LLM extraction evaluation.
- `src/run_retrieval_evaluation.py`: Retrieval evaluation runner.
- `src/run_answer_evaluation.py`: GraphRAG answer-quality benchmark.
- `src/run_saved_answer_judging.py`: Judge previously generated answers.
- `tests/`: Automated test suite.

## Prerequisites

- Python 3.13
- Git and pip
- OpenAI API key for live LLM extraction and answer generation

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sharathxdev/llm-kg-eval-harness.git
cd llm-kg-eval-harness
```

### 2. Create a virtual environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

For application execution:

```bash
python -m pip install -r requirements.txt
```

For development and quality checks:

```bash
python -m pip install -r requirements-dev.txt
```

### 4. Configure environment variables

Create a local environment file:

```bash
cp .env.example .env
```

Configure:

- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_MODEL`: Model name, defaulting to `gpt-4o-mini`.

Never commit your real API key. The `.env` file is excluded from Git.

## Running the Application

### Run automated tests

```bash
python -m pytest -q
```

### Start FastAPI

```bash
uvicorn src.app:app --reload
```

API base URL: `http://127.0.0.1:8000`

Interactive API documentation: `http://127.0.0.1:8000/docs`

### Available API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/retrieve` | Retrieve documents and graph context |
| POST | `/query` | Retrieve evidence and generate an LLM answer |

## Retrieval Evaluation

The retrieval benchmark contains 10 queries and uses a corpus of
15 synthetic enterprise documents.

Evaluate TF-IDF, BM25 and RRF at K = 3, 5 and 10:

```bash
for method in tfidf bm25 rrf; do
    for k in 3 5 10; do
        python -m src.run_retrieval_evaluation \
            --k "$k" \
            --hops 2 \
            --retrieval-method "$method"
    done
done
```

Generate the comparison report:

```bash
python -m src.retrieval_comparison
```

## Retrieval Benchmark Results

Results from the current synthetic benchmark:

| Method | K | Precision | Recall | Hit Rate | MRR | NDCG |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 3 | 0.8000 | 0.5950 | 1.0000 | 1.0000 | 0.8531 |
| BM25 | 3 | 0.8000 | 0.5658 | 1.0000 | 1.0000 | 0.8531 |
| RRF | 3 | 0.7667 | 0.5617 | 1.0000 | 1.0000 | 0.8296 |
| TF-IDF | 5 | 0.7000 | 0.8250 | 1.0000 | 1.0000 | 0.8889 |
| BM25 | 5 | 0.6200 | 0.7025 | 1.0000 | 1.0000 | 0.8125 |
| RRF | 5 | 0.6600 | 0.7717 | 1.0000 | 1.0000 | 0.8509 |
| TF-IDF | 10 | 0.4200 | 0.9417 | 1.0000 | 1.0000 | 0.9237 |
| BM25 | 10 | 0.4100 | 0.9250 | 1.0000 | 1.0000 | 0.8906 |
| RRF | 10 | 0.4200 | 0.9417 | 1.0000 | 1.0000 | 0.9188 |

TF-IDF achieved the strongest overall ranking results in this benchmark.
RRF was competitive with BM25 but did not consistently outperform TF-IDF.

Graph entity evaluation reported an F1 score of 1.0 for all three
methods. This measures query-entity matching against the evaluation
expectations, not relationship-extraction accuracy.

## LLM Extraction and Evaluation

Run graph extraction:

```bash
python -m src.run_graph_extraction
```

Run LLM extraction evaluation:

```bash
python -m src.run_llm_evaluation
```

These commands make live LLM API requests and may incur API charges.

## GraphRAG Answer-Quality Evaluation

The answer-quality evaluator measures two dimensions:

**Faithfulness (1–5)**

Measures whether factual claims in the generated answer are supported
by the retrieved document and knowledge graph context.

**Answer relevance (1–5)**

Measures whether the answer directly addresses the user's question.

The overall score is the arithmetic mean of the two dimension scores.

### Generate GraphRAG answers

Run the complete 10-question benchmark:

```bash
python -m src.run_answer_evaluation \
    --judge-mode mock \
    --retrieval-method tfidf \
    --top-k 3 \
    --hops 2 \
    --output evals/reports/answer_quality_full.json
```

Important: mock judge mode only replaces the judge with a deterministic
lexical-overlap heuristic. GraphRAG answer generation still calls the
configured LLM and requires an OpenAI API key.

The mock scores are not real LLM judgments.

### Run real LLM-as-a-Judge evaluation

Evaluate the previously generated answers:

```bash
python -m src.run_saved_answer_judging \
    --input evals/reports/answer_quality_full.json \
    --output evals/reports/answer_quality_full_llm_judged.json
```

The judge evaluates faithfulness and relevance independently.

For reports created by the current answer runner, the original
retrieval evidence is preserved and reused during judging. Older
reports without an evidence snapshot are supported through a
retrieval fallback.

### Run generation and judging in one operation

The answer runner can also use the real LLM judge directly:

```bash
python -m src.run_answer_evaluation \
    --judge-mode llm \
    --retrieval-method tfidf \
    --top-k 3 \
    --hops 2 \
    --output evals/reports/answer_quality_full_llm.json
```

This performs answer generation and both judge evaluations for every
sample in a single run.

These workflows make live LLM API requests and may incur API charges.

### Answer-quality benchmark results

Results from the completed 10-query benchmark using 15 synthetic
enterprise documents, TF-IDF retrieval, top K = 3 and two-hop graph
context:

| Metric | Result |
|---|---:|
| Questions | 10 |
| Successful generations | 10/10 |
| Successful LLM judgments | 10/10 |
| Average faithfulness | 5.0/5 |
| Average answer relevance | 4.8/5 |
| Average overall answer quality | 4.9/5 |
| Average generation latency | 2,834 ms |
| Average judge latency | 5,303 ms |

The LLM judge used `gpt-4o-mini`.

These values are results from one benchmark execution. They are
model-judged scores, not independently verified ground-truth accuracy.
The benchmark is small and synthetic, and should be expanded with
independently annotated enterprise questions and human review.

## Evaluation Metrics

### Retrieval

- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR@K)
- Normalized Discounted Cumulative Gain (NDCG@K)

### Entity and relationship extraction

- True positives
- False positives
- False negatives
- Precision
- Recall
- F1 score

### Generated answer quality

- Faithfulness
- Answer relevance
- Overall answer-quality score
- Generation latency
- Judge latency

## Code Quality

Install the development tools:

```bash
python -m pip install -r requirements-dev.txt
```

Run Ruff on the answer-evaluation modules:

```bash
ruff check \
    src/answer_evaluator.py \
    src/judge_prompts.py \
    src/run_answer_evaluation.py \
    src/run_saved_answer_judging.py \
    tests/test_answer_evaluator.py \
    tests/test_run_answer_evaluation.py \
    tests/test_run_saved_answer_judging.py
```

Check formatting:

```bash
ruff format --check \
    src/answer_evaluator.py \
    src/judge_prompts.py \
    src/run_answer_evaluation.py \
    src/run_saved_answer_judging.py \
    tests/test_answer_evaluator.py \
    tests/test_run_answer_evaluation.py \
    tests/test_run_saved_answer_judging.py
```

Run Mypy:

```bash
mypy --follow-imports=skip --ignore-missing-imports \
    src/answer_evaluator.py \
    src/judge_prompts.py \
    src/run_answer_evaluation.py \
    src/run_saved_answer_judging.py
```

Run Bandit:

```bash
bandit -q \
    src/answer_evaluator.py \
    src/judge_prompts.py \
    src/run_answer_evaluation.py \
    src/run_saved_answer_judging.py
```

These focused checks cover the new answer-evaluation modules.

## Testing and Continuous Integration

Run the complete suite:

```bash
python -m pytest -q
```

GitHub Actions automatically runs:

- Ruff lint and formatting checks.
- Mypy type checks for the answer-evaluation modules.
- Bandit security scanning.
- The complete Pytest suite.

The automated workflow runs on pushes and pull requests targeting
the `main` branch.

The full test suite covers data cleaning, graph construction,
multi-hop traversal, extraction, retrieval, evaluation metrics,
benchmark comparison, answer evaluation, saved-answer judging,
GraphRAG generation and API behavior.

## Current Limitations

- The benchmark datasets are synthetic and relatively small.
- LLM-as-a-Judge scores may be biased and are not a substitute for
  human-verified ground truth.
- Entity detection relies on known entity names, IDs and configured
  aliases in the query.
- RRF currently combines TF-IDF and BM25 rankings, not dense
  Sentence Transformer rankings.
- Rule-based graph relationship extraction uses configured entity
  names and a finite set of relationship patterns.
- Production deployment requires additional authentication, rate
  limiting, monitoring, persistent storage and security hardening.

## Future Improvements

- Dense-vector and sparse BM25 rank fusion.
- Larger and more diverse golden datasets.
- Human-annotated answer-quality ground truth.
- LLM judge calibration and multi-judge agreement analysis.
- Per-query and statistical benchmark analysis.
- Improved entity linking and alias resolution.
- Persistent vector and graph storage.
- Production deployment and observability.
