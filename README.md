# Enterprise GraphRAG Evaluation Harness

An end-to-end Python project for LLM-based information extraction, knowledge graph construction, document retrieval, graph-enhanced retrieval-augmented generation (GraphRAG), and quantitative evaluation.

## Key Features

- Structured entity and relationship extraction using LLMs.
- Pydantic validation of extraction results.
- Knowledge graph construction using NetworkX.
- Multi-hop graph traversal and relationship queries.
- TF-IDF and BM25 document retrieval.
- Sentence Transformers embedding support.
- Evidence-grounded LLM answer generation.
- FastAPI REST endpoints.
- Extraction and retrieval evaluation.
- Automated testing with Pytest.

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.13 |
| LLM | OpenAI Python SDK |
| Validation | Pydantic v2 |
| Graph | NetworkX |
| Embeddings | Sentence Transformers |
| Retrieval | TF-IDF, BM25 |
| API | FastAPI, Uvicorn |
| Evaluation | Precision, Recall, F1, MRR, NDCG |
| Testing | Pytest |
| CI | GitHub Actions |

## Project Structure

- `data/`: Source enterprise documents.
- `evals/`: Golden datasets and evaluation reports.
- `src/app.py`: FastAPI application.
- `src/config.py`: Application configuration.
- `src/data_processor.py`: Document validation and cleaning.
- `src/evaluator.py`: Extraction evaluation.
- `src/extractor.py`: LLM entity and relationship extraction.
- `src/graph_builder.py`: Knowledge graph construction.
- `src/graph_engine.py`: Graph querying and traversal.
- `src/graph_loader.py`: Exported graph loading.
- `src/pipeline.py`: GraphRAG answer generation.
- `src/retriever.py`: TF-IDF, embedding and BM25 retrieval.
- `src/retrieval_evaluator.py`: Retrieval metrics.
- `src/retrieval_comparison.py`: TF-IDF and BM25 comparison.
- `src/run_evaluation.py`: Extraction evaluation runner.
- `src/run_graph_extraction.py`: Graph extraction runner.
- `src/run_llm_evaluation.py`: Live LLM evaluation runner.
- `src/run_retrieval_evaluation.py`: Retrieval evaluation runner.
- `tests/`: Automated tests.

## Prerequisites

- Python 3.13
- Git and pip
- OpenAI API key for live LLM extraction and answer generation

## Getting Started

### 1. Clone the repository

    git clone https://github.com/sharathxdev/llm-kg-eval-harness.git
    cd llm-kg-eval-harness

### 2. Create a virtual environment

    python3.13 -m venv .venv
    source .venv/bin/activate

### 3. Install dependencies

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

### 4. Configure environment variables

Create a local environment file:

    cp .env.example .env

Configure the following values inside `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_MODEL`: `gpt-4o-mini` by default.

Never commit your real API key. The `.env` file is excluded from Git.

Retrieval benchmarks and automated tests do not require a live OpenAI API key.

## Running the Application

### Run automated tests

    python -m pytest -q

### Start the FastAPI server

    uvicorn src.app:app --reload

API base URL: `http://127.0.0.1:8000`

Interactive API documentation: `http://127.0.0.1:8000/docs`

### Available API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/retrieve` | Retrieve documents and graph context |
| POST | `/query` | Retrieve evidence and generate an LLM answer |

## Retrieval Evaluation

The retrieval benchmark contains 10 queries and uses a corpus of 15 documents.

Evaluate TF-IDF and BM25 at K = 3, 5 and 10:

    for method in tfidf bm25; do
        for k in 3 5 10; do
            python -m src.run_retrieval_evaluation \
                --k "$k" \
                --hops 2 \
                --retrieval-method "$method"
        done
    done

Generate the comparison report:

    python -m src.retrieval_comparison

## Benchmark Results

Results from the current synthetic benchmark (10 queries, 15 documents):

| Method | K | Precision | Recall | MRR | NDCG |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 3 | 0.8000 | 0.5950 | 1.0000 | 0.8531 |
| BM25 | 3 | 0.8000 | 0.5658 | 1.0000 | 0.8531 |
| TF-IDF | 5 | 0.7000 | 0.8250 | 1.0000 | 0.8889 |
| BM25 | 5 | 0.6200 | 0.7025 | 1.0000 | 0.8125 |
| TF-IDF | 10 | 0.4200 | 0.9417 | 1.0000 | 0.9237 |
| BM25 | 10 | 0.4100 | 0.9250 | 1.0000 | 0.8906 |

TF-IDF achieved higher recall and NDCG at the evaluated K values. Both methods achieved a Hit Rate and MRR of 1.0.

Graph entity evaluation reported an F1 score of 1.0 for both methods on this dataset.

These results describe this particular synthetic benchmark. They are not a guarantee of performance on larger or real-world enterprise datasets.

## LLM Extraction and Evaluation

Run graph extraction:

    python -m src.run_graph_extraction

Run LLM extraction evaluation:

    python -m src.run_llm_evaluation

These commands make real LLM API requests and may incur API charges. Configure your API key before running them.

## Evaluation Metrics

Document retrieval:
- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR@K)
- Normalized Discounted Cumulative Gain (NDCG@K)

Graph entity and extraction evaluation:
- True positives
- False positives
- False negatives
- Precision
- Recall
- F1 score

## Testing

Run the full test suite:

    python -m pytest -q

The tests cover document processing, graph construction, graph traversal, extraction validation, retrieval, evaluation metrics, benchmark comparison, GraphRAG generation, and API behavior.

## Current Limitations

- The benchmark datasets are synthetic and relatively small.
- Entity detection relies on known entity names, IDs and configured aliases in the query.
- TF-IDF and BM25 are evaluated separately; their scores are not fused into one ranking.
- The automatic embedding provider prefers Sentence Transformers and falls back to TF-IDF when the transformer model cannot be loaded.
- Production deployment would require additional authentication, rate limiting, monitoring, persistent storage and security hardening.

## Future Improvements

- Dense and sparse retrieval score fusion.
- Larger and more diverse golden datasets.
- Per-query and statistical benchmark analysis.
- Improved entity linking and alias resolution.
- Persistent vector and graph storage.
- Production deployment and observability.
