# Tenant-aware knowledge retrieval

## The problem

A useful RAG system must retrieve the right evidence and must never mix customers' documents. This service puts tenant filtering in the SQL query before ranking, chunks documents deterministically, combines SQLite FTS5 BM25 with token-overlap ranking through reciprocal rank fusion, and offers optional OpenAI embedding similarity and a cited answer.

## Architecture and how it works

`file → fixed-size word chunks → SQLite + FTS5 → tenant-filtered candidates → reciprocal rank fusion → cited answer`

Ingestion replaces prior chunks for the same tenant and source in one transaction. A stable chunk ID lets evaluations point to the exact source span. Search returns chunk text and source names, not just an opaque answer. `grounded_answer` sends only retrieved excerpts to the OpenAI Responses API and asks for chunk-ID citations. The query and excerpts are still untrusted content.

Ranking ties use chunk IDs as a stable order, so reingesting unchanged content preserves result order, including at candidate cutoffs.

Embeddings must have at least one dimension, and their values must be finite. Invalid values reject ingestion without replacing the existing index, and searches reject invalid query or stored vectors rather than silently dropping results.
Vector similarity normalizes with scaling so very large or small finite magnitudes preserve ranking.

Chunk IDs escape colons and percent signs in tenant/source names before hashing so distinct pairs cannot share the same hash input. Reingest existing sources whose tenant or source contains those characters to update their IDs; other IDs stay unchanged.

## Concepts and choices

This project deliberately exposes retrieval mechanics instead of hiding them behind LangChain. SQLite FTS5 gives a strong lexical baseline with no external service. Token overlap supplies a second signal; optional OpenAI embeddings add a semantic signal. Reciprocal rank fusion combines the rankings. `langchain_tool.py` wraps the retriever as a LangChain tool with the tenant fixed outside the model's arguments. The separate evaluation lab can score ranked results exported from this retriever. Source code is in `knowledge_retrieval/retrieval.py`, CLI in `knowledge_retrieval/cli.py`, and sample documents in `fixtures/`.

## Run and example

From this repository after `python -m pip install -e ".[dev]"`:

```bash
python -m knowledge_retrieval.cli --db knowledge.db ingest --tenant demo fixtures/handoff.md
python -m knowledge_retrieval.cli --db knowledge.db search --tenant demo 'incident handoff checklist'
```

Ingestion defaults to the file basename as its source. For files with the same basename, pass distinct stable names, for example `ingest --tenant demo --source team-one/runbook.md team-one/runbook.md`. Reuse the same `--source` to replace that document; blank source names are rejected. Both commands reject blank tenant names before creating a database or reading input files. The store also rejects blank tenants at the search boundary, including empty queries and zero-result limits.

Add `--answer` to the search command for an OpenAI-generated answer; set `OPENAI_API_KEY` first. Ingest `security.md` under another tenant and confirm that `demo` searches cannot see it.
The search command rejects blank queries before creating a database or requesting an answer.
Pass `--embeddings` on both ingest and search to include vector similarity. Keep that mode consistent for a database; the demo does not yet version embedding models.

## Trade-offs, limitations, and next production steps

Word-count chunking can split a sentence and ignores document structure. The token-overlap signal is not semantic and may fail on synonyms. The vector path makes one paid API call per chunk and does not batch requests; its model version is not stored. Add batched embedding generation with versioned vectors, reranking, document-level ACLs, incremental ingestion, source freshness, and citation verification before a production rollout. FTS query construction limits terms to alphanumeric tokens, but unbounded ingestion still needs quotas and MIME validation. A model can hallucinate citations despite the prompt; validate cited IDs and evaluate answer faithfulness separately.

## Interview preparation

Be ready to explain pre-filtering versus post-filtering for security, chunking trade-offs, BM25, reciprocal rank fusion, recall@k, answer faithfulness, and why retrieval quality should be measured separately from generation quality.

## Verify

Run `python -m pytest -q` and `python -m ruff check .` from this repository.
