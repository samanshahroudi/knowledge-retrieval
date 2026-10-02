import hashlib
import sqlite3

import pytest

from knowledge_retrieval.langchain_tool import make_search_tool
from knowledge_retrieval.retrieval import Store


def test_tenant_filter_and_reingestion(tmp_path):
    store = Store(str(tmp_path / "knowledge.db"))
    store.ingest("a", "one", "The incident handoff checklist names an owner and next action")
    store.ingest("b", "secret", "The incident handoff checklist contains a private token")
    assert [hit["source"] for hit in store.search("a", "incident handoff checklist")] == ["one"]
    assert [hit["source"] for hit in make_search_tool(store, "a").invoke({"query": "incident handoff"})] == ["one"]
    store.ingest("a", "one", "Replacement content only")
    assert store.search("a", "incident handoff checklist") == []


def test_vector_finds_synonym(tmp_path):
    def embed(text):
        return [1.0, 0.0] if "outage" in text or "downtime" in text else [0.0, 1.0]

    store = Store(str(tmp_path / "vectors.db"), embed=embed)
    store.ingest("a", "runbook", "outage response procedure")
    assert store.search("a", "downtime")[0]["source"] == "runbook"


def test_empty_reingestion_removes_stale_chunks(tmp_path):
    store = Store(str(tmp_path / "knowledge.db"))
    store.ingest("a", "runbook", "incident handoff checklist")
    assert store.ingest("a", "runbook", "   ") == 0
    assert store.search("a", "incident handoff") == []
    with pytest.raises(ValueError, match="positive"):
        store.ingest("a", "runbook", "text", size=0)


def test_negative_search_limit_is_rejected(tmp_path):
    store = Store(str(tmp_path / "knowledge.db"))
    store.ingest("a", "runbook", "incident handoff checklist")
    with pytest.raises(ValueError, match="limit"):
        store.search("a", "incident", limit=-1)


def test_zero_search_limit_skips_embedding(tmp_path):
    calls = []

    def embed(text):
        calls.append(text)
        return [1.0, 0.0]

    store = Store(str(tmp_path / "vectors.db"), embed=embed)
    store.ingest("a", "runbook", "outage response procedure")
    calls.clear()
    assert store.search("a", "downtime", limit=0) == []
    assert calls == []


@pytest.mark.parametrize("pairs", [
    [("a:b", "c"), ("a", "b:c")],
    [("a:b", "c"), ("a%3Ab", "c")],
])
def test_chunk_ids_distinguish_tenant_source_boundaries(tmp_path, pairs):
    store = Store(str(tmp_path / "knowledge.db"))
    for tenant, source in pairs:
        assert store.ingest(tenant, source, "incident handoff checklist") == 1
    hits = [store.search(tenant, "incident") for tenant, source in pairs]
    assert [rows[0]["source"] for rows in hits] == [source for tenant, source in pairs]
    assert hits[0][0]["id"] != hits[1][0]["id"]
    # Replacing one tenant's source must leave the other tenant's chunks intact.
    store.ingest(*pairs[0], "replacement content")
    assert store.search(pairs[0][0], "incident") == []
    assert store.search(pairs[1][0], "incident")[0]["id"] == hits[1][0]["id"]


def test_chunk_ids_remain_stable_without_reserved_characters(tmp_path):
    store = Store(str(tmp_path / "knowledge.db"))
    store.ingest("demo", "handoff.md", "incident handoff checklist")
    expected = hashlib.sha256(b"demo:handoff.md:0").hexdigest()[:20]
    assert store.search("demo", "handoff")[0]["id"] == expected


def test_failed_embedding_reingestion_preserves_original_index(tmp_path):
    def embed(text):
        if "failure" in text:
            raise RuntimeError("embedding unavailable")
        return [1.0, 0.0]

    store = Store(str(tmp_path / "vectors.db"), embed=embed)
    store.ingest("demo", "runbook", "original outage procedure", size=3)
    with sqlite3.connect(store.path) as db:
        before = {table: db.execute(f"SELECT * FROM {table}").fetchall()
                  for table in ("chunks", "chunk_fts", "vectors")}
    # Fail after writing a replacement chunk, exercising rollback of deletes and inserts.
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        store.ingest("demo", "runbook", "replacement outage procedure failure", size=3)
    with sqlite3.connect(store.path) as db:
        for table, rows in before.items():
            assert db.execute(f"SELECT * FROM {table}").fetchall() == rows
    assert store.search("demo", "original")[0]["text"] == "original outage procedure"


@pytest.mark.parametrize("use_embeddings", [False, True])
def test_tied_results_keep_order_after_reingestion(tmp_path, use_embeddings):
    store = Store(str(tmp_path / "knowledge.db"),
                  embed=(lambda text: [1.0, 0.0]) if use_embeddings else None)
    # Exceed the candidate cutoff to exercise tie ordering before truncation.
    for index in range(35):
        store.ingest("demo", f"source-{index}", "incident handoff checklist")
    before = store.search("demo", "incident")
    store.ingest("demo", before[0]["source"], before[0]["text"])
    assert store.search("demo", "incident") == before
