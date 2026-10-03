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


def test_vector_ranking_normalizes_magnitude_and_omits_nonpositive_scores(tmp_path):
    vectors = {"query": [1.0, 0.0], "closest": [1.0, 0.0], "large": [10.0, 10.0],
               "opposite": [-1.0, 0.0], "orthogonal": [0.0, 1.0], "zero": [0.0, 0.0]}
    store = Store(str(tmp_path / "vectors.db"), embed=vectors.__getitem__)
    for source in ["zero", "large", "orthogonal", "opposite", "closest"]:
        store.ingest("demo", source, source)
    # No token overlap: ranking and filtering must come entirely from vectors.
    assert [hit["source"] for hit in store.search("demo", "query")] == ["closest", "large"]


@pytest.mark.parametrize("magnitude", [1e-300, 1e300, 1.7e308])
def test_vector_ranking_handles_extreme_finite_magnitudes(tmp_path, magnitude):
    vectors = {"query": [magnitude, magnitude], "closest": [magnitude, magnitude],
               "partial": [magnitude, 0.0], "opposite": [-magnitude, -magnitude],
               "zero": [0.0, 0.0]}
    store = Store(str(tmp_path / "vectors.db"), embed=vectors.__getitem__)
    for source in ["zero", "partial", "opposite", "closest"]:
        store.ingest("demo", source, source)
    assert [hit["source"] for hit in store.search("demo", "query")] == ["closest", "partial"]


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


def test_vector_search_filters_tenants_before_dimension_checks(tmp_path):
    def embed(text):
        # Simulate a different embedding model used for another tenant.
        return [1.0, 0.0, 0.0] if "private" in text else [1.0, 0.0]

    store = Store(str(tmp_path / "vectors.db"), embed=embed)
    store.ingest("demo", "runbook", "outage response procedure")
    store.ingest("other", "secret", "private outage response procedure")
    # No lexical overlap: only the vector path can find this synonym.
    hits = make_search_tool(store, "demo").invoke({"query": "downtime"})
    assert [(hit["tenant"], hit["source"], hit["text"]) for hit in hits] == [
        ("demo", "runbook", "outage response procedure")]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_embedding_reingestion_preserves_original_index(tmp_path, invalid):
    store = Store(str(tmp_path / "vectors.db"), embed=lambda text: [1.0, 0.0])
    store.ingest("demo", "runbook", "original outage procedure")
    before = store.search("demo", "outage")
    store.embed = lambda text: [invalid, 0.0]
    with pytest.raises(ValueError, match="embedding values must be finite"):
        store.ingest("demo", "runbook", "replacement outage procedure")
    store.embed = lambda text: [1.0, 0.0]
    assert store.search("demo", "outage") == before


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("location", ["query", "stored"])
def test_search_rejects_nonfinite_embeddings(tmp_path, invalid, location):
    store = Store(str(tmp_path / "vectors.db"), embed=lambda text: [1.0, 0.0])
    store.ingest("demo", "runbook", "outage response procedure")
    if location == "query":
        store.embed = lambda text: [invalid, 0.0]
    else:
        import json
        with sqlite3.connect(store.path) as db:
            db.execute("UPDATE vectors SET vector=?", (json.dumps([invalid, 0.0]),))
    with pytest.raises(ValueError, match="embedding values must be finite"):
        store.search("demo", "downtime")


@pytest.mark.parametrize("location", ["ingest", "query", "stored"])
def test_empty_embeddings_are_rejected(tmp_path, location):
    store = Store(str(tmp_path / "vectors.db"), embed=lambda text: [1.0, 0.0])
    store.ingest("demo", "runbook", "original outage procedure")
    if location == "stored":
        with sqlite3.connect(store.path) as db:
            db.execute("UPDATE vectors SET vector='[]'")
    else:
        store.embed = lambda text: []
    with pytest.raises(ValueError, match="at least one dimension"):
        if location == "ingest":
            store.ingest("demo", "runbook", "replacement outage procedure")
        else:
            store.search("demo", "downtime")
    if location == "ingest":
        store.embed = lambda text: [1.0, 0.0]
        assert store.search("demo", "original")[0]["text"] == "original outage procedure"


@pytest.mark.parametrize("location", ["query", "stored"])
def test_embedding_dimension_mismatch_requires_reindexing(tmp_path, location):
    store = Store(str(tmp_path / "vectors.db"), embed=lambda text: [1.0, 0.0])
    store.ingest("demo", "runbook", "outage response procedure")
    if location == "query":
        store.embed = lambda text: [1.0]
    else:
        with sqlite3.connect(store.path) as db:
            db.execute("UPDATE vectors SET vector='[1.0]'")
    with pytest.raises(ValueError, match="embedding dimensions changed; reindex documents"):
        store.search("demo", "downtime")
    # A failed search must leave the index usable after fixing the model or reindexing.
    store.embed = lambda text: [1.0, 0.0]
    store.ingest("demo", "runbook", "outage response procedure")
    assert store.search("demo", "downtime")[0]["source"] == "runbook"
