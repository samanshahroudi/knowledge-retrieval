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
