from knowledge_retrieval.langchain_tool import make_search_tool
from knowledge_retrieval.retrieval import Store


def test_tool_schema_and_invocation_keep_tenant_bound_outside_model_arguments(tmp_path):
    store = Store(str(tmp_path / "knowledge.db"))
    store.ingest("demo", "shared.md", "incident handoff checklist")
    store.ingest("private", "secret.md", "incident handoff private credentials")
    search = make_search_tool(store, "demo")

    assert search.name == "search_internal_knowledge"
    assert set(search.args) == {"query"}
    expected = store.search("demo", "incident handoff")
    assert expected
    assert search.invoke({"query": "incident handoff"}) == expected
    # Extra model arguments cannot override the tenant captured by the tool.
    assert search.invoke({"query": "incident handoff", "tenant": "private"}) == expected
    assert all(hit["tenant"] == "demo" for hit in expected)
