"""Expose retrieval to a LangChain agent without letting the model choose a tenant."""
from langchain_core.tools import tool

from .retrieval import Store


def make_search_tool(store: Store, tenant: str):
    @tool("search_internal_knowledge")
    def search_internal_knowledge(query: str) -> list[dict]:
        """Search approved internal knowledge and return cited source excerpts."""
        return store.search(tenant, query)

    return search_internal_knowledge
