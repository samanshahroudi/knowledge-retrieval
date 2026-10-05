from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from knowledge_retrieval.retrieval import grounded_answer, openai_embed


@pytest.mark.parametrize("operation", ["answer", "embedding"])
@pytest.mark.parametrize("fail", [False, True])
def test_provider_clients_close_after_success_or_failure(monkeypatch, operation, fail):
    client = Mock()
    monkeypatch.setattr("openai.OpenAI", Mock(return_value=client))
    if operation == "answer":
        call = client.responses.create
        call.return_value = SimpleNamespace(output_text="Recovery [one].")
        invoke = lambda: grounded_answer("Recovery?", [{"id": "one", "text": "Recovered"}])
        expected = "Recovery [one]."
    else:
        call = client.embeddings.create
        call.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])
        invoke = lambda: openai_embed("Recovered")
        expected = [1.0, 0.0]
    if fail:
        call.side_effect = RuntimeError("provider unavailable")
        with pytest.raises(RuntimeError, match="provider unavailable"):
            invoke()
    else:
        assert invoke() == expected
    client.close.assert_called_once_with()


def test_answer_without_hits_does_not_open_provider_client(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("openai.OpenAI", factory)
    assert grounded_answer("Recovery?", []) == "I could not find relevant material."
    factory.assert_not_called()
