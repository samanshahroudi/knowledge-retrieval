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


@pytest.mark.parametrize("answer", ["", " \t\n"])
def test_blank_provider_answer_is_rejected_without_cli_output(
    tmp_path, monkeypatch, capsys, answer,
):
    from knowledge_retrieval.cli import main
    from knowledge_retrieval.retrieval import Store

    path = tmp_path / "knowledge.db"
    Store(str(path)).ingest("demo", "runbook.md", "incident recovery procedure")
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output_text=answer)
    monkeypatch.setattr("openai.OpenAI", Mock(return_value=client))
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(path), "search",
                                    "--tenant", "demo", "incident", "--answer"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "model returned no answer text" in captured.err
    assert "Traceback" not in captured.err
    client.responses.create.assert_called_once()
    client.close.assert_called_once_with()
