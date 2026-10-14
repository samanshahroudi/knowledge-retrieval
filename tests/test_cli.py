import sqlite3

import pytest

from knowledge_retrieval.cli import main
from knowledge_retrieval.retrieval import Store


@pytest.mark.parametrize("operation", ["ingest", "search", "answer"])
def test_provider_failures_are_usage_errors_without_partial_output(
    tmp_path, monkeypatch, capsys, operation,
):
    import httpx
    from openai import AuthenticationError

    path = tmp_path / "knowledge.db"
    store = Store(str(path), embed=lambda text: [1.0])
    store.ingest("demo", "runbook.md", "original incident procedure")
    error = AuthenticationError("private provider response", response=httpx.Response(
        401, request=httpx.Request("POST", "https://provider.example.test/responses")), body=None)

    def fail(*args):
        raise error

    monkeypatch.setattr("knowledge_retrieval.cli.openai_embed", fail)
    monkeypatch.setattr("knowledge_retrieval.cli.grounded_answer", fail)
    document = tmp_path / "runbook.md"
    document.write_text("replacement incident procedure", encoding="utf-8")
    command = "ingest" if operation == "ingest" else "search"
    flags = [] if operation == "answer" else ["--embeddings"]
    argument = str(document) if command == "ingest" else "incident"
    answer_flags = ["--answer"] if command == "search" else []
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(path), *flags, command,
                                    "--tenant", "demo", argument, *answer_flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "model provider request failed" in captured.err
    assert "private provider response" not in captured.err
    assert "Traceback" not in captured.err
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT body FROM chunks").fetchall() == [
            ("original incident procedure",)]
        assert db.execute("SELECT vector FROM vectors").fetchall() == [("[1.0]",)]


def test_successful_answer_prints_retrieved_hits_and_answer(tmp_path, monkeypatch, capsys):
    import json

    path = tmp_path / "knowledge.db"
    store = Store(str(path))
    store.ingest("demo", "runbook.md", "incident recovery procedure")
    hits = store.search("demo", "incident")

    def answer(question, evidence):
        assert question == "incident"
        assert evidence == hits
        return "Recovery confirmed."

    monkeypatch.setattr("knowledge_retrieval.cli.grounded_answer", answer)
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(path), "search",
                                    "--tenant", "demo", "incident", "--answer"])
    main()
    captured = capsys.readouterr()
    parsed, end = json.JSONDecoder().raw_decode(captured.out)
    assert parsed == hits
    assert captured.out[end:].strip() == "Recovery confirmed."
    assert captured.err == ""


@pytest.mark.parametrize("command", ["ingest", "search"])
def test_database_operation_errors_are_usage_errors(tmp_path, monkeypatch, capsys, command):
    path = tmp_path / "knowledge.db"
    store = Store(str(path))
    store.ingest("demo", "runbook.md", "original incident procedure")
    document = tmp_path / "runbook.md"
    document.write_text("replacement incident procedure", encoding="utf-8")
    with sqlite3.connect(path) as db:
        if command == "ingest":
            db.execute("CREATE TRIGGER fail_chunk BEFORE INSERT ON chunks "
                       "BEGIN SELECT RAISE(ABORT, 'chunk storage unavailable'); END")
    if command == "search":
        search = Store.search

        def fail_search(self, *args):
            # Fail a real read after the CLI has initialized its store.
            with sqlite3.connect(self.path) as db:
                db.execute("DROP TABLE chunk_fts")
            return search(self, *args)

        monkeypatch.setattr(Store, "search", fail_search)
    monkeypatch.setattr("knowledge_retrieval.cli.grounded_answer",
                        lambda *args: pytest.fail("failed retrieval must not request an answer"))
    argument = str(document) if command == "ingest" else "incident"
    flags = [] if command == "ingest" else ["--answer"]
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(path), command,
                                    "--tenant", "demo", argument, *flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"database operation failed for {path}" in captured.err
    assert "Traceback" not in captured.err
    if command == "ingest":
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT body FROM chunks").fetchall() == [
                ("original incident procedure",)]


@pytest.mark.parametrize("problem", ["directory", "parent_file", "invalid_database"])
def test_database_initialization_errors_are_usage_errors(tmp_path, monkeypatch, capsys, problem):
    path = tmp_path / "knowledge.db"
    if problem == "directory":
        path.mkdir()
    elif problem == "parent_file":
        path.write_text("not a directory")
        path = path / "knowledge.db"
    else:
        path.write_text("not a SQLite database")
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(path), "search",
                                    "--tenant", "demo", "incident"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"cannot open database {path}" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
@pytest.mark.parametrize("answer", [False, True])
def test_blank_query_is_usage_error_before_io(tmp_path, monkeypatch, capsys, query, answer):
    path = tmp_path / "new" / "knowledge.db"
    monkeypatch.setattr("knowledge_retrieval.cli.grounded_answer",
                        lambda *args: pytest.fail("blank query must not request an answer"))
    monkeypatch.setattr("sys.argv", ["retrieval", "--db", str(path), "search",
                                    "--tenant", "demo", query, *(["--answer"] if answer else [])])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "query cannot be blank" in captured.err
    assert not path.parent.exists()


@pytest.mark.parametrize("input_kind", ["missing", "directory", "invalid_utf8"])
def test_unreadable_input_is_usage_error_before_database_creation(
    tmp_path, monkeypatch, capsys, input_kind
):
    path = tmp_path / "runbook.md"
    if input_kind == "directory":
        path.mkdir()
    elif input_kind == "invalid_utf8":
        path.write_bytes(b"\xff")
    db = tmp_path / "new" / "knowledge.db"
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                    "--tenant", "demo", str(path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot read input file" in captured.err
    assert not db.parent.exists()


def test_utf8_ingestion_preserves_document_text(tmp_path, monkeypatch):
    path = tmp_path / "runbook.md"
    path.write_text("incident café recovery", encoding="utf-8")
    db = tmp_path / "knowledge.db"
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                    "--tenant", "demo", str(path)])
    main()
    assert Store(str(db)).search("demo", "incident")[0]["text"] == "incident café recovery"


def test_explicit_sources_keep_same_basename_documents_separate(tmp_path, monkeypatch, capsys):
    db = tmp_path / "knowledge.db"
    files = []
    for directory in ("team-one", "team-two"):
        folder = tmp_path / directory
        folder.mkdir()
        path = folder / "runbook.md"
        path.write_text(f"incident response for {directory}")
        files.append(path)
        monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                        "--tenant", "demo", "--source", directory, str(path)])
        main()
        assert capsys.readouterr().out.strip() == "1"
    store = Store(str(db))
    assert {hit["source"] for hit in store.search("demo", "incident")} == {
        "team-one", "team-two"}
    files[0].write_text("incident revised procedure")
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                    "--tenant", "demo", "--source", "team-one", str(files[0])])
    main()
    hits = {hit["source"]: hit["text"] for hit in store.search("demo", "incident")}
    assert hits == {"team-one": "incident revised procedure",
                    "team-two": "incident response for team-two"}


def test_default_source_remains_basename(tmp_path, monkeypatch):
    db = tmp_path / "knowledge.db"
    path = tmp_path / "runbook.md"
    path.write_text("incident response procedure")
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                    "--tenant", "demo", str(path)])
    main()
    assert Store(str(db)).search("demo", "incident")[0]["source"] == "runbook.md"


@pytest.mark.parametrize("source", ["", "   "])
def test_blank_source_rejected_before_creating_database(tmp_path, monkeypatch, capsys, source):
    db = tmp_path / "knowledge.db"
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "ingest",
                                    "--tenant", "demo", "--source", source, "missing.md"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--source cannot be blank" in capsys.readouterr().err
    assert not db.exists()


@pytest.mark.parametrize("tenant", ["", "   ", "\t\n"])
@pytest.mark.parametrize("command", ["ingest", "search"])
def test_blank_tenant_is_usage_error_before_io(tmp_path, monkeypatch, capsys, tenant, command):
    path = tmp_path / "knowledge.db"
    # The nonexistent file also checks that validation precedes reading input.
    argument = str(tmp_path / "missing.md") if command == "ingest" else "outage"
    monkeypatch.setattr("sys.argv", ["retrieval", "--db", str(path), command,
                                    "--tenant", tenant, argument])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--tenant cannot be blank" in captured.err
    assert not path.exists()


@pytest.mark.parametrize("command", ["ingest", "search"])
def test_invalid_embeddings_are_usage_errors_without_output(
    tmp_path, monkeypatch, capsys, command,
):
    db = tmp_path / "knowledge.db"
    store = Store(str(db), embed=lambda text: [1.0, 0.0])
    store.ingest("demo", "runbook.md", "incident recovery")
    before = store.search("demo", "incident")
    source = tmp_path / "runbook.md"
    source.write_text("replacement incident procedure", encoding="utf-8")
    # Ingest rejects an empty vector; search rejects a changed model dimension.
    monkeypatch.setattr("knowledge_retrieval.cli.openai_embed",
                        lambda text: [] if command == "ingest" else [1.0])
    flags = ["--answer"] if command == "search" else []
    argument = str(source) if command == "ingest" else "incident"
    monkeypatch.setattr("knowledge_retrieval.cli.grounded_answer",
                        lambda *args: pytest.fail("invalid retrieval must not request an answer"))
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "--embeddings",
                                    command, "--tenant", "demo", argument, *flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    reason = "at least one dimension" if command == "ingest" else "reindex documents"
    assert reason in captured.err
    assert "Traceback" not in captured.err
    assert store.search("demo", "incident") == before


@pytest.mark.parametrize("command", ["ingest", "search"])
@pytest.mark.parametrize("invalid", [[True], ["1"]])
def test_malformed_embeddings_are_usage_errors(tmp_path, monkeypatch, capsys, command, invalid):
    db = tmp_path / "knowledge.db"
    store = Store(str(db), embed=lambda text: [1.0])
    store.ingest("demo", "runbook.md", "original incident procedure")
    before = store.search("demo", "incident")
    document = tmp_path / "runbook.md"
    document.write_text("replacement incident procedure", encoding="utf-8")
    monkeypatch.setattr("knowledge_retrieval.cli.openai_embed", lambda text: invalid)
    monkeypatch.setattr("knowledge_retrieval.cli.grounded_answer",
                        lambda *args: pytest.fail("invalid vectors must not request an answer"))
    argument = str(document) if command == "ingest" else "incident"
    flags = [] if command == "ingest" else ["--answer"]
    monkeypatch.setattr("sys.argv", ["knowledge", "--db", str(db), "--embeddings",
                                    command, "--tenant", "demo", argument, *flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "embedding must be a list of numbers" in captured.err
    assert "Traceback" not in captured.err
    assert store.search("demo", "incident") == before
