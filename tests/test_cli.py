import pytest

from knowledge_retrieval.cli import main
from knowledge_retrieval.retrieval import Store


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
