import pytest

from knowledge_retrieval.cli import main
from knowledge_retrieval.retrieval import Store


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
