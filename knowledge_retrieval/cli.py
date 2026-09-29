import argparse
import json
import os

from .retrieval import Store, grounded_answer, openai_embed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.getenv("PORTFOLIO_DB", "knowledge.db"))
    parser.add_argument("--embeddings", action="store_true", help="Use OpenAI vectors for ingestion and search")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--tenant", required=True)
    ingest.add_argument("file")
    search = sub.add_parser("search")
    search.add_argument("--tenant", required=True)
    search.add_argument("query")
    search.add_argument("--answer", action="store_true")
    args = parser.parse_args()
    store = Store(args.db, embed=openai_embed if args.embeddings else None)
    if args.command == "ingest":
        from pathlib import Path
        path = Path(args.file)
        print(store.ingest(args.tenant, path.name, path.read_text()))
    else:
        hits = store.search(args.tenant, args.query)
        print(json.dumps(hits, indent=2))
        if args.answer:
            print(grounded_answer(args.query, hits))


if __name__ == "__main__":
    main()
