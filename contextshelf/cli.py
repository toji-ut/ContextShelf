from __future__ import annotations

import argparse
import os
import re
import sys
import time

from contextshelf import __version__
from contextshelf.retrieval.store import DEFAULT_INDEX_PATH

_URL_LIKE = re.compile(r"^https?://", re.I)
_DOMAIN_LIKE = re.compile(r"^[\w.-]+\.[a-z]{2,}(/|$)", re.I)


def _is_url(target: str) -> bool:
    t = target.strip()
    if _URL_LIKE.match(t):
        return True
    if os.path.isdir(t) or os.path.isfile(t):
        return False
    return bool(_DOMAIN_LIKE.match(t))


def _gather_docs(
    target: str,
    extensions: str | None,
    *,
    max_pages: int,
    skip_optional: bool,
    fetch_progress=None,
):
    from contextshelf.retrieval.sources.agent_docs import load_agent_docs
    from contextshelf.retrieval.sources.local import load_local_folder

    t = target.strip()
    if _is_url(t):
        return load_agent_docs(
            t,
            max_pages=max_pages,
            skip_optional=skip_optional,
            progress_callback=fetch_progress,
        )
    path = os.path.abspath(t)
    if not os.path.isdir(path):
        raise SystemExit(f"Not a directory: {path}")
    return load_local_folder(path, extensions), path


def _progress_bar(done: int, total: int) -> None:
    pct = int(100 * done / total) if total else 100
    filled = min(20, int(pct / 5))
    bar = "#" * filled + "." * (20 - filled)
    print(f"\r  Embedding [{bar}] {pct}% ({done}/{total})", end="", flush=True)


def _fetch_progress(done: int, total: int, label: str) -> None:
    print(f"\r  Fetching [{done}/{total}] {label[:52]:<52}", end="", flush=True)


def cmd_index(args: argparse.Namespace) -> None:
    from contextshelf.retrieval.indexer import index_docs_into_store
    from contextshelf.retrieval.registry import add_source
    from contextshelf.retrieval.store import VectorStore

    target = args.target.strip()
    is_url = _is_url(target)
    t0 = time.time()
    print(f"Indexing {target}…")

    docs, reg_target = _gather_docs(
        target,
        args.extensions,
        max_pages=args.max_pages,
        skip_optional=args.skip_optional,
        fetch_progress=_fetch_progress if is_url else None,
    )
    if is_url:
        print()

    n_files = len({d[0] for d in docs}) if docs else 0
    print(f"  Found {n_files} pages → chunking…")

    store = VectorStore(args.index_path)
    store.load()

    chunks = index_docs_into_store(docs, batch_size=32, progress_callback=_progress_bar)
    print()
    store.add(chunks)
    store.save()

    kind = "agent_docs" if is_url else "local"
    add_source(args.index_path, kind, reg_target)
    elapsed = int(time.time() - t0)
    print(f"Done. {len(chunks):,} chunks indexed in {elapsed}s (index now {store.size():,} total).")


def cmd_search(args: argparse.Namespace) -> None:
    from contextshelf.retrieval.retriever import retrieve
    from contextshelf.retrieval.store import VectorStore

    store = VectorStore(args.index_path)
    store.load()
    chunks = retrieve(
        store,
        args.query,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    print(f"{len(chunks)} matches (threshold={args.threshold}):")
    for i, ch in enumerate(chunks, 1):
        meta = ch.metadata.get("heading") or ch.metadata.get("symbol") or ""
        label = f"{ch.source}" + (f" ({meta})" if meta else "")
        preview = ch.content.replace("\n", " ")[:280]
        print(f"\n{i}. {label}\n   {preview}")


def cmd_sources_list(args: argparse.Namespace) -> None:
    from contextshelf.retrieval.registry import load_registry

    reg = load_registry(args.index_path)
    if not reg:
        print("No indexed sources.")
        return
    for e in reg:
        kind = e.get("kind", "?")
        print(f"{kind}\t{e.get('target', '')}")


def _source_prefixes(target: str) -> set[str]:
    t = target.strip().rstrip("/")
    prefixes = {t}
    if not t.startswith(("http://", "https://")):
        prefixes.add(os.path.abspath(t))
    return prefixes


def cmd_sources_remove(args: argparse.Namespace) -> None:
    from contextshelf.retrieval.registry import remove_registry_target
    from contextshelf.retrieval.store import VectorStore

    prefixes = _source_prefixes(args.target)

    store = VectorStore(args.index_path)
    store.load()
    removed = 0
    for prefix in prefixes:
        removed += store.remove_source_prefix(prefix)
    store.save()
    for prefix in prefixes:
        remove_registry_target(args.index_path, prefix)
    print(f"Removed {removed} chunks from index.")


def cmd_sources_clear(args: argparse.Namespace) -> None:
    from contextshelf.retrieval.registry import load_registry, save_registry
    from contextshelf.retrieval.store import VectorStore

    store = VectorStore(args.index_path)
    store.load()
    n_chunks = store.size()
    n_sources = len(load_registry(args.index_path))

    if not args.yes:
        answer = input(f"Clear {n_chunks} chunks and {n_sources} sources? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    import numpy as np

    store.embeddings = np.zeros((0, 384), dtype=np.float32)
    store.chunks = []
    store.save()
    save_registry(args.index_path, [])
    print(f"Cleared {n_chunks} chunks and {n_sources} sources.")


def cmd_sources_sync(args: argparse.Namespace) -> None:
    from contextshelf.retrieval.registry import load_registry
    from contextshelf.retrieval.store import VectorStore

    reg = load_registry(args.index_path)
    if not reg:
        print("No sources to sync.")
        return

    store = VectorStore(args.index_path)
    store.load()
    for e in reg:
        for prefix in _source_prefixes(e.get("target", "")):
            store.remove_source_prefix(prefix)
    store.save()

    print("Re-indexing all registered sources…")
    for e in reg:
        cmd_index(
            argparse.Namespace(
                target=e["target"],
                extensions=None,
                index_path=args.index_path,
                max_pages=200,
                skip_optional=False,
            )
        )


def cmd_status(args: argparse.Namespace) -> None:
    from contextshelf.retrieval.registry import load_registry
    from contextshelf.retrieval.store import VectorStore

    store = VectorStore(args.index_path)
    store.load()
    reg = load_registry(args.index_path)
    kinds: dict[str, int] = {}
    for e in reg:
        k = e.get("kind", "unknown")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"ContextShelf v{__version__}")
    print("──────────────────────────────────────────")
    print(f"Index path   : {args.index_path}")
    print(f"Sources      : {len(reg)}")
    if kinds:
        print(f"  local      : {kinds.get('local', 0)}")
        print(f"  agent_docs : {kinds.get('agent_docs', 0)}")
    print(f"Chunks       : {store.size():,}")
    print("──────────────────────────────────────────")


def main() -> None:
    PRIMARY = {"index", "sources", "status", "search"}
    HELP_VER = {"-h", "--help", "--version"}

    argv = list(sys.argv)
    if len(argv) <= 1:
        argv = [argv[0], "status"]
    elif argv[1] in HELP_VER:
        pass
    elif argv[1] not in PRIMARY:
        if argv[1].startswith("-"):
            argv = [argv[0], "status"] + argv[1:]
        else:
            argv = [argv[0], "index"] + argv[1:]
    sys.argv = argv

    parser = argparse.ArgumentParser(
        prog="contextshelf",
        description="ContextShelf — index projects and agent docs into a searchable store",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--index-path",
        default=DEFAULT_INDEX_PATH,
        help=f"Where to store the index (default: {DEFAULT_INDEX_PATH})",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show index summary (default)")

    p_index = sub.add_parser("index", help="Index a local folder or agent docs URL")
    p_index.add_argument("target", help="Project path or docs URL (e.g. code.claude.com/docs)")
    p_index.add_argument("--extensions", help="Local only: e.g. .py,.md,.ts")
    p_index.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help="Agent docs: max pages to fetch from llms.txt (default: 200)",
    )
    p_index.add_argument(
        "--skip-optional",
        action="store_true",
        help="Agent docs: skip the ## Optional section in llms.txt",
    )

    p_search = sub.add_parser("search", help="Search the index for relevant context")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--threshold", type=float, default=0.35)

    p_sources = sub.add_parser("sources", help="Manage indexed sources")
    src_sub = p_sources.add_subparsers(dest="sources_cmd", required=True)
    src_sub.add_parser("list", help="List indexed sources")
    p_rem = src_sub.add_parser("remove", help="Remove a source")
    p_rem.add_argument("target", help="Project path or docs URL")
    src_sub.add_parser("sync", help="Re-index all sources")
    p_clear = src_sub.add_parser("clear", help="Remove all chunks and sources")
    p_clear.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
        return

    if args.command == "index":
        cmd_index(args)
        return

    if args.command == "search":
        cmd_search(args)
        return

    if args.command == "sources":
        if args.sources_cmd == "list":
            cmd_sources_list(args)
        elif args.sources_cmd == "remove":
            cmd_sources_remove(args)
        elif args.sources_cmd == "sync":
            cmd_sources_sync(args)
        elif args.sources_cmd == "clear":
            cmd_sources_clear(args)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
