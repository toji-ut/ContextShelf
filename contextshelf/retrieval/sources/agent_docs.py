from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

USER_AGENT = "ContextShelf/1.0.0 (agent-docs indexer)"
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
DEFAULT_MAX_PAGES = 200
FETCH_DELAY_SEC = 1.0


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = False
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()


def _fetch(url: str, *, timeout: float = 30.0) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            charset = "utf-8"
            ct = resp.headers.get("Content-Type", "")
            if "charset=" in ct.lower():
                charset = ct.lower().split("charset=", 1)[1].split(";")[0].strip()
            return raw.decode(charset, errors="replace")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def _normalize_seed(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urllib.parse.urlparse(url)
    if not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")
    path = parsed.path.rstrip("/")
    if path.endswith(".txt") or path.endswith(".md"):
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _origin(url: str) -> str:
    p = urllib.parse.urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def llms_txt_candidates(seed: str) -> list[str]:
    """URLs to try when looking for an llms.txt index."""
    seed = _normalize_seed(seed)
    parsed = urllib.parse.urlparse(seed)
    origin = _origin(seed)
    out: list[str] = []

    def add(u: str) -> None:
        u = u.rstrip("/")
        if u not in out:
            out.append(u)

    if parsed.path.rstrip("/").endswith("llms.txt"):
        add(seed)
        return out

    path = parsed.path.rstrip("/")
    if path:
        parts = [p for p in path.split("/") if p]
        for i in range(len(parts), 0, -1):
            add(f"{origin}/{'/'.join(parts[:i])}/llms.txt")
    add(f"{origin}/llms.txt")
    return out


def parse_llms_txt(
    body: str,
    *,
    base_url: str,
    skip_optional: bool = False,
) -> list[tuple[str, str, str]]:
    """
    Parse llms.txt into (title, absolute_url, description) entries.
    Skips the ## Optional section when skip_optional=True.
    """
    in_optional = False
    entries: list[tuple[str, str, str]] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## optional"):
            in_optional = True
            continue
        if stripped.startswith("## "):
            in_optional = False
            continue
        if skip_optional and in_optional:
            continue
        for title, href in LINK_RE.findall(line):
            abs_url = urllib.parse.urljoin(base_url, href.strip())
            desc = ""
            if "]:" in line:
                after = line.split("]:", 1)[-1].strip()
                if after:
                    desc = after
            entries.append((title.strip(), abs_url, desc))
    return entries


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        return parser.text()
    except Exception:
        return ""


def _page_to_text(url: str, body: str) -> str:
    lower = url.lower()
    if lower.endswith((".md", ".mdx", ".txt", ".rst")) or "text/plain" in body[:200].lower():
        return body.strip()
    if "<html" in body[:500].lower() or "<body" in body[:500].lower():
        text = _html_to_text(body)
        return text if text else body.strip()
    return body.strip()


def _discover_llms_txt(seed: str) -> tuple[str, str] | None:
    for candidate in llms_txt_candidates(seed):
        body = _fetch(candidate)
        if body and "#" in body and LINK_RE.search(body):
            return candidate, body
        time.sleep(0.2)
    return None


def load_agent_docs(
    seed_url: str,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    skip_optional: bool = False,
    progress_callback=None,
) -> tuple[list[tuple[str, str, str]], str]:
    """
    Index AI-agent documentation from a seed URL.

    1. Discover /llms.txt (or use seed if it points at one)
    2. Optionally ingest /llms-full.txt when present
    3. Fetch linked pages listed in the index

    Returns (documents, registry_target) where documents are
    (source_uri, path_suffix, content) tuples like local sources.
    """
    seed = _normalize_seed(seed_url)
    reg_target = seed
    docs: list[tuple[str, str, str]] = []
    seen_urls: set[str] = set()

    def add_doc(page_url: str, content: str, suffix: str = "") -> None:
        page_url = page_url.split("#")[0].rstrip("/")
        if not content.strip() or page_url in seen_urls:
            return
        seen_urls.add(page_url)
        docs.append((page_url, suffix, content))

    # llms-full.txt — single-file dump when available
    full_candidates = [
        urllib.parse.urljoin(seed + "/", "llms-full.txt"),
        f"{_origin(seed)}/llms-full.txt",
    ]
    for full_url in full_candidates:
        full_body = _fetch(full_url)
        if full_body and len(full_body) > 500 and "#" in full_body:
            add_doc(full_url, full_body, "llms-full.txt")
            reg_target = full_url.rsplit("/llms-full.txt", 1)[0] or _origin(seed)
            if progress_callback:
                progress_callback(1, 1, "llms-full.txt")
            return docs, reg_target

    discovered = _discover_llms_txt(seed)
    if not discovered:
        body = _fetch(seed)
        if body:
            add_doc(seed, _page_to_text(seed, body))
        return docs, reg_target

    index_url, index_body = discovered
    reg_target = index_url.rsplit("/llms.txt", 1)[0] or _origin(seed)
    add_doc(index_url, index_body, "llms.txt")

    entries = parse_llms_txt(index_body, base_url=index_url, skip_optional=skip_optional)
    if max_pages > 0:
        entries = entries[:max_pages]

    total = len(entries)
    for i, (title, page_url, desc) in enumerate(entries, 1):
        if progress_callback:
            progress_callback(i, total, title[:60])
        body = _fetch(page_url)
        if body:
            text = _page_to_text(page_url, body)
            if desc and not text.startswith("#"):
                text = f"# {title}\n\n> {desc}\n\n{text}"
            add_doc(page_url, text)
        if i < total:
            time.sleep(FETCH_DELAY_SEC)

    return docs, reg_target


def registry_target_for(seed_url: str) -> str:
    """Best-effort registry key without fetching linked pages."""
    seed = _normalize_seed(seed_url)
    for candidate in llms_txt_candidates(seed):
        if candidate.endswith("/llms.txt"):
            return candidate[: -len("/llms.txt")] or _origin(seed)
    return seed
