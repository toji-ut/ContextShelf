from __future__ import annotations

from contextshelf.retrieval.sources.agent_docs import (
    _html_to_text,
    _normalize_seed,
    llms_txt_candidates,
    parse_llms_txt,
    registry_target_for,
)


SAMPLE_LLMS = """# Example Docs

> Short summary for agents.

## Docs

- [Getting started](https://example.com/docs/start.md): Install and run.
- [API reference](https://example.com/docs/api.md): Endpoints.

## Optional

- [Changelog](https://example.com/changelog.md): Old releases.
"""


class TestLlmsTxtCandidates:
    def test_seed_is_llms_txt(self):
        c = llms_txt_candidates("https://code.claude.com/docs/llms.txt")
        assert c == ["https://code.claude.com/docs/llms.txt"]

    def test_docs_path_includes_parent(self):
        c = llms_txt_candidates("https://code.claude.com/docs/en/overview")
        assert "https://code.claude.com/docs/llms.txt" in c
        assert "https://code.claude.com/llms.txt" in c


class TestParseLlmsTxt:
    def test_parses_links(self):
        entries = parse_llms_txt(SAMPLE_LLMS, base_url="https://example.com/llms.txt")
        titles = [t for t, _u, _d in entries]
        assert "Getting started" in titles
        assert "API reference" in titles
        assert "Changelog" in titles

    def test_skip_optional(self):
        entries = parse_llms_txt(
            SAMPLE_LLMS,
            base_url="https://example.com/llms.txt",
            skip_optional=True,
        )
        titles = [t for t, _u, _d in entries]
        assert "Getting started" in titles
        assert "Changelog" not in titles

    def test_resolves_relative_urls(self):
        entries = parse_llms_txt(
            "- [Home](/docs/home.md)\n",
            base_url="https://example.com/llms.txt",
        )
        assert entries[0][1] == "https://example.com/docs/home.md"


class TestAgentDocsHelpers:
    def test_normalize_seed_adds_scheme(self):
        assert _normalize_seed("code.claude.com/docs").startswith("https://")

    def test_registry_target_for_llms_path(self):
        t = registry_target_for("https://code.claude.com/docs/llms.txt")
        assert t == "https://code.claude.com/docs"

    def test_html_to_text_strips_tags(self):
        html = "<html><body><p>Hello <b>world</b></p><script>ignore()</script></body></html>"
        text = _html_to_text(html)
        assert "Hello" in text
        assert "world" in text
        assert "ignore" not in text
