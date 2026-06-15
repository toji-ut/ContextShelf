from __future__ import annotations

import os
import tempfile

from contextshelf.cli import _is_url, _source_prefixes


def test_is_url_detects_http():
    assert _is_url("https://code.claude.com/docs")
    assert _is_url("http://example.com")


def test_is_url_local_path():
    with tempfile.TemporaryDirectory() as d:
        assert not _is_url(d)


def test_is_url_domain_like():
    assert _is_url("code.claude.com/docs")


def test_source_prefixes_local():
    with tempfile.TemporaryDirectory() as d:
        p = _source_prefixes(d)
        assert os.path.abspath(d) in p


def test_source_prefixes_url():
    p = _source_prefixes("https://code.claude.com/docs")
    assert p == {"https://code.claude.com/docs"}
