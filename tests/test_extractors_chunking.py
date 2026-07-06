"""Unit tests for pure helpers: byte->text extraction and text chunking.

Both modules are plain functions -- no DB, no LLM, no network. These tests pin
their documented behavior (the fallback decoder and the slicing chunker).
"""

from app.ingest.chunking import chunk_text
from app.ingest.extractors import extract


def test_extract_plain_text_returns_decoded_text() -> None:
    assert extract("text/plain", b"hello world") == "hello world"


def test_extract_ignores_charset_param() -> None:
    # content_type is split on ';' before lookup, so params don't matter.
    assert extract("text/plain; charset=utf-8", b"hello") == "hello"


def test_extract_unknown_content_type_falls_back_to_text() -> None:
    # Unknown mime -> default _text decoder (utf-8, errors ignored). No raise.
    assert extract("application/octet-stream", b"plain bytes") == "plain bytes"


def test_extract_ignores_invalid_utf8_bytes() -> None:
    # _text uses errors="ignore": the bad 0xff byte is dropped, rest survives.
    assert extract("text/plain", b"ab\xffcd") == "abcd"


def test_extract_html_strips_tags_and_script() -> None:
    html = (
        b"<html><head><title>t</title></head>"
        b"<body><script>x=1</script><p>Hi there</p></body></html>"
    )
    result = extract("text/html", html)
    assert "Hi there" in result
    assert "x=1" not in result


def test_chunk_text_long_string_returns_nonempty_str_pieces() -> None:
    text = "a" * 2500
    chunks = chunk_text(text, size=1000)
    assert len(chunks) == 3
    assert all(isinstance(c, str) and c for c in chunks)
    assert "".join(chunks) == text


def test_chunk_text_empty_string_returns_empty_list() -> None:
    # range(0, 0, size) is empty -> no chunks.
    assert chunk_text("") == []


def test_chunk_text_short_string_returns_single_chunk() -> None:
    assert chunk_text("short", size=1000) == ["short"]
