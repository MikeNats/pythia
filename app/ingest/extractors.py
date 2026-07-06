"""Turn raw bytes into text, by content type. Plain functions, no class."""

import io

from bs4 import BeautifulSoup
from pypdf import PdfReader


def _text(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def _pdf(data: bytes) -> str:
    return "\n".join(
        page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages
    )


def _html(data: bytes) -> str:
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "head"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


_EXTRACTORS = {
    "application/pdf": _pdf,
    "text/html": _html,
}


def extract(content_type: str, data: bytes) -> str:
    mime = content_type.split(";")[0].strip().lower()
    return _EXTRACTORS.get(mime, _text)(data)
