"""Ingest service tests — upload path (direct) + web path (fetch mocked)."""

import app.ingest.service as svc_mod
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.ingest.schemas import WebIngestFetchSuccess
from app.ingest.service import IngestService


class FakeUpload:
    """Minimal stand-in for FastAPI's UploadFile (only what ingest_upload uses)."""

    def __init__(self, filename: str, content_type: str, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data


async def test_ingest_upload_saves_document(session) -> None:  # type: ignore[no-untyped-def]
    svc = IngestService(session)
    data = b"Return policy is 30 days. " * 30
    uploads = [FakeUpload("notes.txt", "text/plain", data)]
    results = await svc.ingest_upload(uploads)  # type: ignore[arg-type]
    assert len(results) == 1
    assert results[0].name == "notes.txt"


async def test_ingest_upload_isolates_failures(session) -> None:  # type: ignore[no-untyped-def]
    svc = IngestService(session)
    # invalid PDF bytes → extract() raises → caught per-file → UploadIngestFailure
    bad = [FakeUpload("broken.pdf", "application/pdf", b"not a real pdf")]
    results = await svc.ingest_upload(bad)  # type: ignore[arg-type]
    assert len(results) == 1
    assert hasattr(results[0], "reason")  # it's an UploadIngestFailure


async def test_ingest_web_fetch_mocked(session, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    async def fake_fetch(url: object) -> WebIngestFetchSuccess:
        return WebIngestFetchSuccess(
            status="ok",
            url="http://example.com",
            content_type="text/plain",
            body="Shipping is free over 50 dollars. " * 20,
            byte_size=680,
        )

    monkeypatch.setattr(svc_mod, "fetch_or_fail", fake_fetch)
    svc = IngestService(session)
    results = await svc.ingest_web(["http://example.com"])  # type: ignore[list-item]
    assert results[0].status == "ok"


def test_configure_logging_smoke() -> None:
    configure_logging(settings)  # must not raise
