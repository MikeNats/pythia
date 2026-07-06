import asyncio

from fastapi import UploadFile
from pydantic import HttpUrl
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingest.chunking import chunk_text
from app.ingest.extractors import extract
from app.ingest.fetcher import fetch_or_fail
from app.ingest.models import Chunk, Document, SourceType
from app.ingest.schemas import (
    UploadIngestFailure,
    UploadIngestSuccess,
    WebIngestFetchFailure,
    WebIngestFetchSuccess,
)
from app.retrieval.embedder import get_embedder


class IngestService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _embed_chunks(self, text: str) -> list[Chunk]:
        pieces = chunk_text(text)
        vectors = await asyncio.to_thread(get_embedder().embed, pieces)
        return [
            Chunk(position=i, text=p, embedding=v)
            for i, (p, v) in enumerate(zip(pieces, vectors, strict=True))
        ]

    async def ingest_web(
        self, urls: list[HttpUrl]
    ) -> list[WebIngestFetchSuccess | WebIngestFetchFailure]:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(fetch_or_fail(u)) for u in urls]
        results = [t.result() for t in tasks]

        docs: list[Document] = []
        for r in results:
            if r.status != "ok":
                continue
            text = extract(r.content_type, r.body.encode("utf-8"))
            docs.append(
                Document(
                    name=str(r.url),
                    source_type=SourceType.web,
                    source_ref=str(r.url),
                    content_type=r.content_type,
                    byte_size=r.byte_size,
                    chunks=await self._embed_chunks(text),
                )
            )

        self.session.add_all(docs)
        await self.session.commit()
        return results

    async def ingest_upload(
        self, files: list[UploadFile]
    ) -> list[UploadIngestSuccess | UploadIngestFailure]:
        docs: list[Document] = []
        results: list[UploadIngestSuccess | UploadIngestFailure] = []
        for f in files:
            name = f.filename or "upload"
            try:
                content_type = f.content_type or ""
                data = await f.read()
                byte_size = len(data)
                text = extract(content_type, data)
                docs.append(
                    Document(
                        name=name,
                        source_type=SourceType.upload,
                        source_ref=name,
                        content_type=content_type,
                        byte_size=byte_size,
                        chunks=await self._embed_chunks(text),
                    )
                )
                results.append(
                    UploadIngestSuccess(
                        name=name, content_type=content_type, byte_size=byte_size
                    )
                )
            except Exception as e:
                results.append(UploadIngestFailure(name=name, reason=str(e)))

        self.session.add_all(docs)
        await self.session.commit()
        return results
