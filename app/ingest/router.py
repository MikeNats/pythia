from fastapi import APIRouter, UploadFile

from app.core.dependencies import DbSession
from app.ingest.schemas import UploadIngestResponse, WebIngestRequest, WebIngestResponse
from app.ingest.service import IngestService

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/web", response_model=WebIngestResponse)
async def ingest(request: WebIngestRequest, session: DbSession) -> WebIngestResponse:
    results = await IngestService(session).ingest_web(request.urls)
    return WebIngestResponse(results=results)


@router.post("/upload")
async def upload_files(
    files: list[UploadFile], session: DbSession
) -> UploadIngestResponse:
    results = await IngestService(session).ingest_upload(files)
    return UploadIngestResponse(results=results)
