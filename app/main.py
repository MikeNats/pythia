import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.db import engine
from app.core.dependencies import get_current_user
from app.core.exceptions import NotFoundError
from app.core.health import router as health_router
from app.core.logging_config import configure_logging
from app.core.middlewares import AuditLogMiddleware, RequestIdMiddleware
from app.ingest.router import router as ingest_router
from app.llm.guardrails import GuardrailError
from app.llm.router import LLMModelNotFoundError, LLMNotFoundError
from app.mcp.router import router as mcp_router
from app.retrieval.router import router as retrieval_router

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings)
    logger.info("startup complete")
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
    logger.info("not found on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=404, content={"detail": "Not found"})


async def bad_llm_request_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.add_exception_handler(LLMNotFoundError, bad_llm_request_handler)
app.add_exception_handler(LLMModelNotFoundError, bad_llm_request_handler)


@app.exception_handler(GuardrailError)
async def guardrail_handler(request: Request, exc: GuardrailError) -> JSONResponse:
    logger.warning("guardrail blocked on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": "Request blocked"})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


app.add_middleware(RequestIdMiddleware)
app.add_middleware(AuditLogMiddleware)


protected = APIRouter(dependencies=[Depends(get_current_user)])

protected.include_router(ingest_router)
protected.include_router(retrieval_router)
protected.include_router(mcp_router)

app.include_router(health_router)
app.include_router(protected)
