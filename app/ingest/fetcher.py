import asyncio

import httpx
from pydantic import HttpUrl

from app.core.decorators import RetryError, retry
from app.ingest.net_guard import UnsafeURL, validate_url
from app.ingest.schemas import WebIngestFetchFailure, WebIngestFetchSuccess

async_client = httpx.AsyncClient()
sem = asyncio.Semaphore(10)


async def fetch_or_fail(url: HttpUrl) -> WebIngestFetchSuccess | WebIngestFetchFailure:
    try:
        await validate_url(str(url))
    except UnsafeURL as e:
        return WebIngestFetchFailure(status="error", url=url, reason=f"blocked: {e}")
    try:
        return await retrieve_document(url)
    except RetryError as e:
        return WebIngestFetchFailure(
            status="error",
            url=url,
            reason=f"network error after retries: {e.__cause__}",
        )


@retry(retry_on=(httpx.RequestError,))
async def retrieve_document(
    url: HttpUrl,
) -> WebIngestFetchSuccess | WebIngestFetchFailure:
    async with sem, async_client.stream("GET", str(url)) as response:
        if response.status_code != 200:
            return WebIngestFetchFailure(
                status="error",
                url=url,
                reason=f"HTTP {response.status_code}",
            )
        body = await response.aread()
        content_type = response.headers.get("content-type", "application/octet-stream")
        return WebIngestFetchSuccess(
            status="ok",
            url=url,
            content_type=content_type,
            body=body.decode("utf-8", errors="ignore"),
            byte_size=len(body),
        )
