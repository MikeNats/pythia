"""MCP tool handlers. Importing this module registers the tools."""

from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import UserSession
from app.ingest.service import IngestService
from app.mcp.registry import mcp_tool
from app.retrieval.services import search_query_to_chunks


class SearchDocsInput(BaseModel):
    query: str = Field(description="The question or text to search for")
    k: int = Field(default=5, description="Max number of chunks to return")


@mcp_tool(
    name="search_docs",
    description=(
        "Search the ingested documents for text relevant to the query. Returns "
        "the top matching chunks with their source document id and a relevance "
        "score from 0 to 1. Use these chunks to ground your answer and cite them."
    ),
    input_model=SearchDocsInput,
)
async def search_docs(
    session: AsyncSession, args: SearchDocsInput, user: UserSession
) -> list[dict[str, Any]]:
    hits = await search_query_to_chunks(session, args.query, args.k)
    return [
        {"text": h.text, "document_id": str(h.document_id), "score": round(h.score, 3)}
        for h in hits
    ]


class IngestMdInput(BaseModel):
    name: str = Field(description="A name/title for the document")
    content: str = Field(description="The markdown (or plain text) content to ingest")


@mcp_tool(
    name="ingest_md",
    description=(
        "Ingest a markdown or plain-text document into the knowledge base so it "
        "becomes searchable later. Provide the document's name and its full text. "
        "Returns the new document id and how many chunks were stored."
    ),
    input_model=IngestMdInput,
)
async def ingest_md(
    session: AsyncSession, args: IngestMdInput, user: UserSession
) -> dict[str, object]:
    doc = await IngestService(session).ingest_text(args.name, args.content)
    return {"document_id": str(doc.id), "name": doc.name, "chunks": len(doc.chunks)}
