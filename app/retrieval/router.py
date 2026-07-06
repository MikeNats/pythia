import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.dependencies import CurrentUser, DbSession
from app.retrieval.schemas import (
    ConversationResponse,
    DeleteConversationResponse,
    GetConversationResponse,
    GetConversationsResponse,
    GetMessagesResponse,
    MessageResponse,
    QuestionRequest,
    QuestionResponse,
    SearchResponse,
)
from app.retrieval.services import (
    chat,
    delete_conversation,
    get_conversation_messages,
    get_user_conversations,
    question,
    search_query_to_chunks,
    set_conversation_title,
    stream_chat,
)


async def _sse(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    async for tok in tokens:
        yield f"data: {json.dumps({'token': tok})}\n\n"
    yield "data: [DONE]\n\n"


router = APIRouter(prefix="/search", tags=["search"])


@router.get("/chunks", response_model=SearchResponse)
async def search_chunks_endpoint(
    q: str, session: DbSession, k: int = 5
) -> SearchResponse:
    hits = await search_query_to_chunks(session, q, k)
    return SearchResponse(query=q, hits=hits)


@router.post("/question", response_model=QuestionResponse)
async def search_question_endpoint(
    req: QuestionRequest,
    session: DbSession,
) -> QuestionResponse:
    return await question(req, session)


@router.post("/chat", response_model=QuestionResponse)
async def chat_endpoint(
    req: QuestionRequest,
    session: DbSession,
    user: CurrentUser,
) -> QuestionResponse | StreamingResponse:
    if req.stream:
        return StreamingResponse(
            _sse(stream_chat(req, session)), media_type="text/event-stream"
        )
    return await chat(req, session, user)


@router.delete("/conversation/{conversation_id}")
async def delete_conversation_endpoint(
    conversation_id: UUID, session: DbSession, user: CurrentUser
) -> DeleteConversationResponse:
    await delete_conversation(conversation_id, session, user)
    return DeleteConversationResponse(
        message="Conversation deleted successfully.",
        conversation_id=conversation_id,
    )


@router.get("/conversations", response_model=GetConversationsResponse)
async def get_conversations_endpoint(
    session: DbSession, user: CurrentUser
) -> GetConversationsResponse:
    conversations = await get_user_conversations(session, user)
    return GetConversationsResponse(
        conversations=[ConversationResponse.model_validate(c) for c in conversations]
    )


@router.put(
    "/conversation/{conversation_id}/title", response_model=GetConversationResponse
)
async def set_conversation_title_endpoint(
    conversation_id: UUID,
    title: str,
    session: DbSession,
    user: CurrentUser,
) -> GetConversationResponse:
    conversation = await set_conversation_title(session, conversation_id, title, user)
    return GetConversationResponse(
        conversation=ConversationResponse.model_validate(conversation)
    )


@router.get(
    "/conversation/{conversation_id}/messages", response_model=GetMessagesResponse
)
async def get_conversation_endpoint(
    conversation_id: UUID, session: DbSession, user: CurrentUser
) -> GetMessagesResponse:
    messages = await get_conversation_messages(session, conversation_id, user)
    return GetMessagesResponse(
        messages=[MessageResponse.model_validate(m) for m in messages]
    )
