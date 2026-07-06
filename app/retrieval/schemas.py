from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.llm.router import LLMProviders
from app.retrieval.models import MessageRole


class SearchHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class QuestionRequest(BaseModel):
    question: str
    provider: LLMProviders
    model: str
    k: int = 5
    conversation_id: UUID
    stream: bool = False


class Citation(BaseModel):
    chunk_id: UUID
    document_id: UUID
    text: str


class QuestionResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list[Citation])


class AnswerWithCitations(BaseModel):
    answer: str = Field(description="Answer using only the provided context.")
    cited_indices: list[int] = Field(
        default_factory=list[int],
        description="The [n] block numbers from the context that support the answer.",
    )


class DeleteConversationRequest(BaseModel):
    conversation_id: UUID


class DeleteConversationResponse(BaseModel):
    message: str = Field(
        description="Confirmation message for the deleted conversation."
    )
    conversation_id: UUID = Field(description="The ID of the deleted conversation.")


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    role: MessageRole
    text: str
    created_at: datetime


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    title: str
    created_at: datetime


class GetMessagesResponse(BaseModel):
    messages: list[MessageResponse]


class GetConversationsResponse(BaseModel):
    conversations: list[ConversationResponse]


class GetConversationResponse(BaseModel):
    conversation: ConversationResponse
