from uuid import UUID

from app.core.exceptions import NotFoundError


class ConversationNotFound(NotFoundError):
    def __init__(self, conversation_id: UUID) -> None:
        super().__init__(f"conversation {conversation_id} not found")
