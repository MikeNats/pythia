import contextvars
import uuid

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
tenant_id_var: contextvars.ContextVar[uuid.UUID | None] = contextvars.ContextVar(
    "tenant_id", default=None
)
user_id_var: contextvars.ContextVar[uuid.UUID | None] = contextvars.ContextVar(
    "user_id", default=None
)


def get_request_id() -> str:
    return request_id_var.get()


def get_tenant_id() -> uuid.UUID | None:
    return tenant_id_var.get()


def get_user_id() -> uuid.UUID | None:
    return user_id_var.get()
