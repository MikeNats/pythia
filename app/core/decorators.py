import asyncio
import functools
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

AsyncFunc = TypeVar("AsyncFunc", bound=Callable[..., Coroutine[Any, Any, Any]])


class RetryError(Exception):
    pass


def retry(
    max_attempts: int = 3,
    backoff: float = 0.1,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[AsyncFunc], AsyncFunc]:
    def decorator(fn: AsyncFunc) -> AsyncFunc:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except retry_on as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(backoff * (2**attempt))
            raise RetryError(f"Gave up after {max_attempts} attempts") from last_exc

        return wrapper  # type: ignore[return-value]

    return decorator
