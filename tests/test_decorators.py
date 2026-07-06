"""Tests for the async @retry decorator in app/core/decorators.py.

- retry: a func that fails N-1 times then succeeds returns its value.
- exhaustion: a func that always raises a retry_on error raises RetryError.
- passthrough: an error NOT in retry_on propagates immediately (no retry).

`backoff=0` keeps the sleeps instant so the suite stays fast.
"""

import pytest

from app.core.decorators import RetryError, retry


async def test_retries_then_succeeds() -> None:
    calls = 0

    @retry(max_attempts=3, backoff=0)
    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("not yet")
        return "ok"

    assert await flaky() == "ok"
    assert calls == 3  # failed twice, succeeded on the third


async def test_raises_retry_error_after_max_attempts() -> None:
    calls = 0

    @retry(max_attempts=3, backoff=0, retry_on=(ValueError,))
    async def always_fails() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("nope")

    with pytest.raises(RetryError):
        await always_fails()
    assert calls == 3  # exactly max_attempts tries, no more


async def test_retry_error_chains_last_exception() -> None:
    @retry(max_attempts=2, backoff=0, retry_on=(ValueError,))
    async def always_fails() -> str:
        raise ValueError("boom")

    with pytest.raises(RetryError) as exc_info:
        await always_fails()
    assert isinstance(exc_info.value.__cause__, ValueError)


async def test_non_matching_exception_propagates_immediately() -> None:
    calls = 0

    @retry(max_attempts=3, backoff=0, retry_on=(ValueError,))
    async def wrong_error() -> str:
        nonlocal calls
        calls += 1
        raise KeyError("different")

    with pytest.raises(KeyError):
        await wrong_error()
    assert calls == 1  # not retried
