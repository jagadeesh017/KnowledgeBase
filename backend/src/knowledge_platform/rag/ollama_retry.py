import time
from collections.abc import Callable

import httpx
import ollama

from knowledge_platform.exceptions import ServiceUnavailableError

NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 409, 422}
RETRYABLE_ERRORS = (ollama.RequestError, httpx.TimeoutException, ConnectionError)


def call_with_retries[T](
    operation: Callable[[], T],
    *,
    attempts: int,
    backoff_seconds: float,
) -> T:
    for attempt in range(attempts):
        try:
            return operation()
        except ollama.ResponseError as error:
            if (
                error.status_code in NON_RETRYABLE_STATUS_CODES
                or attempt == attempts - 1
            ):
                raise ServiceUnavailableError(
                    f"Ollama rejected the request: {error}"
                ) from error
        except RETRYABLE_ERRORS as error:
            if attempt == attempts - 1:
                raise ServiceUnavailableError(
                    f"Ollama is unavailable: {error}"
                ) from error
        time.sleep(backoff_seconds * (2**attempt))

    raise ServiceUnavailableError("Ollama is unavailable")
