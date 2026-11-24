import logging
from contextvars import ContextVar
from uuid import uuid4

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s [request_id=%(request_id)s] %(message)s",
        force=True,
    )
    for handler in logging.root.handlers:
        handler.addFilter(_RequestIdFilter())


def new_request_id() -> str:
    return uuid4().hex


def current_request_id() -> str:
    return request_id_var.get()


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True
