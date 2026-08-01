"""用户可见 session 能力的边界模块。

`bauhinia_agent.context` 仍然负责底层事件日志、上下文投影、checkpoint 和压缩；
本包后续只承载 catalog、resume 编排、只读 transcript 和 share export 等用户入口。
"""

from bauhinia_agent.session.errors import (
    SessionCorruptError,
    SessionEmptyError,
    SessionError,
    SessionInvalidIdError,
    SessionNotFoundError,
)
from bauhinia_agent.session.models import (
    RedactionOptions,
    SessionRecord,
    ShareOptions,
    Transcript,
    TranscriptEntry,
)
from bauhinia_agent.session.share import SessionShareService
from bauhinia_agent.session.transcript import TranscriptBuilder

__all__ = [
    "RedactionOptions",
    "SessionCorruptError",
    "SessionEmptyError",
    "SessionError",
    "SessionInvalidIdError",
    "SessionNotFoundError",
    "SessionRecord",
    "SessionShareService",
    "ShareOptions",
    "Transcript",
    "TranscriptBuilder",
    "TranscriptEntry",
]
