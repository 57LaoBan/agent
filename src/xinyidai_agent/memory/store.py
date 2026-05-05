from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from xinyidai_agent.protocol import AgentEvent


class TranscriptStore(Protocol):
    def append_event(self, session_id: str, turn_id: str, sequence: int, event: AgentEvent) -> None:
        ...

    def append_turn_summary(self, session_id: str, turn_id: str, summary: dict[str, Any]) -> None:
        ...

    def list_events(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        ...


class JsonlTranscriptStore:
    """JSONL transcript store for audit and replay."""

    def __init__(self, base_dir: Path | str = Path(".data") / "transcripts") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def append_event(self, session_id: str, turn_id: str, sequence: int, event: AgentEvent) -> None:
        self._append(
            session_id,
            {
                "record_type": "event",
                "session_id": session_id,
                "turn_id": turn_id,
                "sequence": sequence,
                "event_type": event.event_type,
                "visibility": event.visibility,
                "payload": event.payload,
                "created_at": event.timestamp,
            },
        )

    def append_turn_summary(self, session_id: str, turn_id: str, summary: dict[str, Any]) -> None:
        self._append(
            session_id,
            {
                "record_type": "turn_summary",
                "session_id": session_id,
                "turn_id": turn_id,
                "summary": summary,
            },
        )

    def list_events(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows[-limit:] if limit is not None else rows

    def _append(self, session_id: str, row: dict[str, Any]) -> None:
        path = self._path(session_id)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _path(self, session_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)
        return self._base_dir / f"{safe}.jsonl"
