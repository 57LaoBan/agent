from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.memory.store import JsonlTranscriptStore  # noqa: E402
from xinyidai_agent.protocol import AgentEvent  # noqa: E402


class TranscriptStoreTest(unittest.TestCase):
    def test_jsonl_store_appends_and_lists_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonlTranscriptStore(Path(tmp))
            event = AgentEvent(
                turn_id="turn-1",
                sequence=1,
                event_type="turn_started",
                visibility="diagnostic",
                timestamp="2026-05-04T00:00:00+00:00",
                payload={"user_message": "测试"},
            )

            store.append_event("session-1", event.turn_id, event.sequence, event)
            events = store.list_events("session-1")

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["session_id"], "session-1")
            self.assertEqual(events[0]["event_type"], "turn_started")

    def test_turn_summary_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonlTranscriptStore(Path(tmp))
            store.append_turn_summary("session-1", "turn-1", {"stop_reason": "completed"})

            events = store.list_events("session-1")

            self.assertEqual(events[0]["record_type"], "turn_summary")
            self.assertEqual(events[0]["summary"]["stop_reason"], "completed")


if __name__ == "__main__":
    unittest.main()
