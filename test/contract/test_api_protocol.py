from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.api import create_app  # noqa: E402
from xinyidai_agent.protocol import (  # noqa: E402
    AgentEvent,
    ChatResponse,
    DiagnosticEvent,
    RetrievalTrace,
    SourceDocument,
)


class FakeLoop:
    def answer(self, request):
        return ChatResponse(
            answer=f"收到：{request.user_message}",
            sources=[
                SourceDocument(
                    source_id="source-1",
                    title="测试来源",
                    content="测试证据",
                    source_type="policy",
                )
            ],
            retrieval_trace=RetrievalTrace(
                query=request.user_message,
                top_k=request.top_k,
                results_count=1,
            ),
            diagnostics=[DiagnosticEvent(name="fake_loop")],
        )

    def run(self, request):
        yield AgentEvent(
            turn_id="turn-1",
            sequence=1,
            event_type="turn_started",
            visibility="diagnostic",
            timestamp="2026-04-28T00:00:00+00:00",
            payload={"user_message": request.user_message},
        )
        yield AgentEvent(
            turn_id="turn-1",
            sequence=2,
            event_type="final_answer",
            visibility="user",
            timestamp="2026-04-28T00:00:01+00:00",
            payload={"answer": f"收到：{request.user_message}"},
        )


class ApiProtocolTest(unittest.TestCase):
    def test_chat_endpoint_returns_protocol_response(self) -> None:
        client = TestClient(create_app(loop=FakeLoop()))
        response = client.post("/chat", json={"user_message": "测试问题", "top_k": 2})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "收到：测试问题")
        self.assertEqual(payload["sources"][0]["source_type"], "policy")
        self.assertEqual(payload["retrieval_trace"]["top_k"], 2)
        self.assertEqual(payload["diagnostics"][0]["name"], "fake_loop")

    def test_chat_stream_returns_sse_events(self) -> None:
        client = TestClient(create_app(loop=FakeLoop()))
        response = client.post("/chat/stream", json={"user_message": "测试问题"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: turn_started", response.text)
        self.assertIn("event: final_answer", response.text)


if __name__ == "__main__":
    unittest.main()
