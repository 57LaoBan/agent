from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import psycopg
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.config import SessionStorageConfig  # noqa: E402
from xinyidai_agent.api import create_app  # noqa: E402
from xinyidai_agent.memory import (  # noqa: E402
    MemoryManager,
    PostgresConversationStore,
    PostgresSessionStore,
)
from xinyidai_agent.protocol import ChatRequest, RouteDecision, SessionStateSnapshot  # noqa: E402
from xinyidai_agent.runtime import ControlledAgentLoop  # noqa: E402


class FakeModel:
    """用于会话持久化测试的确定性模型替身。"""

    def complete(self, messages: list[dict[str, str]], **_: object) -> str:
        """根据系统提示返回系统工具空操作或固定回答。"""
        system = messages[0]["content"] if messages else ""
        if "update_session_state" in system:
            return '{"tool_name":"update_session_state","arguments":{"operations":[]}}'
        return "这是持久化测试回答"


class SmalltalkRouter:
    """固定返回无需工具调用的路由结果。"""

    def route(self, request: ChatRequest) -> RouteDecision:
        """返回小聊场景，触发直接回答分支。"""
        return RouteDecision(
            scene="SMALLTALK",
            intent="SMALLTALK",
            confidence=0.9,
            route_reason="持久化测试直接回答",
            route_source="test_router",
            should_call_tool=False,
        )


class StoreOnlyLoop:
    """只用于验证 API 会话恢复接口的 Agent 替身。"""

    def __init__(self, conversation_store: PostgresConversationStore) -> None:
        """保存 API 读取会话所需的会话记录存储。"""
        self.conversation_store = conversation_store

    def answer(self, request: ChatRequest):
        """测试中不会调用该方法。"""
        raise AssertionError("chat endpoint is not used in this test")

    def run(self, request: ChatRequest):
        """测试中不会调用该方法。"""
        raise AssertionError("stream endpoint is not used in this test")


class PostgresSessionPersistenceTest(unittest.TestCase):
    """验证 PostgreSQL 会话状态、聊天记录和审计记录闭环。"""

    @classmethod
    def setUpClass(cls) -> None:
        """连接本地 PostgreSQL，并在不可用时跳过集成测试。"""
        cls.dsn = SessionStorageConfig.from_env().dsn
        try:
            with psycopg.connect(cls.dsn, connect_timeout=3):
                pass
        except psycopg.Error as exc:
            raise unittest.SkipTest(f"PostgreSQL is not available: {exc}") from exc

    def setUp(self) -> None:
        """初始化表结构并清理本测试创建的会话。"""
        self.conversation_store = PostgresConversationStore(self.dsn)
        self.session_store = PostgresSessionStore(self.dsn, ensure_schema=False)
        self._delete_test_sessions()

    def tearDown(self) -> None:
        """清理本测试创建的会话数据。"""
        self._delete_test_sessions()

    def test_state_and_transcript_roundtrip(self) -> None:
        """状态快照和聊天记录可以分别落库，并一起恢复。"""
        session_id = f"test-session-{uuid4()}"
        state = SessionStateSnapshot(
            session_id=session_id,
            confirmed_slots={"company_name": "测试企业"},
            updated_at=datetime.now(UTC).isoformat(),
        )

        self.session_store.save(state)
        self.conversation_store.append_message(
            session_id=session_id,
            role="user",
            content="查询测试企业额度",
        )
        self.conversation_store.append_message(
            session_id=session_id,
            role="assistant",
            content="测试企业当前可用额度为 50 万元",
        )

        restored_state = self.session_store.get(session_id)
        transcript = self.conversation_store.get_transcript(session_id)

        self.assertIsNotNone(restored_state)
        self.assertEqual(restored_state.confirmed_slots["company_name"], "测试企业")
        self.assertIsNotNone(transcript)
        self.assertEqual([message.role for message in transcript.messages], ["user", "assistant"])
        self.assertEqual(transcript.session_state.confirmed_slots["company_name"], "测试企业")

    def test_agent_turn_persists_messages_state_and_audit_events(self) -> None:
        """Agent 正常回答时会同步保存用户消息、助手消息、状态和审计事件。"""
        session_id = f"test-session-{uuid4()}"
        loop = ControlledAgentLoop(
            model=FakeModel(),
            router=SmalltalkRouter(),
            memory_manager=MemoryManager(self.session_store),
            conversation_store=self.conversation_store,
        )

        response = loop.answer(ChatRequest(user_message="你好", session_id=session_id))
        transcript = self.conversation_store.get_transcript(session_id)

        self.assertEqual(response.answer, "这是持久化测试回答")
        self.assertIsNotNone(transcript)
        self.assertEqual([message.role for message in transcript.messages], ["user", "assistant"])
        self.assertEqual(transcript.messages[0].content, "你好")
        self.assertEqual(transcript.messages[1].content, "这是持久化测试回答")
        self.assertIsNotNone(transcript.session_state)
        self.assertGreater(self._count_audit_events(session_id), 0)

    def test_api_restores_session_transcript_from_postgres(self) -> None:
        """API 可以从 PostgreSQL 恢复网页端需要的会话记录。"""
        session_id = f"test-session-{uuid4()}"
        state = SessionStateSnapshot(
            session_id=session_id,
            confirmed_slots={"company_name": "接口测试企业"},
            updated_at=datetime.now(UTC).isoformat(),
        )
        self.session_store.save(state)
        self.conversation_store.append_message(
            session_id=session_id,
            role="user",
            content="恢复这个会话",
        )
        self.conversation_store.append_message(
            session_id=session_id,
            role="assistant",
            content="已恢复",
        )

        client = TestClient(create_app(loop=StoreOnlyLoop(self.conversation_store)))
        detail_response = client.get(f"/sessions/{session_id}")
        list_response = client.get("/sessions", params={"limit": 10})

        self.assertEqual(detail_response.status_code, 200)
        detail = detail_response.json()
        self.assertEqual(detail["session"]["session_id"], session_id)
        self.assertEqual([message["role"] for message in detail["messages"]], ["user", "assistant"])
        self.assertEqual(detail["session_state"]["confirmed_slots"]["company_name"], "接口测试企业")
        self.assertEqual(list_response.status_code, 200)
        self.assertTrue(any(item["session_id"] == session_id for item in list_response.json()))

    def _delete_test_sessions(self) -> None:
        """删除测试前缀会话，依赖外键级联清理消息和状态。"""
        with psycopg.connect(self.dsn) as conn:
            conn.execute("DELETE FROM chat_sessions WHERE id LIKE 'test-session-%'")

    def _count_audit_events(self, session_id: str) -> int:
        """统计指定会话的审计事件数量。"""
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                "SELECT count(*) FROM session_audit_events WHERE session_id = %s",
                (session_id,),
            ).fetchone()
        return int(row[0])


if __name__ == "__main__":
    unittest.main()
