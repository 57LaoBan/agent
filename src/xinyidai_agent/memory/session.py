from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from xinyidai_agent.protocol import (
    ChatRequest,
    PendingAction,
    RouteDecision,
    SessionStateSnapshot,
    SessionToolResult,
    ToolResult,
)


class SessionStore(Protocol):
    """会话状态存储接口协议。

    定义会话状态持久化的标准接口，支持不同的存储实现（内存、Redis、数据库等）。
    """

    def get(self, session_id: str) -> SessionStateSnapshot | None:
        """根据会话 ID 获取会话状态快照。

        Args:
            session_id: 会话唯一标识符

        Returns:
            会话状态快照，如果不存在则返回 None
        """
        ...

    def save(self, state: SessionStateSnapshot) -> None:
        """保存会话状态快照。

        Args:
            state: 要保存的会话状态快照
        """
        ...


class InMemorySessionStore:
    """基于内存的会话状态存储实现。

    适用于单机部署或开发测试环境。生产环境建议使用 Redis 或数据库实现。
    注意：进程重启后所有会话状态会丢失。
    """

    def __init__(self) -> None:
        """初始化内存存储。"""
        self._states: dict[str, SessionStateSnapshot] = {}

    def get(self, session_id: str) -> SessionStateSnapshot | None:
        """从内存中获取会话状态。

        Args:
            session_id: 会话唯一标识符

        Returns:
            会话状态快照，如果不存在则返回 None
        """
        return self._states.get(session_id)

    def save(self, state: SessionStateSnapshot) -> None:
        """将会话状态保存到内存。

        Args:
            state: 要保存的会话状态快照
        """
        self._states[state.session_id] = state


class MemoryManager:
    """会话短期记忆管理器。

    负责管理单个会话的结构化业务状态，包括：
    - 槽位信息（公司名、产品名、申请 ID 等）
    - 业务上下文（当前场景、能力、等待的槽位）
    - 工具调用历史和结果缓存
    - 确认状态和待处理动作

    核心职责：
    1. 加载和保存会话状态
    2. 根据系统事件（路由、工具结果）自动更新状态
    3. 将状态注入到请求 metadata 中供下游使用

    设计原则：
    - 规则驱动：通过确定性逻辑处理系统事件，保证性能和可预测性
    - 单一职责：只管理状态的 CRUD，不负责业务决策
    - 可扩展：支持自定义存储实现（内存、Redis、数据库）
    """

    def __init__(self, store: SessionStore | None = None, max_tool_results: int = 5) -> None:
        """初始化记忆管理器。

        Args:
            store: 会话状态存储实现，默认使用内存存储
            max_tool_results: 保留的最近工具调用结果数量，用于控制状态大小
        """
        self._store = store or InMemorySessionStore()
        self._max_tool_results = max_tool_results

    def load(self, session_id: str | None) -> SessionStateSnapshot:
        """加载或初始化会话状态。

        如果会话 ID 存在则从存储中加载，否则创建新的会话状态。
        支持传入 None 来自动生成新会话 ID。

        Args:
            session_id: 会话唯一标识符，为 None 时自动生成

        Returns:
            会话状态快照
        """
        resolved_session_id = session_id or str(uuid4())
        existing = self._store.get(resolved_session_id)
        if existing is not None:
            return existing
        return SessionStateSnapshot(
            session_id=resolved_session_id,
            updated_at=_now(),
        )

    def save(self, state: SessionStateSnapshot) -> None:
        """保存会话状态到存储。

        自动更新 updated_at 时间戳。

        Args:
            state: 要保存的会话状态快照
        """
        self._store.save(state.model_copy(update={"updated_at": _now()}))

    def enrich_request(self, request: ChatRequest, state: SessionStateSnapshot) -> ChatRequest:
        """将会话状态注入到请求的 metadata 中。

        将会话中已确认的槽位、业务上下文等信息注入到请求 metadata，
        供路由器、工具规划器等下游组件使用。

        注入规则：
        1. 已确认的槽位（company_name、product_name 等）
        2. 业务上下文字段（selected_company_name、selected_product_name）
        3. 内部状态字段（以 _ 开头）：
           - _session_awaiting_slots: 等待填充的槽位列表
           - _session_resume_route: 可恢复的路由信息
           - _session_summary: 会话状态摘要

        注意：只有当 metadata 中不存在对应 key 时才注入，避免覆盖请求中的显式参数。

        Args:
            request: 原始聊天请求
            state: 当前会话状态

        Returns:
            注入了会话状态的新请求对象
        """
        metadata = dict(request.metadata)
        metadata["session_id"] = state.session_id

        for key, value in state.confirmed_slots.items():
            if value and key not in metadata:
                metadata[key] = value

        if state.selected_company_name and "company_name" not in metadata:
            metadata["company_name"] = state.selected_company_name
        if state.selected_product_name and "product_name" not in metadata:
            metadata["product_name"] = state.selected_product_name

        if state.awaiting_slots:
            metadata["_session_awaiting_slots"] = list(state.awaiting_slots)
        if state.last_route and self._can_resume_route(state, metadata):
            metadata["_session_resume_route"] = state.last_route.model_dump()
        if state.short_summary:
            metadata["_session_summary"] = state.short_summary

        return request.model_copy(update={"session_id": state.session_id, "metadata": metadata})

    def update(
        self,
        state: SessionStateSnapshot,
        request: ChatRequest,
        route: RouteDecision | None = None,
        tool_result: ToolResult | None = None,
        pending_action: PendingAction | None = None,
        stop_reason: str | None = None,
    ) -> SessionStateSnapshot:
        """根据系统事件更新会话状态。

        这是会话状态更新的核心方法，根据不同的系统事件（路由决策、工具执行结果等）
        自动更新会话状态。所有更新逻辑都是规则驱动的，保证确定性和性能。

        更新来源和优先级：
        1. 路由结果（route）：更新场景、能力、槽位填充情况
        2. 工具结果（tool_result）：更新工具调用历史、业务结果缓存
        3. 确认动作（pending_action）：更新确认状态
        4. 停止原因（stop_reason）：影响确认状态的清理

        状态更新规则：
        - confirmed_slots: 从路由填充的槽位和工具返回的槽位合并
        - pending_slots: 当前轮次待确认的槽位
        - awaiting_slots: 路由识别出的缺失槽位
        - selected_company_name/selected_product_name: 从 confirmed_slots 同步
        - last_credit_amount: 从 query_credit_amount 工具结果缓存
        - last_tool_results: 保留最近 N 次工具调用结果

        注意：
        - 工具结果中的槽位会覆盖路由填充的槽位（工具结果优先级更高）
        - 更新后自动保存到存储
        - 自动生成 short_summary 用于日志和调试

        Args:
            state: 当前会话状态
            request: 本轮聊天请求
            route: 路由决策结果，包含场景、槽位填充情况
            tool_result: 工具执行结果，包含业务状态和输出数据
            pending_action: 待用户确认的动作
            stop_reason: 本轮对话停止原因

        Returns:
            更新后的会话状态快照
        """
        confirmed_slots = dict(state.confirmed_slots)
        pending_slots = dict(state.pending_slots)
        awaiting_slots = list(state.awaiting_slots)
        last_route = state.last_route
        active_scene = state.active_scene
        active_capability_id = state.active_capability_id
        selected_company_name = state.selected_company_name
        selected_product_name = state.selected_product_name
        last_credit_amount = state.last_credit_amount
        last_tool_results = list(state.last_tool_results)
        confirmation_status = state.confirmation_status
        next_pending_action = pending_action if pending_action is not None else state.pending_action

        # 处理路由结果：更新场景、能力、槽位
        if route is not None:
            last_route = route
            active_scene = route.scene
            active_capability_id = route.capability_id
            for slot, value in route.filled_slots.items():
                if value:
                    confirmed_slots[slot] = value
            pending_slots = dict(route.filled_slots)
            awaiting_slots = list(route.missing_slots)
            if not awaiting_slots and stop_reason != "waiting_confirmation":
                pending_slots = {}

        # 从 confirmed_slots 同步到业务字段
        selected_company_name = confirmed_slots.get("company_name") or selected_company_name
        selected_product_name = confirmed_slots.get("product_name") or selected_product_name

        # 处理确认状态
        if pending_action is not None:
            confirmation_status = "waiting"
        elif stop_reason != "waiting_confirmation":
            confirmation_status = "none"
            next_pending_action = None

        # 处理工具结果：记录历史、缓存业务数据
        if tool_result is not None:
            last_tool_results.append(
                SessionToolResult(
                    tool_name=tool_result.tool_name,
                    business_status=tool_result.business_status,
                    output=tool_result.output,
                    created_at=_now(),
                )
            )
            last_tool_results = last_tool_results[-self._max_tool_results :]
            output = tool_result.output
            # 工具返回的槽位优先级高于路由填充
            if output.get("company_name"):
                confirmed_slots["company_name"] = output["company_name"]
                selected_company_name = str(output["company_name"])
            # 缓存额度查询结果，避免重复调用
            if tool_result.tool_name == "query_credit_amount" and output.get("credit_amount"):
                last_credit_amount = {
                    "company_name": output.get("company_name"),
                    "credit_amount": output.get("credit_amount"),
                    "data_time": output.get("data_time"),
                    "business_status": tool_result.business_status,
                }

        updated = state.model_copy(
            update={
                "active_scene": active_scene,
                "active_capability_id": active_capability_id,
                "confirmed_slots": confirmed_slots,
                "pending_slots": pending_slots,
                "awaiting_slots": awaiting_slots,
                "last_route": last_route,
                "last_tool_results": last_tool_results,
                "pending_action": next_pending_action,
                "confirmation_status": confirmation_status,
                "selected_company_name": selected_company_name,
                "selected_product_name": selected_product_name,
                "last_credit_amount": last_credit_amount,
                "short_summary": _build_summary(
                    active_scene=active_scene,
                    active_capability_id=active_capability_id,
                    selected_company_name=selected_company_name,
                    selected_product_name=selected_product_name,
                    awaiting_slots=awaiting_slots,
                    last_credit_amount=last_credit_amount,
                    confirmation_status=confirmation_status,
                ),
                "turn_count": state.turn_count + 1,
                "updated_at": _now(),
            }
        )
        self._store.save(updated)
        return updated

    def _can_resume_route(self, state: SessionStateSnapshot, metadata: dict[str, Any]) -> bool:
        """判断是否可以恢复上一次的路由。

        当上一轮对话因缺少槽位而中断，本轮对话的 metadata 中补充了缺失槽位时，
        可以恢复上一次的路由，避免重新进行意图识别。

        Args:
            state: 当前会话状态
            metadata: 本轮请求的 metadata

        Returns:
            True 表示可以恢复路由，False 表示需要重新路由
        """
        if state.last_route is None or not state.awaiting_slots:
            return False
        return any(metadata.get(slot) for slot in state.awaiting_slots)


def _build_summary(
    active_scene: str | None,
    active_capability_id: str | None,
    selected_company_name: str | None,
    selected_product_name: str | None,
    awaiting_slots: list[str],
    last_credit_amount: dict[str, Any] | None,
    confirmation_status: str,
) -> str:
    """构建会话状态的简短摘要。

    生成一行文本摘要，用于日志记录、调试和传递给下游组件。
    摘要格式：key=value; key=value; ...

    Args:
        active_scene: 当前激活的业务场景
        active_capability_id: 当前激活的能力 ID
        selected_company_name: 选中的企业名称
        selected_product_name: 选中的产品名称
        awaiting_slots: 等待填充的槽位列表
        last_credit_amount: 最近一次额度查询结果
        confirmation_status: 确认状态

    Returns:
        会话状态摘要字符串
    """
    parts: list[str] = []
    if active_scene:
        parts.append(f"active_scene={active_scene}")
    if active_capability_id:
        parts.append(f"capability={active_capability_id}")
    if selected_company_name:
        parts.append(f"company={selected_company_name}")
    if selected_product_name:
        parts.append(f"product={selected_product_name}")
    if last_credit_amount:
        parts.append(f"last_credit_amount={last_credit_amount.get('credit_amount')}")
    if awaiting_slots:
        parts.append(f"awaiting={','.join(awaiting_slots)}")
    if confirmation_status != "none":
        parts.append(f"confirmation={confirmation_status}")
    return "; ".join(parts)


def _now() -> str:
    """获取当前 UTC 时间的 ISO 格式字符串。

    Returns:
        ISO 8601 格式的时间字符串
    """
    return datetime.now(UTC).isoformat()
