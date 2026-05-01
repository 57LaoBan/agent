import type { AgentEvent, PendingAction } from "./types";

type Emit = (event: AgentEvent) => void;

const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

function createEmitter(turnId: string, emit: Emit) {
  let sequence = 0;
  return (event_type: AgentEvent["event_type"], visibility: AgentEvent["visibility"], payload: Record<string, unknown>) => {
    sequence += 1;
    emit({
      turn_id: turnId,
      sequence,
      event_type,
      visibility,
      timestamp: new Date().toISOString(),
      payload,
    });
  };
}

function mockRoute() {
  return {
    scene: "UNKNOWN",
    intent: "MOCK_MODE_DISABLED_ROUTING",
    confidence: 0,
    allowed_tools: [],
    missing_slots: [],
    risk_level: "read_only",
    route_reason: "前端 mock 模式不模拟业务意图识别；请切换到 Live SSE 验证真实模型路由。",
  };
}

async function streamText(emitEvent: ReturnType<typeof createEmitter>, text: string) {
  const chunks = text.match(/.{1,9}/g) ?? [text];
  let answer = "";
  for (const chunk of chunks) {
    answer += chunk;
    emitEvent("assistant_delta", "user", { text: chunk, answer });
    await sleep(72);
  }
  emitEvent("final_answer", "user", { answer });
}

export async function runMockTurn(message: string, emit: Emit) {
  const turnId = crypto.randomUUID();
  const emitEvent = createEmitter(turnId, emit);

  emitEvent("turn_started", "diagnostic", { user_message: message });
  emitEvent("state_changed", "diagnostic", { state: "ROUTING" });
  await sleep(180);

  emitEvent("route_started", "diagnostic", { strategy: "frontend_mock_no_business_routing" });
  const route = mockRoute();
  await sleep(260);
  emitEvent("route_decision", "diagnostic", route);

  emitEvent("state_changed", "diagnostic", { state: "ANSWERING" });
  await streamText(emitEvent, "当前是前端 mock 模式，不执行业务意图识别和工具调用。请关闭 VITE_USE_MOCK_AGENT，连接后端 Live SSE 后验证真实模型判断。");

  emitEvent("turn_finished", "diagnostic", { stop_reason: "completed" });
}

export async function confirmMockAction(action: PendingAction, emit: Emit) {
  const turnId = crypto.randomUUID();
  const emitEvent = createEmitter(turnId, emit);

  emitEvent("turn_started", "diagnostic", { confirmed_action_id: action.action_id });
  emitEvent("state_changed", "diagnostic", { state: "EXECUTING_TOOL" });
  emitEvent("tool_started", "diagnostic", {
    tool_name: "create_application",
    action_id: action.action_id,
  });
  await sleep(520);
  emitEvent("tool_result", "diagnostic", {
    tool_name: "create_application",
    status: "success",
    output: {
      application_id: "APP-20260428-0001",
      apply_url: "https://xyd.example/apply/APP-20260428-0001",
      auth_url: "https://xyd.example/auth/APP-20260428-0001",
      business_status: "APPLICATION_DRAFT_CREATED",
      next_required_action: "AUTHORIZATION_REQUIRED",
    },
  });
  emitEvent("state_changed", "diagnostic", { state: "ANSWERING" });
  await streamText(
    emitEvent,
    "申请草稿已创建，申请编号为 APP-20260428-0001。请先完成企业授权，授权完成后再继续提交贷款申请。",
  );
  emitEvent("turn_finished", "diagnostic", { stop_reason: "completed" });
}
