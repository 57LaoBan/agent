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

function classify(message: string) {
  if (message.includes("申请") || message.includes("办理")) {
    return {
      scene: "LOAN_APPLY",
      intent: "CREATE_APPLICATION",
      confidence: 0.88,
      allowed_tools: ["search_product", "create_application", "create_authorization_link"],
      missing_slots: [],
      risk_level: "state_create",
      route_reason: "用户表达了贷款申请意图，需要在创建申请前确认企业和产品信息。",
    };
  }

  if (message.includes("额度") || message.includes("能贷") || message.includes("多少钱")) {
    return {
      scene: "DATA_QUERY",
      intent: "CREDIT_LIMIT_QUERY",
      confidence: 0.91,
      allowed_tools: ["query_credit_amount"],
      missing_slots: [],
      risk_level: "read_only",
      route_reason: "用户询问可贷额度，必须通过只读数据工具返回数值。",
    };
  }

  return {
    scene: "KNOWLEDGE_QA",
    intent: "POLICY_OR_PRODUCT_QA",
    confidence: 0.8,
    allowed_tools: ["rag_search"],
    missing_slots: [],
    risk_level: "read_only",
    route_reason: "用户咨询政策或产品信息，优先走知识库问答。",
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

  emitEvent("route_started", "diagnostic", { strategy: "rule_first_then_model" });
  const route = classify(message);
  await sleep(260);
  emitEvent("route_decision", "diagnostic", route);

  if (route.scene === "LOAN_APPLY") {
    emitEvent("state_changed", "diagnostic", { state: "WAITING_CONFIRMATION" });
    const pendingAction: PendingAction = {
      action_id: crypto.randomUUID(),
      title: "确认创建贷款申请",
      summary: "将为示例企业创建税贷申请，并生成后续授权链接。",
      risk_level: "state_create",
      confirm_label: "确认创建",
      cancel_label: "取消",
      details: [
        { label: "企业", value: "杭州示例科技有限公司" },
        { label: "产品", value: "小微税贷" },
        { label: "动作", value: "创建申请草稿" },
      ],
    };
    emitEvent("confirmation_required", "user", { pending_action: pendingAction });
    emitEvent("turn_finished", "diagnostic", { stop_reason: "waiting_confirmation" });
    return;
  }

  const toolName = route.scene === "DATA_QUERY" ? "query_credit_amount" : "rag_search";
  emitEvent("tool_call_proposed", "diagnostic", {
    tool_name: toolName,
    risk_level: "read_only",
    input: { query: message, company_name: "杭州示例科技有限公司" },
  });
  await sleep(180);
  emitEvent("state_changed", "diagnostic", { state: "EXECUTING_TOOL" });
  emitEvent("tool_started", "diagnostic", { tool_name: toolName });
  await sleep(520);

  if (route.scene === "DATA_QUERY") {
    emitEvent("tool_result", "diagnostic", {
      tool_name: toolName,
      status: "success",
      output: {
        company_name: "杭州示例科技有限公司",
        credit_amount: "50万元",
        data_time: "2026-04-28",
      },
      terminal: true,
    });
    emitEvent("state_changed", "diagnostic", { state: "ANSWERING" });
    await streamText(
      emitEvent,
      "根据授信额度查询结果，杭州示例科技有限公司当前可申请额度为 50 万元。该结果来自只读 mock 数据接口，实际生产环境需要以银行和平台实时数据为准。",
    );
  } else {
    emitEvent("tool_result", "diagnostic", {
      tool_name: toolName,
      status: "success",
      output: {
        sources: [
          { title: "信易贷政策说明", score: 0.92 },
          { title: "小微企业融资服务指南", score: 0.87 },
        ],
      },
      terminal: true,
    });
    emitEvent("state_changed", "diagnostic", { state: "ANSWERING" });
    await streamText(
      emitEvent,
      "信易贷主要面向企业融资服务场景，通过政策、产品和信用信息匹配，帮助企业了解可申请产品、准入条件和办理路径。当前回答基于模拟知识库证据生成，后续会接入真实 RAG 来源。",
    );
  }

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
