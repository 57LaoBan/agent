export type EventVisibility = "user" | "diagnostic";

export type AgentEventType =
  | "turn_started"
  | "session_loaded"
  | "system_tool_started"
  | "system_tool_result"
  | "session_updated"
  | "state_changed"
  | "runtime_step_started"
  | "runtime_step_decision"
  | "runtime_step_finished"
  | "route_started"
  | "route_decision"
  | "tool_call_proposed"
  | "pending_action_validated"
  | "confirmation_required"
  | "handoff_required"
  | "tool_started"
  | "tool_result"
  | "assistant_delta"
  | "final_answer"
  | "turn_finished"
  | "error";

export type AgentEvent = {
  turn_id: string;
  sequence: number;
  event_type: AgentEventType;
  visibility: EventVisibility;
  timestamp: string;
  payload: Record<string, unknown>;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: "streaming" | "done";
};

export type ConfirmActionRequest = {
  session_id: string;
  action_id: string;
  confirmed: boolean;
  user_message?: string | null;
  metadata: Record<string, unknown>;
};

export type PendingAction = {
  action_id: string;
  tool_call?: Record<string, unknown>;
  title: string;
  summary: string;
  risk_level?: "read_only" | "link_create" | "state_create" | "state_update" | "final_submit";
  confirm_label: string;
  cancel_label: string;
  details: Array<{ label: string; value: string }>;
  session_id?: string | null;
  scene?: string | null;
  capability_id?: string | null;
  stage?: string | null;
  slot_snapshot?: Record<string, unknown>;
  precondition_hash?: string | null;
  expires_at?: string | null;
  created_at?: string | null;
};

export type RuntimeStepDecision = {
  step_type:
    | "ROUTE"
    | "ASK_USER"
    | "ANSWER_WITHOUT_TOOL"
    | "PROPOSE_TOOL"
    | "EXECUTE_TOOL"
    | "PROPOSE_PENDING_ACTION"
    | "WAIT_CONFIRMATION"
    | "GENERATE_FINAL_ANSWER"
    | "HANDOFF"
    | "STOP";
  reason: string;
  confidence: number;
  tool_call?: Record<string, unknown> | null;
  pending_action?: PendingAction | null;
  answer?: string | null;
  question?: string | null;
  missing_slots: string[];
  handoff_reason?: string | null;
};

export type RouteSnapshot = {
  scene: string;
  intent: string;
  raw_intent?: string | null;
  capability_id?: string | null;
  capability_source?: string;
  confirmation_required?: boolean;
  confidence: number;
  allowed_tools: string[];
  allowed_tool_categories: string[];
  missing_slots: string[];
  risk_level: string;
  route_reason: string;
  route_source?: string;
};

export type ToolSnapshot = {
  tool_call_id: string;
  tool_name: string;
  tool_category?: string;
  status: "proposed" | "running" | "success" | "failed" | "blocked";
  risk_level: string;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  phases: string[];
};

export type SessionStateSnapshot = {
  session_id: string;
  active_scene?: string | null;
  active_capability_id?: string | null;
  active_flow?: string | null;
  current_stage?: string | null;
  confirmed_slots: Record<string, unknown>;
  pending_slots: Record<string, unknown>;
  awaiting_slots: string[];
  selected_company_name?: string | null;
  selected_product_name?: string | null;
  last_credit_amount?: Record<string, unknown> | null;
  last_application_id?: string | null;
  authorization_status?: string | null;
  confirmation_status: "none" | "waiting" | "confirmed" | "cancelled";
  short_summary: string;
  recent_turns: Array<Record<string, unknown>>;
  completed_stages: string[];
  stage_history: Array<Record<string, unknown>>;
  turn_count: number;
  updated_at: string;
};
