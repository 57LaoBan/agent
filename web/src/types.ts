export type EventVisibility = "user" | "diagnostic";

export type AgentEventType =
  | "turn_started"
  | "state_changed"
  | "route_started"
  | "route_decision"
  | "tool_call_proposed"
  | "confirmation_required"
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

export type PendingAction = {
  action_id: string;
  tool_call?: Record<string, unknown>;
  title: string;
  summary: string;
  risk_level?: "read_only" | "link_create" | "state_create" | "state_update" | "final_submit";
  confirm_label: string;
  cancel_label: string;
  details: Array<{ label: string; value: string }>;
};

export type RouteSnapshot = {
  scene: string;
  intent: string;
  confidence: number;
  allowed_tools: string[];
  allowed_tool_categories: string[];
  missing_slots: string[];
  risk_level: string;
  route_reason: string;
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
