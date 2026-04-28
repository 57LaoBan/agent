import {
  Activity,
  ArrowUp,
  Bot,
  Check,
  CircleAlert,
  ClipboardList,
  FileSearch,
  Gauge,
  Loader2,
  MessageSquareText,
  PanelRight,
  ShieldCheck,
  Sparkles,
  UserRound,
  X,
} from "lucide-react";
import { FormEvent, useMemo, useState } from "react";
import { confirmMockAction, runMockTurn } from "./mockAgent";
import type { AgentEvent, ChatMessage, PendingAction, RouteSnapshot, ToolSnapshot } from "./types";

const presets = ["我能贷多少钱？", "我要申请小微税贷", "信易贷适合哪些企业？"];
const tabs = ["意图", "工具", "状态", "事件"] as const;

export function App() {
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "你好，我是信易贷 Agent。你可以咨询政策、查询额度，也可以发起贷款申请流程。",
      status: "done",
    },
  ]);
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [activeTab, setActiveTab] = useState<(typeof tabs)[number]>("意图");

  const route = useMemo(() => {
    const event = [...events].reverse().find((item) => item.event_type === "route_decision");
    return (event?.payload ?? null) as RouteSnapshot | null;
  }, [events]);

  const tools = useMemo(() => {
    const snapshots: ToolSnapshot[] = [];
    for (const event of events) {
      if (event.event_type === "tool_call_proposed") {
        snapshots.push({
          tool_name: String(event.payload.tool_name),
          status: "proposed",
          risk_level: String(event.payload.risk_level ?? "unknown"),
          input: event.payload.input as Record<string, unknown> | undefined,
        });
      }
      if (event.event_type === "tool_started") {
        snapshots.push({
          tool_name: String(event.payload.tool_name),
          status: "running",
          risk_level: "unknown",
        });
      }
      if (event.event_type === "tool_result") {
        snapshots.push({
          tool_name: String(event.payload.tool_name),
          status: "success",
          risk_level: "unknown",
          output: event.payload.output as Record<string, unknown> | undefined,
        });
      }
    }
    return snapshots;
  }, [events]);

  const states = useMemo(
    () => events.filter((event) => event.event_type === "state_changed").map((event) => String(event.payload.state)),
    [events],
  );

  const appendEvent = (event: AgentEvent) => {
    setEvents((current) => [...current, event]);

    if (event.visibility !== "user") return;

    if (event.event_type === "assistant_delta") {
      const answer = String(event.payload.answer ?? "");
      setMessages((current) => {
        const last = current[current.length - 1];
        if (last?.role === "assistant" && last.status === "streaming") {
          return [...current.slice(0, -1), { ...last, content: answer }];
        }
        return [
          ...current,
          { id: `${event.turn_id}-assistant`, role: "assistant", content: answer, status: "streaming" },
        ];
      });
    }

    if (event.event_type === "final_answer") {
      const answer = String(event.payload.answer ?? "");
      setMessages((current) => {
        const last = current[current.length - 1];
        if (last?.role === "assistant" && last.status === "streaming") {
          return [...current.slice(0, -1), { ...last, content: answer, status: "done" }];
        }
        return [...current, { id: `${event.turn_id}-final`, role: "assistant", content: answer, status: "done" }];
      });
    }

    if (event.event_type === "confirmation_required") {
      setPendingAction(event.payload.pending_action as PendingAction);
    }
  };

  const sendMessage = async (message: string) => {
    const text = message.trim();
    if (!text || running) return;

    setInput("");
    setPendingAction(null);
    setRunning(true);
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", content: text }]);

    try {
      await runMockTurn(text, appendEvent);
    } finally {
      setRunning(false);
    }
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void sendMessage(input);
  };

  const confirmAction = async () => {
    if (!pendingAction || running) return;
    const action = pendingAction;
    setPendingAction(null);
    setRunning(true);
    try {
      await confirmMockAction(action, appendEvent);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={20} />
          </div>
          <div>
            <h1>信易贷 Agent 工作台</h1>
            <p>问答和工具调用的受控流式原型</p>
          </div>
        </div>
        <div className="topbar-status">
          <span className="status-dot" />
          Mock Event Stream
        </div>
      </header>

      <main className="workspace">
        <section className="chat-pane">
          <div className="chat-header">
            <div>
              <h2>业务会话</h2>
              <p>聊天区只展示用户可见内容</p>
            </div>
            <div className="loop-badge">
              {running ? <Loader2 className="spin" size={16} /> : <Activity size={16} />}
              {running ? "执行中" : "就绪"}
            </div>
          </div>

          <div className="messages">
            {messages.map((message) => (
              <div className={`message-row ${message.role}`} key={message.id}>
                <div className="avatar">{message.role === "assistant" ? <Bot size={17} /> : <UserRound size={17} />}</div>
                <div className="bubble">
                  <p>{message.content}</p>
                  {message.status === "streaming" && <span className="cursor" />}
                </div>
              </div>
            ))}

            {pendingAction && (
              <div className="confirmation">
                <div className="confirmation-title">
                  <CircleAlert size={18} />
                  <span>{pendingAction.title}</span>
                </div>
                <p>{pendingAction.summary}</p>
                <div className="detail-grid">
                  {pendingAction.details.map((item) => (
                    <div key={item.label}>
                      <span>{item.label}</span>
                      <strong>{item.value}</strong>
                    </div>
                  ))}
                </div>
                <div className="confirmation-actions">
                  <button className="secondary-button" onClick={() => setPendingAction(null)} type="button">
                    <X size={16} />
                    {pendingAction.cancel_label}
                  </button>
                  <button className="primary-button" disabled={running} onClick={confirmAction} type="button">
                    <Check size={16} />
                    {pendingAction.confirm_label}
                  </button>
                </div>
              </div>
            )}
          </div>

          <form className="composer" onSubmit={onSubmit}>
            <div className="preset-row">
              {presets.map((preset) => (
                <button disabled={running} key={preset} onClick={() => void sendMessage(preset)} type="button">
                  <Sparkles size={14} />
                  {preset}
                </button>
              ))}
            </div>
            <div className="input-row">
              <MessageSquareText size={20} />
              <input
                aria-label="输入问题"
                disabled={running}
                onChange={(event) => setInput(event.target.value)}
                placeholder="输入你的问题，例如：这家企业能贷多少钱？"
                value={input}
              />
              <button className="send-button" disabled={running || !input.trim()} type="submit">
                <ArrowUp size={18} />
              </button>
            </div>
          </form>
        </section>

        <aside className="diagnostic-pane">
          <div className="diagnostic-header">
            <div>
              <h2>检测窗口</h2>
              <p>路由、工具、状态和原始事件</p>
            </div>
            <PanelRight size={19} />
          </div>

          <div className="tabs">
            {tabs.map((tab) => (
              <button className={tab === activeTab ? "active" : ""} key={tab} onClick={() => setActiveTab(tab)} type="button">
                {tab}
              </button>
            ))}
          </div>

          <div className="diagnostic-content">
            {activeTab === "意图" && <RoutePanel route={route} />}
            {activeTab === "工具" && <ToolPanel tools={tools} />}
            {activeTab === "状态" && <StatePanel states={states} />}
            {activeTab === "事件" && <EventPanel events={events} />}
          </div>
        </aside>
      </main>
    </div>
  );
}

function RoutePanel({ route }: { route: RouteSnapshot | null }) {
  if (!route) return <EmptyState icon={<FileSearch size={20} />} text="等待意图识别事件" />;

  return (
    <div className="panel-stack">
      <Metric label="场景" value={route.scene} />
      <Metric label="意图" value={route.intent} />
      <Metric label="置信度" value={`${Math.round(route.confidence * 100)}%`} />
      <Metric label="风险等级" value={route.risk_level} />
      <div className="tool-list">
        <span>允许工具</span>
        {route.allowed_tools.map((tool) => (
          <code key={tool}>{tool}</code>
        ))}
      </div>
      <p className="reason">{route.route_reason}</p>
    </div>
  );
}

function ToolPanel({ tools }: { tools: ToolSnapshot[] }) {
  if (tools.length === 0) return <EmptyState icon={<ClipboardList size={20} />} text="暂无工具调用" />;

  return (
    <div className="panel-stack">
      {tools.map((tool, index) => (
        <div className="tool-item" key={`${tool.tool_name}-${index}`}>
          <div>
            <strong>{tool.tool_name}</strong>
            <span>{tool.status}</span>
          </div>
          <pre>{JSON.stringify(tool.input ?? tool.output ?? {}, null, 2)}</pre>
        </div>
      ))}
    </div>
  );
}

function StatePanel({ states }: { states: string[] }) {
  if (states.length === 0) return <EmptyState icon={<Gauge size={20} />} text="暂无状态迁移" />;

  return (
    <div className="state-list">
      {states.map((state, index) => (
        <div className="state-item" key={`${state}-${index}`}>
          <span>{index + 1}</span>
          <strong>{state}</strong>
        </div>
      ))}
    </div>
  );
}

function EventPanel({ events }: { events: AgentEvent[] }) {
  if (events.length === 0) return <EmptyState icon={<Activity size={20} />} text="暂无事件" />;

  return (
    <div className="event-list">
      {[...events].reverse().map((event) => (
        <div className="event-item" key={`${event.turn_id}-${event.sequence}`}>
          <div>
            <strong>{event.event_type}</strong>
            <span>#{event.sequence}</span>
          </div>
          <pre>{JSON.stringify(event.payload, null, 2)}</pre>
        </div>
      ))}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyState({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <div className="empty-state">
      {icon}
      <span>{text}</span>
    </div>
  );
}
