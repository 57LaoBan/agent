import { runMockTurn } from "./mockAgent";
import type { AgentEvent } from "./types";

type Emit = (event: AgentEvent) => void;

const apiBaseUrl = import.meta.env.VITE_AGENT_API_BASE_URL ?? "http://127.0.0.1:8000";
const useMockAgent = import.meta.env.VITE_USE_MOCK_AGENT === "true";

export function getAgentRuntimeLabel() {
  return useMockAgent ? "Mock Event Stream" : `Live SSE ${apiBaseUrl}`;
}

export async function runAgentTurn(message: string, emit: Emit) {
  if (useMockAgent) {
    await runMockTurn(message, emit);
    return;
  }

  const response = await fetch(`${apiBaseUrl}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_message: message,
      top_k: 5,
    }),
  });

  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw new Error(`Agent SSE 请求失败：${response.status} ${detail}`);
  }

  await readSse(response.body, emit);
}

async function readSse(stream: ReadableStream<Uint8Array>, emit: Emit) {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const event = parseSseFrame(frame);
      if (event) emit(event);
    }
  }

  buffer += decoder.decode();
  const event = parseSseFrame(buffer);
  if (event) emit(event);
}

function parseSseFrame(frame: string): AgentEvent | null {
  const dataLines = frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart());

  if (dataLines.length === 0) return null;

  return JSON.parse(dataLines.join("\n")) as AgentEvent;
}
