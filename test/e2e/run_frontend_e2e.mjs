import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import path from "node:path";
import process from "node:process";
import { spawn } from "node:child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "../..");
const webDir = path.join(root, "web");
const fixturePath = path.join(__dirname, "fixtures", "e2e_cases.json");
const backendUrl = process.env.E2E_AGENT_API_URL ?? "http://127.0.0.1:8000";
const frontendUrl = process.env.E2E_FRONTEND_URL ?? "http://127.0.0.1:5173";
const requireFromWeb = createRequire(path.join(webDir, "package.json"));
const { chromium } = requireFromWeb("playwright");

const started = [];

async function main() {
  console.log("[e2e] 读取前端端到端样本");
  const args = new Set(process.argv.slice(2));
  const fixtures = JSON.parse(await readFile(fixturePath, "utf-8"));
  const caseId = readArgValue("--case");
  const cases = caseId
    ? fixtures.frontend_cases.filter((item) => item.id === caseId)
    : fixtures.frontend_cases;

  if (cases.length === 0) {
    throw new Error(`未找到前端 e2e case：${caseId}`);
  }

  if (args.has("--start-backend")) {
    console.log(`[e2e] 启动后端：${backendUrl}`);
    started.push(startBackend());
    await waitForHttp(`${backendUrl}/health`, 30000);
  }

  if (args.has("--start-frontend")) {
    console.log(`[e2e] 启动前端：${frontendUrl}`);
    started.push(startFrontend());
    await waitForHttp(frontendUrl, 30000);
  }

  console.log("[e2e] 启动浏览器");
  const browser = await chromium.launch({ headless: !args.has("--headed") });
  try {
    const page = await browser.newPage();
    for (const item of cases) {
      await runCase(page, item);
    }
    console.log(`[e2e] 前端端到端测试通过：${cases.length} 个 case`);
  } finally {
    await browser.close();
    for (const child of started.reverse()) {
      child.kill();
    }
  }
}

async function runCase(page, item) {
  await page.goto(frontendUrl, { waitUntil: "networkidle" });
  await page.getByLabel("输入问题").fill(item.message);
  await page.locator(".send-button").click();

  await page.getByText(item.expected_visible_text).waitFor({ timeout: 120000 });
  await page.getByText(item.expected_scene).waitFor({ timeout: 10000 });
  await page.getByText(item.expected_tool).waitFor({ timeout: 10000 });
  await page.getByText(item.expected_tool_category).waitFor({ timeout: 10000 });

  if (item.expected_allowed_tools) {
    for (const tool of item.expected_allowed_tools) {
      await page.getByText(tool).waitFor({ timeout: 10000 });
    }
  }

  console.log(`[e2e] PASS frontend/${item.id}`);
}

function startBackend() {
  const child = spawn(
    process.platform === "win32" ? "python" : "python3",
    [
      "-m",
      "uvicorn",
      "xinyidai_agent.api:create_app",
      "--factory",
      "--host",
      "127.0.0.1",
      "--port",
      backendUrl.split(":").at(-1),
    ],
    {
      cwd: root,
      env: {
        ...process.env,
        PYTHONPATH: path.join(root, "src"),
      },
      stdio: "inherit",
    },
  );
  return child;
}

function startFrontend() {
  const port = frontendUrl.split(":").at(-1);
  const env = {
    ...process.env,
    VITE_AGENT_API_BASE_URL: backendUrl,
    VITE_USE_MOCK_AGENT: "false",
  };
  const child =
    process.platform === "win32"
      ? spawn(
          "powershell.exe",
          ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", `npm run dev -- --port ${port}`],
          {
            cwd: webDir,
            env,
            stdio: "inherit",
          },
        )
      : spawn("npm", ["run", "dev", "--", "--port", port], {
          cwd: webDir,
          env,
          stdio: "inherit",
        });
  return child;
}

async function waitForHttp(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // 等待服务启动。
    }
    await sleep(500);
  }
  throw new Error(`服务未在 ${timeoutMs}ms 内就绪：${url}`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function readArgValue(name) {
  const index = process.argv.indexOf(name);
  if (index === -1) return null;
  return process.argv[index + 1] ?? null;
}

main().catch((error) => {
  console.error(error);
  for (const child of started.reverse()) {
    child.kill();
  }
  process.exit(1);
});
