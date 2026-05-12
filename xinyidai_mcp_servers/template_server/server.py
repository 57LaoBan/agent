"""通用模板生成 MCP Server。

对外工具：
- generate_markdown_template：按场景生成 Markdown 模板文件；
- generate_excel_template：按列定义生成 Excel 模板文件；
- list_generated_templates：列出已生成的模板文件。
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from openpyxl import Workbook


_OUTPUT_DIR = Path(os.environ.get("TEMPLATE_OUTPUT_DIR", ".data/mcp_templates")).resolve()

_MARKDOWN_TEMPLATES: dict[str, str] = {
    "loan_apply": (
        "# 贷款申请采集模板\n\n"
        "## 基本信息\n"
        "- 客户名称：\n"
        "- 统一社会信用代码：\n"
        "- 联系电话：\n\n"
        "## 申请要素\n"
        "- 申请额度（万元）：\n"
        "- 期限（月）：\n"
        "- 用途：\n"
    ),
    "customer_intake": (
        "# 客户准入信息采集模板\n\n"
        "## 主体信息\n"
        "- 主体名称：\n"
        "- 注册地址：\n"
        "- 法人代表：\n\n"
        "## 经营信息\n"
        "- 主营业务：\n"
        "- 上一年度营收（万元）：\n"
    ),
}


server = Server("xinyidai-template-server")


@server.list_tools()
async def _list_tools() -> list[Tool]:
    """注册 MCP 工具清单。"""
    return [
        Tool(
            name="generate_markdown_template",
            description="按场景生成 Markdown 业务模板文件，落地到工作区。",
            inputSchema={
                "type": "object",
                "properties": {
                    "scene": {
                        "type": "string",
                        "description": "模板场景，如 loan_apply、customer_intake。",
                    },
                    "title": {
                        "type": "string",
                        "description": "模板标题，写入文件首行 H1。",
                    },
                },
                "required": ["scene"],
            },
        ),
        Tool(
            name="generate_excel_template",
            description="按列定义生成 Excel 模板文件，首行为列标题。",
            inputSchema={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "列标题数组。",
                    },
                    "filename": {
                        "type": "string",
                        "description": "目标文件名，不带扩展名。",
                    },
                },
                "required": ["columns", "filename"],
            },
        ),
        Tool(
            name="list_generated_templates",
            description="列出工作区已生成的模板文件清单。",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """统一工具分发入口。"""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if name == "generate_markdown_template":
        return [TextContent(type="text", text=_generate_markdown(arguments))]
    if name == "generate_excel_template":
        return [TextContent(type="text", text=_generate_excel(arguments))]
    if name == "list_generated_templates":
        return [TextContent(type="text", text=_list_templates())]
    raise ValueError(f"未知工具：{name}")


def _generate_markdown(arguments: dict[str, Any]) -> str:
    """根据 scene 选择模板，写入文件并返回路径。"""
    scene = str(arguments.get("scene") or "").strip()
    title = str(arguments.get("title") or "").strip()
    if not scene:
        raise ValueError("缺少必填字段：scene")
    body = _MARKDOWN_TEMPLATES.get(scene)
    if body is None:
        raise ValueError(f"未知场景：{scene}")
    if title:
        body = body.replace(body.splitlines()[0], f"# {title}", 1)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = _OUTPUT_DIR / f"{scene}_{timestamp}.md"
    target.write_text(body, encoding="utf-8")
    return f"已生成 Markdown 模板：{target.as_posix()}"


def _generate_excel(arguments: dict[str, Any]) -> str:
    """生成首行带标题的 Excel 模板文件。"""
    columns = arguments.get("columns") or []
    filename = str(arguments.get("filename") or "").strip()
    if not columns:
        raise ValueError("缺少必填字段：columns")
    if not filename:
        raise ValueError("缺少必填字段：filename")
    if not isinstance(columns, list):
        raise TypeError("columns 必须是数组")
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "template"
    sheet.append([str(column) for column in columns])
    target = _OUTPUT_DIR / f"{filename}.xlsx"
    workbook.save(target)
    return f"已生成 Excel 模板：{target.as_posix()}"


def _list_templates() -> str:
    """返回已生成模板文件列表。"""
    if not _OUTPUT_DIR.exists():
        return "尚未生成任何模板。"
    files = sorted(path for path in _OUTPUT_DIR.iterdir() if path.is_file())
    if not files:
        return "尚未生成任何模板。"
    return "\n".join(path.as_posix() for path in files)


async def _main() -> None:
    """stdio 入口，交给 MCP SDK 接管 stdin/stdout。"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
