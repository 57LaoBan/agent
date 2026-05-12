# Template MCP Server

通用业务模板生成 MCP Server，对外暴露三个工具：

- `generate_markdown_template(scene, title?)`
- `generate_excel_template(columns, filename)`
- `list_generated_templates()`

## 运行

```bash
python -m xinyidai_mcp_servers.template_server.server
```

## 产物目录

默认 `./.data/mcp_templates/`；可通过环境变量 `TEMPLATE_OUTPUT_DIR` 覆盖。
