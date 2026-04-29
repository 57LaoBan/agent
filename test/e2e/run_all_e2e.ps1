$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $root

try {
    $env:PYTHONPATH = "src"

    Write-Host "[e2e] 运行后端端到端测试"
    python .\test\e2e\run_backend_e2e.py --start-server

    Write-Host "[e2e] 运行前端浏览器端到端测试"
    node .\test\e2e\run_frontend_e2e.mjs --start-backend --start-frontend
}
finally {
    Pop-Location
}
