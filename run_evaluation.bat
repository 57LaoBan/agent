@echo off
chcp 65001 >nul
echo ================================================================================
echo 运行 Agent 评测并保存结果
echo ================================================================================
echo.

cd /d "%~dp0"

echo [1/3] 运行 Agent 行为评测...
python -m pytest test/evaluation/test_agent_behavior.py -v --tb=short > .evaluation_results\agent_behavior_output.txt 2>&1

echo [2/3] 运行离线评测（不需要 API）...
python -m pytest test/evaluation/test_agent_behavior.py::AgentOfflineTest -v > .evaluation_results\agent_offline_output.txt 2>&1

echo [3/3] 运行评测指标单元测试...
python -m pytest test/contract/test_evaluation_metrics.py -v > .evaluation_results\metrics_unit_test_output.txt 2>&1

echo.
echo ================================================================================
echo 评测完成！结果已保存到 .evaluation_results\ 目录
echo ================================================================================
echo.
echo 查看结果:
echo   - agent_behavior_output.txt      (Agent 行为评测)
echo   - agent_offline_output.txt       (离线评测)
echo   - metrics_unit_test_output.txt   (指标单元测试)
echo.
pause
