"""运行完整评测并保存结果。"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_test(test_path: str, output_name: str) -> tuple[int, str]:
    """运行单个测试并返回结果。"""
    print(f"\n正在运行: {test_path}")
    print("-" * 80)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding='utf-8',
    )

    # 打印输出
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode, result.stdout + "\n\n" + result.stderr


def main():
    """主函数。"""
    print("=" * 80)
    print("Agent 评测系统")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建结果目录
    results_dir = ROOT / ".evaluation_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 测试列表
    tests = [
        {
            "name": "Agent 行为评测",
            "path": "test/evaluation/test_agent_behavior.py",
            "output": f"agent_behavior_{timestamp}.txt",
            "description": "评测路由准确率、工具正确性、反幻觉能力",
        },
        {
            "name": "离线评测",
            "path": "test/evaluation/test_agent_behavior.py::AgentOfflineTest",
            "output": f"agent_offline_{timestamp}.txt",
            "description": "数据集完整性检查（不需要 API）",
        },
        {
            "name": "指标单元测试",
            "path": "test/contract/test_evaluation_metrics.py",
            "output": f"metrics_unit_{timestamp}.txt",
            "description": "检索和答案指标计算逻辑测试",
        },
    ]

    results = []

    # 运行所有测试
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {test['name']}")
        print(f"说明: {test['description']}")

        exit_code, output = run_test(test["path"], test["output"])

        # 保存输出
        output_file = results_dir / test["output"]
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"{test['name']}\n")
            f.write("=" * 80 + "\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试路径: {test['path']}\n")
            f.write(f"说明: {test['description']}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write(output)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit Code: {exit_code}\n")
            f.write(f"状态: {'通过' if exit_code == 0 else '失败'}\n")
            f.write("=" * 80 + "\n")

        results.append({
            "name": test["name"],
            "path": test["path"],
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "output_file": str(output_file),
        })

        print(f"✓ 结果已保存到: {output_file}")

    # 生成汇总报告
    summary_file = results_dir / f"summary_{timestamp}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("评测结果汇总\n")
        f.write("=" * 80 + "\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总测试数: {len(results)}\n")
        f.write(f"通过: {sum(1 for r in results if r['passed'])}\n")
        f.write(f"失败: {sum(1 for r in results if not r['passed'])}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("详细结果:\n")
        f.write("=" * 80 + "\n")
        for r in results:
            status = "✓ 通过" if r["passed"] else "✗ 失败"
            f.write(f"\n{status} - {r['name']}\n")
            f.write(f"  路径: {r['path']}\n")
            f.write(f"  输出: {r['output_file']}\n")

    print("\n" + "=" * 80)
    print("评测完成！")
    print("=" * 80)
    print(f"\n汇总报告: {summary_file}")
    print(f"结果目录: {results_dir}")
    print("\n测试结果:")
    for r in results:
        status = "✓ 通过" if r["passed"] else "✗ 失败"
        print(f"  {status} - {r['name']}")

    # 返回失败的测试数
    failed_count = sum(1 for r in results if not r["passed"])
    return failed_count


if __name__ == "__main__":
    try:
        failed = main()
        sys.exit(failed)
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
