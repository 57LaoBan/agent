"""评测模块 pytest 配置。

评测测试默认不在 CI 中运行（需要 LLM API key + RAG 服务）。
手动运行：pytest test/evaluation/ -v
"""

import os
import sys
from pathlib import Path

# 确保 src 在路径中
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
