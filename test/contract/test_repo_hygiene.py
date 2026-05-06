from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class RepoHygieneTest(unittest.TestCase):
    def test_gitattributes_declares_reviewable_line_endings(self) -> None:
        path = ROOT / ".gitattributes"
        self.assertTrue(path.exists(), ".gitattributes 必须存在以固定行尾策略")

        content = path.read_text(encoding="utf-8")
        required_rules = [
            "*.py text eol=lf",
            "*.md text eol=lf",
            "*.json text eol=lf",
            "*.toml text eol=lf",
            "*.ts text eol=lf",
            "*.tsx text eol=lf",
            "*.ps1 text eol=crlf",
            "*.bat text eol=crlf",
        ]
        for rule in required_rules:
            self.assertIn(rule, content)


if __name__ == "__main__":
    unittest.main()
