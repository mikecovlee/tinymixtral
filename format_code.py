"""跨平台代码格式化与静态检查入口。"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, "-m", "black", "."], check=True)
    subprocess.run([sys.executable, "-m", "isort", ".", "--profile", "black"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "flake8",
            ".",
            "--show-source",
            "--statistics",
            "--max-line-length=128",
            "--max-complexity",
            "99",
            "--ignore=E203,W503,E722",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
