"""包级模块入口测试。"""

import os
import subprocess
import sys
from pathlib import Path

from pi import __version__


def test_python_module_entrypoint_reports_version():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [sys.executable, "-m", "pi", "--version"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"pi {__version__}"
