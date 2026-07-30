"""包级模块入口测试。"""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

from pi import __version__
from pi.coding_agent import CodingAgent, create_coding_agent


def test_distribution_and_command_use_piy_name():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["name"] == "piY"
    assert metadata["project"]["scripts"] == {"piY": "pi.coding_agent.cli:main"}
    assert CodingAgent is not None
    assert callable(create_coding_agent)


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
    assert result.stdout.strip() == f"piY {__version__}"
