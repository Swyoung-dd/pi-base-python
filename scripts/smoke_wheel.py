"""在隔离虚拟环境中安装并验证构建出的 piY wheel。"""

from __future__ import annotations

import os
import subprocess
import tempfile
import venv
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("piy-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one wheel in dist, found {len(wheels)}")

    with tempfile.TemporaryDirectory(prefix="piy-wheel-") as temp_dir:
        environment_dir = Path(temp_dir) / "venv"
        venv.EnvBuilder(with_pip=True).create(environment_dir)
        scripts_dir = environment_dir / ("Scripts" if os.name == "nt" else "bin")
        python = scripts_dir / ("python.exe" if os.name == "nt" else "python")
        piy = scripts_dir / ("piY.exe" if os.name == "nt" else "piY")

        subprocess.run(
            [str(python), "-m", "pip", "install", str(wheels[0])],
            check=True,
        )
        for arguments in (["--version"], ["--help"], ["--list-models"]):
            subprocess.run([str(piy), *arguments], check=True)


if __name__ == "__main__":
    main()
