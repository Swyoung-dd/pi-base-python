"""从 models.catalog.json 生成 Python 模型目录。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from pprint import pformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CATALOG_PATH = SRC / "pi" / "ai" / "models.catalog.json"
OUTPUT_PATH = SRC / "pi" / "ai" / "models_generated.py"
sys.path.insert(0, str(SRC))

from pi.ai.types import Model  # noqa: E402


def render_catalog() -> str:
    """校验目录并渲染确定性的 Python 源码。"""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1:
        raise ValueError("不支持的模型目录版本")
    models = [Model.model_validate(item) for item in catalog.get("models", [])]
    keys = [(model.provider, model.id) for model in models]
    if len(keys) != len(set(keys)):
        raise ValueError("模型目录包含重复的 provider/id")

    records = [model.model_dump(mode="json", exclude_none=True) for model in models]
    body = pformat(records, width=100, sort_dicts=False)
    return (
        '"""由 scripts/generate_models.py 生成，请勿手工修改。"""\n\n'
        "from pi.ai.types import Model\n\n"
        f"MODELS = [Model.model_validate(item) for item in {body}]\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="检查生成文件是否最新")
    args = parser.parse_args()
    rendered = render_catalog()
    if args.check:
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
            print(f"模型目录需要重新生成: {OUTPUT_PATH}", file=sys.stderr)
            return 1
        return 0
    OUTPUT_PATH.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
