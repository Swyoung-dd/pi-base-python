"""模型目录生成与自定义模型测试。"""

import subprocess
import sys
from pathlib import Path

from pi.ai.models import clear_custom_models, get_model, list_models, load_model_file


def test_generated_model_catalog_is_current():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/generate_models.py", "--check"],
        cwd=root,
        check=False,
    )
    assert result.returncode == 0


def test_deepseek_v4_models_are_available():
    flash = get_model("deepseek-v4-flash", "deepseek")
    pro = get_model("deepseek-v4-pro", "deepseek")

    assert flash is not None
    assert pro is not None
    assert flash.base_url == "https://api.deepseek.com"
    assert flash.reasoning
    assert flash.context_window == 1_000_000
    assert pro.max_tokens == 393_216


def test_custom_model_file_registers_and_overrides_models(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        """models:
  - id: local-code
    name: Local Code
    api: openai-chat-completions
    provider: ollama
    base_url: http://localhost:11434/v1
    context_window: 32768
    max_tokens: 4096
""",
        encoding="utf-8",
    )
    clear_custom_models()
    try:
        loaded = load_model_file(path)
        model = get_model("local-code", "ollama")

        assert loaded == [model]
        assert model.base_url == "http://localhost:11434/v1"
        assert model in list_models()
    finally:
        clear_custom_models()
