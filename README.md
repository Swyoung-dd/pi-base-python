# piY

piY 是一个 Python 实现的终端 AI 编码 agent，提供多 LLM provider、流式响应、工具调用、会话恢复、上下文压缩、Skills、扩展和交互式终端。

## 安装

```bash
python -m pip install -e ".[dev]"
```

Python 版本要求为 3.12 或更高。

## 配置

通过环境变量提供凭据，例如 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`，然后运行：

```bash
piY --setup
piY
```

项目配置位于 `.piy/config.yaml`。自定义模型可写入 `.piy/models.yaml`。

使用 DeepSeek V4：

```powershell
$env:DEEPSEEK_API_KEY="你的 API Key"
piY --setup
```

在模型列表中选择 `deepseek/deepseek-v4-flash` 或 `deepseek/deepseek-v4-pro`。

## 命令

```bash
piY --help
piY --list-models
piY -p "Summarize @README.md" --output json
python -m pi --version
```

## 开发检查

```bash
python -m ruff check src tests scripts
python scripts/generate_models.py --check
python -m pytest -q
python -m build
```
