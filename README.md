# piY

piY 是一个 Python 实现的终端 AI 编码 agent，提供多 LLM provider、流式响应、工具调用、会话恢复、上下文压缩、Skills、扩展和交互式终端。

## 环境要求

piY 要求 Python 3.12 或更高版本。推荐使用 uv 和仓库内的 `uv.lock` 创建可复现环境：

```bash
python -m pip install uv==0.12.0
uv sync --locked --all-extras
```

不需要开发依赖时，也可以直接使用 pip 安装：

```bash
python -m pip install .
```

## 运行

使用 uv 时，命令不依赖系统 `PATH`：

```bash
uv run piY --setup
uv run piY
```

使用已激活的虚拟环境或全局安装时：

```bash
piY --setup
piY
```

如果 Windows PowerShell 提示无法识别 `piY`，说明安装脚本目录不在当前 `PATH`。可以激活项目虚拟环境，或直接通过模块入口运行：

```powershell
.\.venv\Scripts\Activate.ps1
piY --setup

# 未激活虚拟环境时
.\.venv\Scripts\python.exe -m pi --setup
.\.venv\Scripts\python.exe -m pi
```

## 配置

项目配置位于 `.piy/config.yaml`。自定义模型可写入 `.piy/models.yaml`。

使用 DeepSeek V4：

```powershell
$env:DEEPSEEK_API_KEY="你的 API Key"
piY --setup
```

在模型列表中选择 `deepseek/deepseek-v4-flash` 或 `deepseek/deepseek-v4-pro`。
也可以进入交互模式后执行 `/model`，选择模型并按提示输入或替换 API Key；已有凭据时直接回车即可保留。凭据保存在用户目录的 `.piy/auth.json`，输入时只显示掩码。主输入框会阻止疑似 API Key 被发送给模型或写入新的输入历史。

## 命令

```bash
piY --help
piY --list-models
piY -p "Summarize @README.md" --output json
python -m pi --version
```

## 打包

版本号位于 `src/pi/__init__.py`。更新版本号后执行：

```bash
uv run --locked python -m build
uv run --locked python scripts/smoke_wheel.py
```

构建产物位于 `dist/`：

- `piy-<version>-py3-none-any.whl`：用于 pip 安装。
- `piy-<version>.tar.gz`：源码分发包。

在新的 Windows 虚拟环境中安装并运行 wheel：

```powershell
py -m venv .venv-run
.\.venv-run\Scripts\python.exe -m pip install .\dist\piy-0.1.0-py3-none-any.whl
.\.venv-run\Scripts\piY.exe --setup
.\.venv-run\Scripts\piY.exe
```

文件名中的版本应替换为 `src/pi/__init__.py` 中的当前版本。`scripts/smoke_wheel.py` 会在临时虚拟环境中验证版本、帮助和模型列表入口。

## 开发检查

```bash
uv run --locked ruff check src tests scripts
uv run --locked python scripts/generate_models.py --check
uv run --locked python -m pytest -q
uv run --locked python -m build
uv run --locked python scripts/smoke_wheel.py
```
