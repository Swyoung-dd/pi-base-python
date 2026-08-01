"""不会持久化疑似凭据的 TUI 输入历史。"""

from prompt_toolkit.history import FileHistory, InMemoryHistory

from pi.coding_agent.model_auth import contains_likely_api_key


class SafeFileHistory(FileHistory):
    """过滤疑似 API Key，避免凭据落入磁盘输入历史。"""

    def store_string(self, string: str) -> None:
        if not contains_likely_api_key(string):
            super().store_string(string)


class SafeInMemoryHistory(InMemoryHistory):
    """过滤疑似 API Key，避免凭据保留在当前进程输入历史。"""

    def store_string(self, string: str) -> None:
        if not contains_likely_api_key(string):
            super().store_string(string)

