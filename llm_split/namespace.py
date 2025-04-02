import ast
from typing import Any


class Namespace:
    def __init__(
        self,
        namespace: dict[str, Any],
        filename: str = "<ast_modified>",
        mode: str = "exec",
    ):
        self._namespace = {} | namespace
        self._filename = filename
        self._mode = mode

    def __getattr__(self, name: str) -> Any:
        if name not in self._namespace:
            return None
        return self._namespace[name]

    def compile_ast(self, node: ast.Module):
        code = compile(node, self._filename, self._mode)
        exec(code, self._namespace)
