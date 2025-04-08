import sys
import ast
from typing import Any


class NameSpace:
    @staticmethod
    def namespace_of(class_typ: type) -> "NameSpace":
        return NameSpace(
            namespace=vars(sys.modules[class_typ.__module__]),
            filename=f"<{class_typ.__name__}_ast_modified>",
            mode="exec",
        )

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
            raise AttributeError(f"{name} not found in namespace")
        return self._namespace[name]

    def compile_ast(self, node: ast.Module):
        code = compile(node, self._filename, self._mode)
        exec(code, self._namespace)
