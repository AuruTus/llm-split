import ast
import inspect
import torch.nn as nn

from typing import TypeVar, Type, Optional

M = TypeVar("M", bound=nn.Module)


def get_module_tree(module_cls: Type[M]) -> ast.Module:
    module_src = inspect.getsource(module_cls)
    ast_tree = ast.parse(module_src)
    return ast_tree


def get_forward_node(model_ast: ast.ClassDef) -> tuple[Optional[ast.FunctionDef], int]:
    for i, b in enumerate(model_ast.body):
        if not isinstance(b, ast.FunctionDef):
            continue
        if b.name == "forward":
            return b, i
    return None, -1


class ModelAST:
    def __init__(self, module_cls: Type[M]):
        self.model = get_module_tree(module_cls)
        self.forward, forward_idx = get_forward_node(self.model)




