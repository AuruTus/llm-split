import ast
import inspect
import torch.nn as nn

from typing import TypeVar, Type, Optional, Any, Union

M = TypeVar("M", bound=nn.Module)


def get_module_tree(module_cls: Type[M]) -> ast.Module:
    module_src = inspect.getsource(module_cls)
    ast_tree = ast.parse(module_src)
    return ast_tree


def get_function_node(
    model_nd: ast.ClassDef, function_name: str
) -> tuple[Optional[ast.FunctionDef], int]:
    for i, b in enumerate(model_nd.body):
        if not isinstance(b, ast.FunctionDef):
            continue
        if b.name == function_name:
            return b, i
    return None, -1


def get_init_node(model_nd: ast.ClassDef) -> tuple[Optional[ast.FunctionDef], int]:
    return get_function_node(model_nd, "__init__")


def get_forward_node(model_nd: ast.ClassDef) -> tuple[Optional[ast.FunctionDef], int]:
    return get_function_node(model_nd, "forward")


class _ModuleAST:
    def __init__(self, module_cls: Type[M]):
        self._tree = get_module_tree(module_cls)

    @property
    def model(self) -> ast.stmt:
        return self._tree.body[0]


class ModelAST:
    def __init__(self, module_cls: Type[M]):
        self._module_ast = _ModuleAST(module_cls)
        self._model_nd = self._module_ast.model
        self._forward_nd, _ = get_forward_node(self._model_nd)
        self._init_nd, _ = get_init_node(self._model_nd)

    @property
    def forward_body(self) -> list[ast.stmt]:
        return self._forward_nd.body

    @property
    def init_body(self) -> list[ast.stmt]:
        return self._init_nd.body



class DecoderAST(ModelAST):
    def __init__(self, module_cls: Type[M]):
        super().__init__(module_cls)
        self.attributes = {}

    def _find_for_loop(self) -> tuple[Optional[ast.For], int]:
        for i, b in enumerate(self.forward_body):
            if not isinstance(b, ast.For):
                continue
            return b, i
        return None, -1

    def find_decoder_blocks(self) -> tuple[Optional[ast.For], int]:
        return self._find_for_loop()

    # def
