import ast
import inspect
import torch.nn as nn

from typing import TypeVar, Type, Optional, Any

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

    def find_for_loop(self) -> tuple[Optional[ast.For], int]:
        for i, b in enumerate(self.forward_body):
            if not isinstance(b, ast.For):
                continue
            return b, i
        return None, -1

    @staticmethod
    def _record_assign(node: ast.AST, assigned: dict[str, Any]):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                assigned[t.id] = t
            return
        if isinstance(node, ast.Attribute):
            # TODO: record self attribute
            return
        if hasattr(node, "body"):
            for b in node.body:
                DecoderAST._record_assign(b, assigned)

    def get_model_attribute(self) -> dict[str, ast.Name]:
        # NOTE: it only records the attributes initialized in __init__ for now
        record = {}
        for b in self.init_body:
            DecoderAST._record_assign(b, record)
        return record

    def get_vars_in_range(self, range: range) -> dict[str, ast.Name]:
        record = {}
        beg, end, step = range.start, range.stop, range.step
        body = self.forward_body[beg:end:step]
        for b in body:
            DecoderAST._record_assign(b, record)
        return record
