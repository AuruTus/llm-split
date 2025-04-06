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


VAR_TYPE = Union[ast.Name, ast.Attribute]


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

    @staticmethod
    def _record_assign(node: ast.AST, assigned: dict[str, Any]):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned[t.id] = t
                elif isinstance(t, ast.Attribute):
                    assigned[DecoderAST._get_attribute_name(t)] = t
            return
        if hasattr(node, "body"):
            for b in node.body:
                DecoderAST._record_assign(b, assigned)

    @staticmethod
    def _collect_assigns(node_body: list[ast.stmt]):
        record = {}
        for b in node_body:
            DecoderAST._record_assign(b, record)
        return record

    @staticmethod
    def _get_iter_target(iter_target: Union[ast.Name, ast.Tuple]) -> dict[str, ast.Name]:
        res = {}
        if isinstance(iter_target, ast.Name):
            res[iter_target.id] = iter_target
        elif isinstance(iter_target, ast.Tuple):
            for e in iter_target.elts:
                res[e.id] = e
        else:
            raise NotImplementedError(f"_get_iter_target: not implemented iter_target type: {type(iter_target)}")
        return res

    @staticmethod
    def _get_attribute_name(attr: Union[ast.Attribute, ast.Name]) -> str:
        if isinstance(attr, ast.Attribute):
            return f"{attr.attr}.{DecoderAST._get_attribute_name(attr.value)}"
        elif isinstance(attr, ast.Name):
            return attr.id
        else:
            raise NotImplementedError(f"_get_attribute_name: not implemented attr type: {type(attr)}")

    @staticmethod
    def _get_iter_source(iter_source: Union[ast.Subscript, ast.Call]) -> dict[str, ast.Name]:
        res = {}
        if isinstance(iter_source, ast.Subscript):
            res[DecoderAST._get_attribute_name(iter_source.value)] = iter_source.value
            if iter_source.slice.upper and (upper_name := DecoderAST._get_attribute_name(iter_source.slice.upper)):
                res[upper_name] = iter_source.slice
            if iter_source.slice.lower and (lower_name := DecoderAST._get_attribute_name(iter_source.slice.lower)):
                res[lower_name] = iter_source.slice
            if iter_source.slice.step and (step_name := DecoderAST._get_attribute_name(iter_source.slice.step)):
                res[step_name] = iter_source.slice
        elif isinstance(iter_source, ast.Name):
            res[iter_source.id] = iter_source
        elif isinstance(iter_source, ast.Call):
            raise NotImplementedError(f"_get_iter_source: not implemented iter_source type: {type(iter_source)}")
            if isinstance(iter_source.func, ast.Attribute):
                res[DecoderAST._get_attribute_name(iter_source.func)] = iter_source.func
            for arg in iter_source.args:
                pass
            for kwarg in iter_source.keywords:
                pass
        else:
            raise NotImplementedError(f"_get_iter_source: not implemented iter_source type: {type(iter_source)}")
        return res

    @staticmethod
    def _get_var_info(node: VAR_TYPE) -> str:
        if isinstance(node, ast.Name):
            return node.id, node
        elif isinstance(node, ast.Attribute):
            return DecoderAST._get_attribute_name(node)

    @staticmethod
    def get_read_write_vars(node: ast.AST) -> tuple[dict[str, VAR_TYPE], dict[str, VAR_TYPE]]:
        read: dict[str, VAR_TYPE] = {}
        write: dict[str, VAR_TYPE] = {}

        def _get_read_write_vars(node: ast.AST, read: dict[str, VAR_TYPE], write: dict[str, VAR_TYPE]):
            if isinstance(node, VAR_TYPE):
                pass

        _get_read_write_vars(node, read, write)
        return read, write

    def get_model_attribute(self) -> dict[str, ast.Name]:
        # NOTE: it only records the attributes initialized in __init__ for now
        return self._collect_assigns(self.init_body)

    def get_forward_vars_in_range(self, range: range) -> dict[str, ast.Name]:
        beg, end, step = range.start, range.stop, range.step
        body = self.forward_body[beg:end:step]
        return self._collect_assigns(body)

    def get_weak_topo_order(self):
        model_attrs = self.get_model_attribute()
        decoder_for_block, decoder_blocks_idx = self.find_decoder_blocks()
        decoder_vars = self._collect_assigns(decoder_for_block.body)
        before_decoder_blocks_vars = self.get_forward_vars_in_range(range(0, decoder_blocks_idx))
        after_decoder_blocks_vars = self.get_forward_vars_in_range(range(decoder_blocks_idx + 1, len(self.forward_body)))
