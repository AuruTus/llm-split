import ast
import inspect
from dataclasses import dataclass
from typing import TypeVar, Type, Optional, Any, Union, Callable
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple


def _find_for_loop(self) -> tuple[Optional[ast.For], int]:
    for i, b in enumerate(self.forward_body):
        if not isinstance(b, ast.For):
            continue
        return b, i
    return None, -1


def find_decoder_blocks(self) -> tuple[Optional[ast.For], int]:
    return self._find_for_loop()


def _record_assign(node: ast.AST, assigned: dict[str, Any]):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                assigned[t.id] = t
            elif isinstance(t, ast.Attribute):
                assigned[_get_attribute_name(t)] = t
        return
    if hasattr(node, "body"):
        for b in node.body:
            _record_assign(b, assigned)


def _collect_assigns(node_body: list[ast.stmt]):
    record = {}
    for b in node_body:
        _record_assign(b, record)
    return record


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


def _get_attribute_name(attr: Union[ast.Attribute, ast.Name]) -> str:
    if isinstance(attr, ast.Attribute):
        return f"{attr.attr}.{_get_attribute_name(attr.value)}"
    elif isinstance(attr, ast.Name):
        return attr.id
    else:
        raise NotImplementedError(f"_get_attribute_name: not implemented attr type: {type(attr)}")


def _get_iter_source(iter_source: Union[ast.Subscript, ast.Call]) -> dict[str, ast.Name]:
    res = {}
    if isinstance(iter_source, ast.Subscript):
        res[_get_attribute_name(iter_source.value)] = iter_source.value
        if iter_source.slice.upper and (upper_name := _get_attribute_name(iter_source.slice.upper)):
            res[upper_name] = iter_source.slice
        if iter_source.slice.lower and (lower_name := _get_attribute_name(iter_source.slice.lower)):
            res[lower_name] = iter_source.slice
        if iter_source.slice.step and (step_name := _get_attribute_name(iter_source.slice.step)):
            res[step_name] = iter_source.slice
    elif isinstance(iter_source, ast.Name):
        res[iter_source.id] = iter_source
    elif isinstance(iter_source, ast.Call):
        raise NotImplementedError(f"_get_iter_source: not implemented iter_source type: {type(iter_source)}")
        if isinstance(iter_source.func, ast.Attribute):
            res[_get_attribute_name(iter_source.func)] = iter_source.func
        for arg in iter_source.args:
            pass
        for kwarg in iter_source.keywords:
            pass
    else:
        raise NotImplementedError(f"_get_iter_source: not implemented iter_source type: {type(iter_source)}")
    return res


VAR_TYPE = Union[ast.Name, ast.Attribute]
CTX_TYPE = Union[ast.Load, ast.Store, ast.Del]


@dataclass
class VarMeta:
    name: str
    ctx: CTX_TYPE


def _get_attribute_meta(attr: VAR_TYPE) -> VarMeta:
    if isinstance(attr, ast.Attribute):
        meta = _get_attribute_meta(attr.value)
        meta.name = f"{meta.name}.{attr.attr}"
        return meta
    elif isinstance(attr, ast.Name):
        return VarMeta(attr.id, attr.ctx)
    else:
        raise NotImplementedError(f"_get_attribute_meta: not implemented attr type: {type(attr)}")


class CallVisitor(ast.NodeVisitor):
    def __init__(self, record: dict[str, ast.Call]):
        self._record = record

    def visit_Call(self, node: ast.Call):
        match node.func:
            case ast.Name():
                pass
            case ast.Attribute():
                pass
            case _ as val:
                raise NotImplementedError(f"CallVisitor.visit_Call: not implemented func type: {type(val)}")
        return self.generic_visit(node)


class VarVisitor(ast.NodeVisitor):
    def __init__(self, read: dict[str, VAR_TYPE], write: dict[str, VAR_TYPE]):
        self._read = read
        self._write = write

    def _record_node(self, node_meta: VarMeta, node: VAR_TYPE):
        match node_meta.ctx:
            case ast.Load():
                self._read[node_meta.name] = node
            case ast.Store():
                self._write[node_meta.name] = node
            case _ as val:
                raise NotImplementedError(f"VarVisitor._record_node: not implemented var type: {type(val)}")

    def visit_Attribute(self, node: ast.Attribute):
        var_meta = _get_attribute_meta(node)
        self._record_node(var_meta, node)

    def visit_Name(self, node: ast.Name):
        var_meta = VarMeta(node.id, node.ctx)
        self._record_node(var_meta, node)


def get_read_write_vars(node: ast.AST) -> tuple[dict[str, VAR_TYPE], dict[str, VAR_TYPE]]:
    read: dict[str, VAR_TYPE] = {}
    write: dict[str, VAR_TYPE] = {}

    visitor = VarVisitor(read, write)
    visitor.visit(node)

    return read, write


@dataclass
class Node:
    nd_in: list[str]
    nd_out: list[str]
    name: str


class EnhancedDependencyGraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.current_targets: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Don't dive into functions for now (skip local vars)
        pass

    def visit_Assign(self, node: ast.Assign):
        self.current_targets = []
        for target in node.targets:
            self._collect_targets(target)
        for target in self.current_targets:
            self.graph[target]  # ensure target is in graph
        self.visit(node.value)
        self.current_targets = []

    def visit_AugAssign(self, node: ast.AugAssign):
        self.current_targets = []
        self._collect_targets(node.target)
        for target in self.current_targets:
            self.graph[target]  # ensure target is in graph
        self.visit(node.value)
        self.current_targets = []

    def visit_For(self, node: ast.For):
        self._collect_targets(node.target)
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node: ast.If):
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def _collect_targets(self, node: ast.AST):
        if isinstance(node, ast.Name):
            self.current_targets.append(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._collect_targets(elt)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            for target in self.current_targets:
                self.graph[target].add(node.id)


def topological_sort(graph: Dict[str, Set[str]]) -> List[str]:
    indegree = defaultdict(int)
    for node in graph:
        for dep in graph[node]:
            indegree[dep] += 1
        if node not in indegree:
            indegree[node] = 0

    queue = deque([n for n in indegree if indegree[n] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for other in graph:
            if node in graph[other]:
                graph[other].remove(node)
                indegree[other] -= 1
                if indegree[other] == 0:
                    queue.append(other)

    return order


def get_variable_order(code: str) -> Tuple[List[str], Dict[str, Set[str]]]:
    tree = ast.parse(code)
    builder = EnhancedDependencyGraphBuilder()
    builder.visit(tree)
    sorted_vars = topological_sort(builder.graph.copy())
    return sorted_vars, builder.graph
