import ast
import inspect
from dataclasses import dataclass
from typing import TypeVar, Type, Optional, Any, Union, Callable
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple
from contextlib import contextmanager


VAR_TYPE = Union[ast.Name, ast.Attribute, ast.Subscript]


@dataclass
class NodeMeta:
    nd_in: list[str]  # names of nodes on which this node depends
    nd_out: list[str]  # names of nodes to which this node gives dependency
    name: str


@dataclass
class Node:
    meta: NodeMeta
    node: VAR_TYPE

    @staticmethod
    def new(ast_node: VAR_TYPE):
        return Node(NodeMeta(nd_in=[], nd_out=[], name=Node.attr_name(ast_node)), ast_node)

    @staticmethod
    def attr_name(attr: VAR_TYPE):
        match attr:
            case ast.Subscript():
                # TODO
                raise NotImplementedError(f"Node.attr_name: not implemented node type: {type(attr)}")
            case ast.Attribute():
                return f"{Node.attr_name(attr.value)}.{attr.attr}"
            case ast.Name():
                return attr.id
            case _:
                raise NotImplementedError(f"Node.attr_name: not implemented node type: {type(attr)}")


class ASTGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def add_node(self, node: Node):
        self.nodes[node.meta.name] = node

    def add_edge(self, from_node: str, to_node: str):
        self.nodes[from_node].meta.nd_out.append(to_node)
        self.nodes[to_node].meta.nd_in.append(from_node)


class TopoGraphBuilder(ast.NodeVisitor):
    graph: ASTGraph
    current_targets: list[Node]

    def __init__(self):
        self.graph = ASTGraph()
        self.current_targets = []

    def visit_Name(self, node: ast.Name):
        """
        leaf node: should be entered via other statement
        """
        # TODO
        raise NotImplementedError(f"TopoGraphBuilder.visit_Name: not implemented node type: {type(node)}")

    def visit_Attribute(self, node: ast.Attribute):
        """
        leaf node: should be entered via other statement
        """
        # TODO
        raise NotImplementedError(f"TopoGraphBuilder.visit_Attribute: not implemented node type: {type(node)}")

    def visit_Subscript(self, node: ast.Subscript):
        """
        leaf node: should be entered via other statement
        """
        # TODO
        raise NotImplementedError(f"TopoGraphBuilder.visit_Subscript: not implemented node type: {type(node)}")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Don't dive into functions for now (skip local vars)
        pass

    def visit_Assign(self, node: ast.Assign):
        with self._visit_context():
            for target in node.targets:
                self._collect_targets(target)
            for target in self.current_targets:
                if target.meta.name not in self.graph.nodes:
                    self.graph.add_node(target.meta.name, target.node)
                else:
                    target = self.graph.nodes[target.meta.name]
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign):
        with self._visit_context():
            self._collect_targets(node.target)
            for target in self.current_targets:
                if target.meta.name not in self.graph.nodes:
                    self.graph.add_node(target.meta.name, target.node)
                else:
                    target = self.graph.nodes[target.meta.name]
            self.visit(node.value)

    def visit_AnnAssign(self, node):
        raise NotImplementedError("TopoGraphBuilder.visit_AnnAssign: not implemented node type: ast.AnnAssign")

    def visit_For(self, node):
        raise NotImplementedError("TopoGraphBuilder.visit_For: not implemented node type: ast.For")

    def visit_If(self, node: ast.If):
        raise NotImplementedError("TopoGraphBuilder.visit_If: not implemented node type: ast.If")

    def visit_With(self, node):
        raise NotImplementedError("TopoGraphBuilder.visit_With: not implemented node type: ast.With")

    def visit_Return(self, node):
        raise NotImplementedError("TopoGraphBuilder.visit_Return: not implemented node type: ast.Return")

    def _collect_targets(self, node: VAR_TYPE):
        match node:
            case VAR_TYPE():
                self.current_targets.append(Node.new(node))
            case _ as val:
                raise NotImplementedError(f"TopoGraphBuilder._collect_targets: not implemented node type: {type(val)}")

    @contextmanager
    def _visit_context(self):
        """
        used to handle the read-write relationship within a statement
        """
        prev = self.current_targets
        self.current_targets = []
        yield
        self.current_targets = prev


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
