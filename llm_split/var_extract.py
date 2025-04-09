import ast
from abc import ABC, abstractmethod
from functools import partial
from typing import Union, Type, Callable
from llm_split.namespace import NameSpace

VAR_TYPE = Union[ast.Name, ast.Attribute]


def var_name(attr: VAR_TYPE) -> str:
    if isinstance(attr, ast.Attribute):
        return f"{var_name(attr.value)}.{attr.attr}"
    elif isinstance(attr, ast.Name):
        return attr.id
    else:
        raise NotImplementedError(f"var_name: not implemented attr type: {type(attr)}")


# Define a visitor class to collect read and assigned variables
class VariableVisitor(ast.NodeVisitor):
    def __init__(self):
        self.read_vars: set[str] = set()
        self.assigned_vars: set[str] = set()
        self.current_scope: set[str] = set()
        self.read_before_assigned: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Save the current scope and create a new scope for the function
        saved_scope = self.current_scope
        self.current_scope = set()
        self.generic_visit(node)
        # Restore the previous scope
        self.current_scope = saved_scope

    def visit_ClassDef(self, node: ast.ClassDef):
        # Save the current scope and create a new scope for the class
        saved_scope = self.current_scope
        self.current_scope = set()
        self.generic_visit(node)
        # Restore the previous scope
        self.current_scope = saved_scope

    def visit_Name(self, node: ast.Name):
        node_name = var_name(node)
        match node.ctx:
            case ast.Load():
                if node_name in self.current_scope:
                    return
                self.read_vars.add(node_name)
                if node_name not in self.assigned_vars:
                    self.read_before_assigned.add(node_name)
            case ast.Store():
                self.assigned_vars.add(node_name)
                self.current_scope.add(node_name)

    def visit_Attribute(self, node: ast.Attribute):
        node_name = var_name(node)
        name = node_name.split(".")
        if node_name.startswith("self."):
            node_name = ".".join(name[:2])
        else:
            node_name = name[0]
        match node.ctx:
            case ast.Load():
                if node_name in self.current_scope:
                    return
                self.read_vars.add(node_name)
                if node_name not in self.assigned_vars:
                    self.read_before_assigned.add(node_name)
            case ast.Store():
                self.assigned_vars.add(node_name)
                self.current_scope.add(node_name)

    def visit_Assign(self, node: ast.Assign):
        # Visit the value (right-hand side) first to handle reads
        self.visit(node.value)
        # Then visit the targets (left-hand side) to handle writes
        for target in node.targets:
            self.visit(target)

    def visit_AugAssign(self, node: ast.AugAssign):
        # Visit the value (right-hand side) first to handle reads
        node.target.ctx = ast.Load()  # Temporarily change context to Load
        self.visit(node.value)
        self.visit(node.target)
        node.target.ctx = ast.Store()
        # Then visit the target (left-hand side) to handle writes
        self.visit(node.target)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        # Visit all comprehensions first to process assignment targets
        for gen in node.generators:
            self.visit(gen)
        # Visit the element expression
        self.visit(node.elt)

    def visit_comprehension(self, node: ast.comprehension):
        # Visit the target in Store context (crucial for assignment tracking)
        if isinstance(node.target, ast.Name):
            # Create a pseudo Assign node to trigger assignment tracking
            mock_assign = ast.Assign(targets=[ast.Name(id=node.target.id, ctx=ast.Store())], value=ast.Constant(value=None))
            self.visit_Assign(mock_assign)
        else:
            # Handle tuple unpacking etc.
            self.visit(node.target)

        # Visit iterable and conditions
        self.visit(node.iter)
        for cond in node.ifs:
            self.visit(cond)

    def visit_DictComp(self, node: ast.DictComp):
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp):
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp):
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr):
        self.visit(node.value)  # Read first (RHS)
        self.visit(node.target)  # Write after (LHS)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value:  # Handle cases like `x: int = 5`
            self.visit(node.value)  # Read RHS first
        self.visit(node.target)  # Write LHS after

    def visit_Global(self, node: ast.Global):
        for name in node.names:
            self.current_scope.add(name)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        for name in node.names:
            self.current_scope.add(name)


class FilterBase(ABC):
    @abstractmethod
    def _filter(self):
        pass

    @abstractmethod
    def res(self) -> set[str]:
        pass

    @property
    @abstractmethod
    def var_list(self) -> list[str]:
        pass


class SelfAttrFilter(FilterBase):
    def __init__(self, func_vars: set[str]):
        self._func_vars = func_vars
        self._self_attr: set[str] = set()
        self._var: set[str] = set()

        self._filter()

    def _filter(self):
        for var in self._func_vars:
            if var.startswith("self."):
                self._self_attr.add(var)
            else:
                self._var.add(var)

    def res(self) -> set[str]:
        return self._var

    @property
    def self_attr_list(self) -> list[str]:
        return sorted(self._self_attr)

    @property
    def var_list(self) -> list[str]:
        return sorted(self._var)


class GlobalVarFilter(FilterBase):
    def __init__(self, func_vars: set[str], namespace: NameSpace):
        self._namespace = namespace
        self._func_vars = func_vars
        self._global_var: set[str] = set()
        self._var: set[str] = set()

        self._filter()

    def _filter(self):
        for var in self._func_vars:
            if hasattr(self._namespace, var):
                self._global_var.add(var)
                continue
            self._var.add(var)

    def res(self) -> set[str]:
        return self._var

    @property
    def global_var_list(self) -> set[str]:
        return sorted(self._global_var)

    @property
    def var_list(self) -> list[str]:
        return sorted(self._var)


class BasicTypeFilter(FilterBase):
    def __init__(self, func_vars: set[str]):
        self._func_vars = func_vars
        self._basic_type: set[str] = set()
        self._var: set[str] = set()

        self._filter()

    def _filter(self):
        import builtins

        builtin_namespace = dir(builtins)
        for var in self._func_vars:
            if var in builtin_namespace:
                self._basic_type.add(var)
                continue
            self._var.add(var)

    def res(self) -> set[str]:
        return self._var

    @property
    def var_list(self) -> list[str]:
        return sorted(self._var)

    @property
    def basic_type_list(self) -> list[str]:
        return sorted(self._basic_type)


class filter:
    """
    It works as a function to combine fileters walking through a set of var name strings.

    usage example:
    ```
        filter(variables set).by([some filter classes...])
    ```
    """

    _FILTER_CLASS = Union[type[FilterBase], Callable[[set[str]], None]]

    def __init__(self, vars: set[str]):
        self._init_vars = vars

    def by(self, filters: list[_FILTER_CLASS]) -> list[str]:
        var = self._init_vars
        for f in filters[:-1]:
            var = f(var).res()
        return filters[-1](var).var_list


def extract_input_args(visitor: VariableVisitor, src_class: type) -> list[str]:
    return filter(visitor.read_before_assigned).by(
        [
            SelfAttrFilter,
            partial(GlobalVarFilter, namespace=NameSpace.namespace_of(src_class)),
            BasicTypeFilter,
        ],
    )
