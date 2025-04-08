import ast
from typing import Union


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
