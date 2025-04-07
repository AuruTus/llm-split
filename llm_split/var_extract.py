import ast


# Define a visitor class to collect read and assigned variables
class VariableVisitor(ast.NodeVisitor):
    def __init__(self):
        self.read_vars = set()
        self.assigned_vars = set()
        self.current_scope = set()
        self.read_before_assigned = set()

    def visit_FunctionDef(self, node):
        # Save the current scope and create a new scope for the function
        saved_scope = self.current_scope
        self.current_scope = set()
        self.generic_visit(node)
        # Restore the previous scope
        self.current_scope = saved_scope

    def visit_ClassDef(self, node):
        # Save the current scope and create a new scope for the class
        saved_scope = self.current_scope
        self.current_scope = set()
        self.generic_visit(node)
        # Restore the previous scope
        self.current_scope = saved_scope

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id in self.current_scope:
                return
            self.read_vars.add(node.id)
            if node.id not in self.assigned_vars:
                self.read_before_assigned.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.assigned_vars.add(node.id)
            self.current_scope.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.attr in self.current_scope:
                return
            self.read_vars.add(node.attr)
            if node.attr not in self.assigned_vars:
                self.read_before_assigned.add(node.attr)
        elif isinstance(node.ctx, ast.Store):
            self.assigned_vars.add(node.attr)
            self.current_scope.add(node.attr)
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Visit the value (right-hand side) first to handle reads
        self.visit(node.value)
        # Then visit the targets (left-hand side) to handle writes
        for target in node.targets:
            self.visit(target)

    def visit_AugAssign(self, node):
        # Visit the value (right-hand side) first to handle reads
        self.visit(node.value)
        # Then visit the target (left-hand side) to handle writes
        self.visit(node.target)

    def visit_Global(self, node):
        for name in node.names:
            self.current_scope.add(name)

    def visit_Nonlocal(self, node):
        for name in node.names:
            self.current_scope.add(name)
