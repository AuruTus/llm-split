from typing import List, Optional
import ast

from llm_split.var_extract import VariableVisitor

class ForwardMeta:
    def __init__(self):
        self.input_vars: set[str] = set()
        self.stmts: ast.Module = ast.Module(body=[], type_ignores=[])
        self.returned_vars: List[str] = []

    def add_input_vars(self, vars: set[str]) -> None:
        """Add input variables with dependency tracking"""
        for var in vars:
            var = var.strip()
            if "." not in var:
                self.input_vars.add(var)

    def add_statements(self, module: ast.Module) -> None:
        """Preserve original AST structure"""
        self.stmts.body.extend(module.body)

    def calculate_returns(self, code1_visitor: VariableVisitor, code2_visitor: VariableVisitor) -> None:
        """Calculate return vars using cross-segment analysis"""
        code1_outputs = code1_visitor.assigned_vars | code1_visitor.read_vars
        code2_inputs = code2_visitor.read_before_assigned
        self.returned_vars = sorted(code2_inputs & code1_outputs)

    @property
    def arg_names(self) -> List[ast.arg]:
        """Include self as first argument"""
        return [ast.arg(arg="self")] + [ast.arg(arg=var) for var in sorted(self.input_vars)]

    def compile_forward(self) -> ast.FunctionDef:
        """Generate complete method AST"""
        return ast.FunctionDef(
            name="forward",
            args=ast.arguments(posonlyargs=[], args=self.arg_names, kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[*self.stmts.body, ast.Return(value=ast.Tuple(elts=[ast.Name(id=v, ctx=ast.Load()) for v in self.returned_vars], ctx=ast.Load()))],
            decorator_list=[],
        )
