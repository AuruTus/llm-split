from typing import List, Optional
import ast


class ForwardMeta:
    def __init__(self):
        self.input_vars: List[str] = []
        self.stmts: ast.Module = ast.Module(body=[], type_ignores=[])
        self.returned_vars: List[str] = []

    def add_input_vars(self, vars: List[str]) -> None:
        """Add input variables while preserving order and removing duplicates"""
        seen = set(self.input_vars)
        for var in vars:
            var = var.strip()
            if not var.startswith("self.") and var not in seen:
                self.input_vars.append(var)
                seen.add(var)

    def add_statements(self, module: ast.Module) -> None:
        """Add AST module contents while maintaining original structure"""
        self.stmts.body.extend(module.body)

    def set_returned_vars(self, vars: List[str]) -> None:
        """Set return variables with order preservation and deduplication"""
        seen = set()
        self.returned_vars = []
        for var in vars:
            var = var.strip()
            if not var.startswith("self.") and var not in seen:
                self.returned_vars.append(var)
                seen.add(var)

    @property
    def arg_names(self) -> List[ast.arg]:
        """Generate AST arguments for function definition"""
        return [ast.arg(arg=var) for var in self.input_vars]

    @property
    def return_statement(self) -> ast.Return:
        """Generate final return statement AST node"""
        return ast.Return(value=ast.Tuple(elts=[ast.Name(id=var, ctx=ast.Load()) for var in self.returned_vars], ctx=ast.Load()))

    def compile_forward(self) -> ast.FunctionDef:
        """Compile complete forward method AST"""
        return ast.FunctionDef(
            name="forward",
            args=ast.arguments(posonlyargs=[], args=self.arg_names, kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[*self.stmts.body, self.return_statement],
            decorator_list=[],
        )
