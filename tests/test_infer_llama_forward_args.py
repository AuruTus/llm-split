import ast
import inspect
from llm_split.model_ast import get_forward_node
from llm_split.var_extract import VariableVisitor, extract_input_args

from transformers import LlamaModel


def _analyze_code_segments(code: ast.Module, src_class: type) -> list[str]:
    visitor = VariableVisitor()

    visitor.visit(code)
    args = extract_input_args(visitor, src_class)

    return ["self"] + args


def test_infer_llama_forward_args():
    code = inspect.getsource(LlamaModel)
    ast_tree = ast.parse(code)
    forward_node, _ = get_forward_node(ast_tree.body[0])

    arg_list = _analyze_code_segments(forward_node, LlamaModel)

    ori_args = forward_node.args
    ori_arg_list = []
    for arg in ori_args.args:
        if arg.arg == "self":
            continue
        ori_arg_list.append(arg.arg)
    ori_arg_list = ["self"] + sorted(ori_arg_list)

    assert ori_arg_list == arg_list, f"original_arg_list: {ori_arg_list}, inferred_arg_list: {arg_list}"
