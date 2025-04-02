import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import ast
import inspect

from llm_split.ast_split import *


class NoOP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("hello")
        return x


class Model(nn.Module):
    """test doc string"""

    var_1 = 1
    var_2 = 2

    def __init__(self, n_layer: int):
        super().__init__()
        self.layers = nn.ModuleList([NoOP()] * n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += 1
        for l in self.layers:
            x = l(x)
        return x


t = get_module_tree(Model)
print(t)
forward_f, forward_idx = get_forward_node(t.body[0])
print(forward_f.body)
forward_f.body.pop(1)
print(forward_f.body)

ast.fix_missing_locations(t)

# m = Model(2)
# m(torch.tensor([1, 1, 1] ))
namespace = {} | globals()
exec(compile(t, filename="<ast>", mode="exec"), namespace)
ModifiedClass = namespace["Model"]
obj = ModifiedClass(2)
print(obj(torch.tensor([1, 1, 1])))
