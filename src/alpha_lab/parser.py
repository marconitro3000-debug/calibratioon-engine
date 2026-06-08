import ast
import operator
import re
from collections.abc import Callable
from typing import Any

from . import operators as ops

_ALLOWED_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "rank": ops.rank,
    "zscore": ops.zscore,
    "scale": ops.scale,
    "delay": ops.delay,
    "delta": ops.delta,
    "returns": ops.returns,
    "ts_mean": ops.ts_mean,
    "ts_std": ops.ts_std,
    "ts_min": ops.ts_min,
    "ts_max": ops.ts_max,
    "ts_rank": ops.ts_rank,
    "ts_corr": ops.ts_corr,
    "ts_cov": ops.ts_cov,
    "decay_linear": ops.decay_linear,
    "signed_power": ops.signed_power,
    "winsorize": ops.winsorize,
    "neutralize": ops.neutralize,
}

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _expand_adv_tokens(expr: str) -> str:
    # WorldQuant-style convenience token: adv20 -> adv(20)
    return re.sub(r"\badv(\d+)\b", r"adv(\1)", expr)


class AlphaParser:
    """Allowlisted formula evaluator for vectorized alpha expressions.

    This parser does not use Python eval. It accepts only names from the input
    data, allowlisted operators, numeric constants and arithmetic expressions.
    """

    def evaluate(self, expression: str, data: dict, groups: dict | None = None):
        expression = _expand_adv_tokens(expression)
        functions = dict(_ALLOWED_FUNCTIONS)

        def adv(n: int):
            if "volume" not in data:
                raise KeyError("adv(n) requires data['volume']")
            return data["volume"].rolling(int(n), min_periods=max(2, int(n) // 3)).mean()

        def group_neutralize(x):
            return ops.group_neutralize(x, groups=groups)

        functions["adv"] = adv
        functions["group_neutralize"] = group_neutralize

        tree = ast.parse(expression, mode="eval")
        return _SafeFormulaEvaluator(data=data, functions=functions).visit(tree)


class _SafeFormulaEvaluator(ast.NodeVisitor):
    def __init__(self, *, data: dict[str, Any], functions: dict[str, Callable[..., Any]]):
        self.data = data
        self.functions = functions

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name):
        if node.id in self.data:
            return self.data[node.id]
        if node.id in self.functions:
            return self.functions[node.id]
        raise NameError(f"name is not allowed in alpha expression: {node.id}")

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, int | float):
            return node.value
        raise ValueError("only numeric constants are allowed in alpha expressions")

    def visit_Call(self, node: ast.Call):
        if node.keywords:
            raise ValueError("keyword arguments are not allowed in alpha expressions")
        func = self.visit(node.func)
        if func not in self.functions.values():
            raise ValueError("function is not allowed in alpha expression")
        args = [self.visit(arg) for arg in node.args]
        return func(*args)

    def visit_BinOp(self, node: ast.BinOp):
        op = _BINARY_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("binary operator is not allowed in alpha expression")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp):
        op = _UNARY_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError("unary operator is not allowed in alpha expression")
        return op(self.visit(node.operand))

    def visit_Attribute(self, node: ast.Attribute):
        raise ValueError("attribute access is not allowed in alpha expressions")

    def visit_Subscript(self, node: ast.Subscript):
        raise ValueError("subscript access is not allowed in alpha expressions")

    def generic_visit(self, node: ast.AST):
        raise ValueError(f"expression node is not allowed: {type(node).__name__}")
