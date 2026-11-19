"""Safe arithmetic for the agent's `calculate` tool.

The model sends an expression string. We parse it into a syntax tree with `ast` and
evaluate only whitelisted node types: numbers, + - * / ** %, unary minus, and a few
named finance functions. Anything else (names, attributes, imports, strings, lambdas...)
is rejected. We never call Python's eval(), which would run arbitrary code.
"""

from __future__ import annotations

import ast
import math
import operator


class CalcError(ValueError):
    pass


def _avg(*xs: float) -> float:
    if not xs:
        raise CalcError("avg() needs at least one number")
    return sum(xs) / len(xs)


def _pct_change(new: float, old: float) -> float:
    """Percent change from old to new."""
    if old == 0:
        raise CalcError("pct_change: old value is 0")
    return (new - old) / abs(old) * 100


def _margin(part: float, whole: float) -> float:
    """part as a percent of whole, e.g. margin(net_income, revenue)."""
    if whole == 0:
        raise CalcError("margin: whole is 0")
    return part / whole * 100


def _cagr(start: float, end: float, years: float) -> float:
    """Compound annual growth rate, in percent."""
    if start <= 0 or years <= 0:
        raise CalcError("cagr: start and years must be positive")
    return ((end / start) ** (1 / years) - 1) * 100


FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *xs: sum(xs),
    "avg": _avg,
    "sqrt": math.sqrt,
    "pct_change": _pct_change,
    "margin": _margin,
    "cagr": _cagr,
}

BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
MAX_EXPRESSION_LENGTH = 500


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e15):
            raise CalcError("exponent too large")  # stops 9**9**9 from hanging the process
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise CalcError("division by zero")
        return BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and not node.keywords
    ):
        return FUNCTIONS[node.func.id](*[_eval(arg) for arg in node.args])
    raise CalcError(f"unsupported syntax: {ast.dump(node)[:80]}")


def calculate(expression: str) -> float:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"could not parse expression: {e.msg}") from None
    if isinstance(tree.body, ast.Tuple):  # "1,577" parses as the tuple (1, 577)
        raise CalcError("don't use thousands separators: write 1577, not 1,577")
    return float(_eval(tree.body))
