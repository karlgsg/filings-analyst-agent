import pytest

from analyst.agent.calculator import CalcError, calculate


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1577 + 3", 1580),
        ("(5363 - 4869) / 4869 * 100", (5363 - 4869) / 4869 * 100),
        ("-370 * 2", -740),
        ("2 ** 10", 1024),
        ("avg(2.3, 3.1, 3.0)", (2.3 + 3.1 + 3.0) / 3),
        ("pct_change(110, 100)", 10.0),
        ("margin(1228, 39403)", 1228 / 39403 * 100),
        ("cagr(100, 121, 2)", 10.0),
        ("round(3.14159, 2)", 3.14),
    ],
)
def test_valid_expressions(expr, expected):
    assert calculate(expr) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hacked')",  # code execution attempt
        "open('/etc/passwd').read()",
        "(1).__class__",  # attribute access
        "x + 1",  # unknown name
        "'a' * 3",  # strings
        "9 ** 9 ** 9",  # would hang the process
        "1 / 0",
        "lambda: 1",
        "1,577 + 3",  # thousands separator
        "round(3.14159, ndigits=2)",  # keyword args
        "1" * 600,  # too long
    ],
)
def test_rejected_expressions(expr):
    with pytest.raises(CalcError):
        calculate(expr)
