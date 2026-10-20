"""The second law of π, enforced: no hand-set value in a decision path.

A rule that is not checked is a rule that will be broken. ``model.py`` is where
every decision of the ear is taken, so it may contain no numeric literal but the
identities 0 and 1, and it may compare nothing to a constant: every boundary is
two computed quantities facing each other. Every number it uses comes from the
weights on disk (``weights/``) or from the stream.
"""

from __future__ import annotations

import ast
from pathlib import Path

import π_ear

DECISIONS = [Path(π_ear.__file__).parent / "model.py"]
ORDERING = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _numbers(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, complex):
            yield node


def test_no_numeric_literal_in_a_decision_module():
    offenders = [
        f"{path.name}:{node.lineno} {node.value!r}"
        for path in DECISIONS
        for node in _numbers(ast.parse(path.read_text()))
        if node.value not in (0, 1)
    ]
    assert not offenders, f"hand-set numbers in a decision path: {offenders}"


def test_no_boundary_is_a_constant():
    offenders = []
    for path in DECISIONS:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Compare) and any(isinstance(op, ORDERING) for op in node.ops):
                operands = [node.left, *node.comparators]
                if any(isinstance(o, ast.Constant) for o in operands):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"a decision compared to a constant: {offenders}"


def test_the_law_test_bites():
    """The checker itself must fail on the v0 shape, or it proves nothing."""
    v0 = ast.parse("if cos_best >= 0.5 and margin >= 3.0 * sigma:\n    admit()")
    assert [n.value for n in _numbers(v0)] == [0.5, 3.0]
