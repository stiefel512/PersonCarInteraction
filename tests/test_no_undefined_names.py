"""Static check for names a function uses but cannot resolve.

Exists because of a live bug: splitting `run_clip` into `track_clip` and
`judge_tracks` left a `judge_name` reference behind in the tracking half, where
no such parameter exists any more. Nothing caught it -- imports succeeded, the
whole test suite passed, and it only surfaced when the LOCO sweep actually
called `track_clip` and died with NameError eight retries deep.

This is the failure mode of every refactor that moves code between functions, so
it is worth a cheap guard rather than a dependency on a linter the project does
not otherwise need.

Deliberately narrow: it flags a bare name that is used in a function body and is
neither a parameter, an assignment in that function, a module-level name, an
import, nor a builtin. It does not attempt scope analysis beyond that, so it
will not catch everything -- but it catches exactly this.
"""
from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "pvi"
BUILTINS = set(dir(builtins))


def module_level_names(tree: ast.Module) -> set[str]:
    """Names genuinely in MODULE scope.

    Deliberately does not collect function arguments. An earlier version did,
    "to be conservative", and that defeated the whole check: `judge_name` is a
    parameter of `run_clip`, so a stale reference to it inside `track_clip` was
    whitelisted by a name belonging to a different function. Over-permissiveness
    in a lint is not caution, it is silence.
    """
    names: set[str] = set()

    def visit_block(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                for t in targets:
                    for sub in ast.walk(t):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
            elif isinstance(node, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
                # Conditional imports and try/except fallbacks live here.
                for attr in ("body", "orelse", "finalbody"):
                    visit_block(getattr(node, attr, []) or [])
                for h in getattr(node, "handlers", []):
                    visit_block(h.body)

    visit_block(tree.body)
    return names


def bound_in_function(fn: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            names.update(node.names)
    return names


def source_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def check_function(fn, enclosing: set[str], path: Path, problems: list[str]) -> None:
    """Check one function, then recurse into the functions nested inside it.

    A closure can read names bound in any enclosing function, so the allowed set
    accumulates down the nesting -- `flush()` reading `buf` from `track_clip` is
    correct Python, not a defect.
    """
    allowed = enclosing | bound_in_function(fn)

    nested = [n for n in ast.walk(fn)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn]
    nested_ids = {id(n) for n in nested}

    for node in ast.walk(fn):
        # Skip names belonging to a nested function; they are checked with that
        # function's own (wider) scope.
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if any(node in ast.walk(n) for n in nested):
                continue
            if node.id not in allowed:
                problems.append(f"{path.name}:{node.lineno} "
                                f"{fn.name}() uses undefined {node.id!r}")

    for n in nested:
        # Only direct children, so each level accumulates once.
        if any(n in ast.walk(other) for other in nested if other is not n):
            continue
        check_function(n, allowed, path, problems)


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_no_unresolvable_names(path: Path):
    tree = ast.parse(path.read_text())
    top = module_level_names(tree) | BUILTINS
    problems: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            check_function(node, top, path, problems)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    check_function(sub, top | {"self", "cls"}, path, problems)
    assert not problems, "\n".join(sorted(set(problems)))
