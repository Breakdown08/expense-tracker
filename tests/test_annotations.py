"""Every variable is annotated (Kirill's rule): mypy --strict checks signatures, this checks the rest.

A name bound by assignment, for, with or := must be declared with a type in its scope, before or
at the binding; so must self.<attr>. Exempt: comprehension variables and `except ... as` names,
which Python gives no place to annotate, and enum members, which the typing spec forbids annotating.
"""

import ast
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
SOURCES: list[Path] = sorted(p for d in ("src", "tests") for p in (ROOT / d).rglob("*.py"))
type Scope = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
SCOPES: tuple[type[ast.AST], ...] = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
NESTED: tuple[type[ast.AST], ...] = (
    *SCOPES,
    ast.Lambda,
    ast.comprehension,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def targets(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.Tuple | ast.List):
        return [t for e in node.elts for t in targets(e)]
    if isinstance(node, ast.Starred):
        return targets(node.value)
    return [node]


def is_enum(scope: Scope) -> bool:
    return isinstance(scope, ast.ClassDef) and any(
        isinstance(b, ast.Name) and b.id.endswith("Enum") for b in scope.bases
    )


def key(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return f"self.{node.attr}"
    return None  # subscripts and foreign attributes mutate, they do not declare


def own_nodes(scope: Scope) -> list[ast.AST]:
    """Nodes of this scope, not of nested functions, classes or comprehensions."""
    out: list[ast.AST] = []
    stack: list[ast.AST] = list(ast.iter_child_nodes(scope))
    while stack:
        node: ast.AST = stack.pop()
        out.append(node)
        if not isinstance(node, NESTED):
            stack.extend(ast.iter_child_nodes(node))
        elif not isinstance(node, (*SCOPES, ast.Lambda)):
            out += [n for n in ast.walk(node) if isinstance(n, ast.NamedExpr)]  # := in a comprehension
    return out


def violations(path: Path, root: Path = ROOT) -> list[str]:
    tree: ast.Module = ast.parse(path.read_text())
    found: list[str] = []
    self_attrs: dict[ast.ClassDef, set[str]] = {}
    k: str | None
    cls: ast.ClassDef
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        self_attrs[cls] = {str(key(n.target)) for n in ast.walk(cls) if isinstance(n, ast.AnnAssign)}
    scopes: list[Scope] = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    scope: Scope
    for scope in scopes:
        if is_enum(scope):
            continue
        nodes: list[ast.AST] = own_nodes(scope)
        declared: dict[str, int] = {}
        node: ast.AST
        for node in nodes:
            if isinstance(node, ast.AnnAssign) and (k := key(node.target)) is not None:
                declared[k] = min(declared.get(k, node.lineno), node.lineno)
        if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
            a: ast.arg
            for a in [*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs]:
                declared[a.arg] = 0
        owner: list[ast.ClassDef] = [c for c, _ in self_attrs.items() if scope in ast.walk(c)]
        bound: list[ast.expr] = []
        for node in nodes:
            if isinstance(node, ast.Assign):
                bound += [t for tgt in node.targets for t in targets(tgt)]
            elif isinstance(node, ast.For | ast.AsyncFor):
                bound += targets(node.target)
            elif isinstance(node, ast.With | ast.AsyncWith):
                bound += [t for item in node.items if item.optional_vars for t in targets(item.optional_vars)]
            elif isinstance(node, ast.NamedExpr):
                bound.append(node.target)
        t: ast.expr
        for t in bound:
            k = key(t)
            if k is None:
                continue
            if k.startswith("self.") and owner and k in self_attrs[owner[-1]]:
                continue
            if declared.get(k, t.lineno + 1) > t.lineno:
                found.append(f"{path.relative_to(root)}:{t.lineno}: {k} is not annotated")
    return sorted(found, key=lambda v: int(v.split(":")[1]))


def test_every_variable_is_annotated() -> None:
    found: list[str] = [v for p in SOURCES for v in violations(p)]
    assert found == []


def test_checker_catches_untyped(tmp_path: Path) -> None:
    src: Path = tmp_path / "x.py"
    src.write_text(
        "a = 1\nb: int = 2\nfor i in []:\n    pass\nx, y = 1, 2\nz: int\nw: int\nz, w = 1, 2\n"
        "q = [n for n in [1] if (m := n)]\n"
        "class E(StrEnum):\n    A = 'a'\n"
    )
    lines: list[str] = [v.split(": ", 1)[1] for v in violations(src, tmp_path)]
    assert lines == [
        "a is not annotated",
        "i is not annotated",
        "x is not annotated",
        "y is not annotated",
        "q is not annotated",
        "m is not annotated",
    ]
