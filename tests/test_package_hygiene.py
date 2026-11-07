"""Guards against leftovers: a module that nothing imports is usually a stale old file."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SRC / "narrative_engine"
ENTRY_POINTS = {
    "narrative_engine.__main__",
    "narrative_engine.eval_cli",
    "narrative_engine.story_tool",
}  # run directly, so nothing needs to import them


def dotted(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_modules() -> set:
    found = set()
    for path in PACKAGE.rglob("*.py"):
        package = dotted(path) if path.name == "__init__.py" else dotted(path).rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    anchor = package.split(".")
                    anchor = anchor[: len(anchor) - (node.level - 1)]
                    base = ".".join(anchor + ([node.module] if node.module else []))
                found.add(base)
                found.update(f"{base}.{a.name}" for a in node.names)  # `from . import module`
    return found


def test_every_module_is_used_by_something():
    modules = {dotted(p) for p in PACKAGE.rglob("*.py") if p.name != "__init__.py"}
    orphans = sorted(m for m in modules - imported_modules() - ENTRY_POINTS)
    assert (
        orphans == []
    ), f"modules that nothing imports (stale files from an older version?): {orphans}"


def console_scripts() -> dict:
    """Read [project.scripts] without tomllib (it only exists from Python 3.11)."""
    scripts, inside = {}, False
    for line in (SRC.parent / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("["):
            inside = line == "[project.scripts]"
        elif inside and "=" in line:
            name, _, target = line.partition("=")
            scripts[name.strip()] = target.strip().strip('"')
    return scripts


def test_entry_points_in_pyproject_point_at_real_functions():
    scripts = console_scripts()
    assert {"narrative-engine", "narrative-story", "narrative-eval"} <= set(scripts)
    for name, target in scripts.items():
        module, _, func = target.partition(":")
        tree = ast.parse((SRC / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
        assert any(isinstance(n, ast.FunctionDef) and n.name == func for n in tree.body), name
