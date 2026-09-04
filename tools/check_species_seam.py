#!/usr/bin/env python3
"""Fail the build if any module outside the species profile branches on species.

The proposal's species-seam claim — that adding sheep touches exactly one object
— is only worth anything if it is enforced. This check is that enforcement. It
scans the package for species names and for the shape of a species branch, and
exempts only the profile package, where species knowledge belongs by definition.

Run with no arguments to check the whole package:

    python tools/check_species_seam.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "lhv"

# The seam itself, plus the definitions it loads.
EXEMPT_PREFIXES = ("profiles/",)

# Species and species-specific vocabulary. A stage that names one of these has
# either hard-coded a species or is reaching around the profile.
SPECIES_WORDS = (
    "cattle",
    "cow",
    "cows",
    "bovine",
    "heifer",
    "calf",
    "dairy",
    "sheep",
    "ewe",
    "lamb",
    "ovine",
    "goat",
    "caprine",
    "pig",
    "swine",
    "porcine",
    "poultry",
    "chicken",
)

_WORD_RE = re.compile(r"\b(" + "|".join(SPECIES_WORDS) + r")\b", re.IGNORECASE)


class Finding:
    def __init__(self, path: Path, line: int, kind: str, detail: str) -> None:
        self.path = path
        self.line = line
        self.kind = kind
        self.detail = detail

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}: {self.detail}"


def _is_exempt(relative: Path) -> bool:
    text = relative.as_posix()
    return any(text.startswith(prefix) for prefix in EXEMPT_PREFIXES)


def _strings_and_names(tree: ast.AST) -> list[tuple[int, str]]:
    """Every string literal, identifier and attribute name, with its line."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append((node.lineno, node.value))
        elif isinstance(node, ast.Name):
            found.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute):
            found.append((node.lineno, node.attr))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found.append((node.lineno, node.name))
        elif isinstance(node, ast.arg):
            found.append((node.lineno, node.arg))
    return found


def check_file(path: Path, root: Path) -> list[Finding]:
    relative = path.relative_to(root)
    if _is_exempt(relative):
        return []

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    # Docstrings may discuss the domain; code may not name a species.
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=False)
            if doc is None:
                continue
            body_first = node.body[0]
            start = body_first.lineno
            end = getattr(body_first, "end_lineno", start)
            docstring_lines.update(range(start, end + 1))

    findings: list[Finding] = []
    for line, text in _strings_and_names(tree):
        if line in docstring_lines:
            continue
        match = _WORD_RE.search(text)
        if match:
            findings.append(
                Finding(
                    relative,
                    line,
                    "species name outside the profile",
                    f"{match.group(0)!r} appears in {text!r}; move it into the species profile",
                )
            )

    # A branch keyed on a species attribute is a species branch even when the
    # species itself is not named literally.
    for node in ast.walk(tree):
        if not isinstance(node, ast.If | ast.Match):
            continue
        subject = node.test if isinstance(node, ast.If) else node.subject
        for inner in ast.walk(subject):
            if isinstance(inner, ast.Attribute) and inner.attr in {"species", "species_profile"}:
                findings.append(
                    Finding(
                        relative,
                        node.lineno,
                        "branch on species",
                        "control flow keyed on the species; put the varying value in the profile",
                    )
                )
            elif isinstance(inner, ast.Name) and inner.id in {"species", "species_profile"}:
                findings.append(
                    Finding(
                        relative,
                        node.lineno,
                        "branch on species",
                        "control flow keyed on the species; put the varying value in the profile",
                    )
                )
    return findings


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else PACKAGE_ROOT
    if not root.exists():
        print(f"nothing to check: {root} does not exist", file=sys.stderr)
        return 1

    findings: list[Finding] = []
    for path in sorted(root.rglob("*.py")):
        findings.extend(check_file(path, root))

    if findings:
        print(f"species seam violated in {len(findings)} place(s):", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        print(
            "\nOnly src/lhv/profiles/ may name a species. Everything else reads the "
            "species profile.",
            file=sys.stderr,
        )
        return 1

    checked = sum(1 for _ in root.rglob("*.py"))
    print(f"species seam holds: {checked} module(s) checked, none branch on species")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
