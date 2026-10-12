"""Read-only .NET entrypoint discovery and shell-free command serialization."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from xml.etree import ElementTree

from literate_ai.adapters.harness_tree import EXCLUDED_DIRECTORY_NAMES

PROJECT_SUFFIXES = frozenset({".csproj", ".fsproj", ".vbproj"})
SOLUTION_SUFFIXES = frozenset({".sln", ".slnx"})


def dotnet_argv(command: str) -> list[str] | None:
    """Decode only the canonical native commands emitted by this detector."""

    if not command.startswith("dotnet "):
        return None
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if (
        len(argv) == 3
        and argv[1] in {"build", "test"}
        and argv[2].startswith("./")
        and Path(argv[2]).suffix.lower() in PROJECT_SUFFIXES | SOLUTION_SUFFIXES
        and shlex.join(argv) == command
    ):
        return argv
    return None


def dotnet_wrapper_argv(command: str) -> list[str] | None:
    """Decode the portable wrapper invocation, bound to Python by its executor."""

    for phase in ("build", "test"):
        argv = ["./litai.harness.py", phase, "legacy"]
        if command == shlex.join(argv):
            return argv
    return None


def _xml(path: Path) -> ElementTree.Element | None:
    try:
        content = path.read_bytes()
        if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
            return None
        return ElementTree.fromstring(content)
    except (OSError, ElementTree.ParseError):
        return None


def _is_test_project(path: Path) -> bool:
    tree = _xml(path)
    if tree is None:
        return False
    # Do not evaluate imports, properties, conditions or MSBuild targets.
    pending = [tree]
    nodes = []
    while pending:
        node = pending.pop()
        if node.get("Condition"):
            continue
        nodes.append(node)
        pending.extend(node)
    if any(
        node.tag.rsplit("}", 1)[-1] == "IsTestProject"
        and (node.text or "").strip().lower() == "false"
        for node in nodes
    ):
        return False
    for node in nodes:
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "IsTestProject" and (node.text or "").strip().lower() == "true":
            return True
        if (
            tag == "PackageReference"
            and node.get("Include", "").lower() == "microsoft.net.test.sdk"
        ):
            return True
    return False


def inspect_dotnet(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Select one solution, otherwise one project; never choose among peers."""

    findings: list[dict[str, str]] = []
    manifests: list[Path] = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        names[:] = sorted(
            name
            for name in names
            if name not in EXCLUDED_DIRECTORY_NAMES
            and name.lower() not in {"bin", "obj"}
            and not name.startswith(".")
            and not (current / name).is_symlink()
        )
        for name in sorted(filenames):
            path = current / name
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix.lower() in PROJECT_SUFFIXES | SOLUTION_SUFFIXES:
                manifests.append(path)
            elif name == "global.json":
                findings.append(
                    {
                        "detector_id": "toolchain.dotnet-sdk",
                        "path": path.relative_to(root).as_posix(),
                        "detail": ".NET SDK configuration (not a build entrypoint)",
                    }
                )
    manifests.sort(key=lambda path: path.relative_to(root).as_posix())
    for path in manifests:
        findings.append(
            {
                "detector_id": "build-system.dotnet",
                "path": path.relative_to(root).as_posix(),
                "detail": ".NET project or solution",
            }
        )
    solutions = [path for path in manifests if path.suffix.lower() in SOLUTION_SUFFIXES]
    projects = [path for path in manifests if path.suffix.lower() in PROJECT_SUFFIXES]
    candidates = solutions or projects
    if len(candidates) != 1:
        if candidates:
            findings.append(
                {
                    "detector_id": "build-system.dotnet-ambiguous",
                    "path": candidates[0].relative_to(root).as_posix(),
                    "detail": (
                        "Multiple .NET entrypoints; no command selected. "
                        "Narrow adoption to one solution or standalone project: "
                    )
                    + ", ".join(
                        path.relative_to(root).as_posix() for path in candidates
                    ),
                }
            )
        return findings, []
    selected = candidates[0]
    members = [selected] if not solutions else _solution_projects(selected, projects)
    relative = selected.relative_to(root).as_posix()
    cwd = selected.parent.relative_to(root).as_posix()
    target = "./" + selected.name
    stages = [
        {
            "id": "build",
            "command": shlex.join(["dotnet", "build", target]),
            "evidence": relative,
            "cwd": cwd,
            "cost": "local-cheap",
        }
    ]
    tests = [path for path in members if _is_test_project(path)]
    if tests:
        evidence = tests[0].relative_to(root).as_posix()
        findings.append(
            {
                "detector_id": "test-runner.dotnet",
                "path": evidence,
                "detail": (
                    "Selected entrypoint contains an explicit .NET test SDK "
                    "or IsTestProject declaration"
                ),
            }
        )
        stages.append(
            {
                "id": "test",
                "command": shlex.join(["dotnet", "test", target]),
                "evidence": evidence,
                "cwd": cwd,
                "cost": "local-cheap",
            }
        )
    return findings, stages


def _solution_projects(solution: Path, projects: list[Path]) -> list[Path]:
    if solution.suffix.lower() == ".slnx":
        tree = _xml(solution)
        paths = (
            []
            if tree is None
            else [
                node.get("Path", "")
                for node in tree.iter()
                if node.tag.rsplit("}", 1)[-1] == "Project"
            ]
        )
    else:
        try:
            content = solution.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            return []
        paths = re.findall(
            r'^Project\("[^"\r\n]+"\)\s*=\s*"[^"\r\n]*",\s*"([^"\r\n]+)"',
            content,
            re.MULTILINE,
        )
    members = {(solution.parent / path.replace("\\", "/")).resolve() for path in paths}
    return [path for path in projects if path.resolve() in members]


def render_dotnet_wrapper(
    stages: list[dict[str, str]],
    legacy_directory: str,
    *,
    projected_root_required: bool,
) -> str:
    """A standalone Python dispatcher containing only the reviewed native argv."""

    commands = {
        stage["id"]: {"argv": dotnet_argv(stage["command"]), "cwd": stage["cwd"]}
        for stage in stages
    }
    return f"""# Generated by litai init --convert; framework-owned native .NET wrapper.
import subprocess
import sys
from pathlib import Path

LITAI_LEGACY = {legacy_directory!r}
COMMANDS = {commands!r}
if len(sys.argv) not in (2, 3) or sys.argv[1] not in COMMANDS:
    raise SystemExit("Specify a recorded stage: " + ", ".join(COMMANDS))
if {projected_root_required!r} and len(sys.argv) != 3:
    raise SystemExit("This retained harness requires a projected workspace root")
root = Path(sys.argv[2]) if len(sys.argv) == 3 else Path(__file__).parent / LITAI_LEGACY
entry = COMMANDS[sys.argv[1]]
relative = Path(entry["cwd"])
if relative.is_absolute() or ".." in relative.parts:
    raise SystemExit("Harness cwd escapes the legacy root")
cwd = (root / relative).resolve()
if not cwd.is_relative_to(root.resolve()):
    raise SystemExit("Harness cwd escapes the legacy root")
raise SystemExit(subprocess.call(entry["argv"], cwd=cwd, shell=False))
"""
