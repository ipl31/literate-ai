"""Explicit multi-root plans never infer custody or call the flat converter."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from literate_ai.adapters.harness_inventory import inspect_harness
from literate_ai.adapters.monorepo_adoption import (
    SELECTION_SCHEMA,
    build_root_candidates,
    plan_monorepo_adoption,
)
from literate_ai.adapters.project_initialization import plan_convert
from literate_ai.cli import main
from tests.support.fixtures_test_operator_adoption import TtyStringIO


class MonorepoAdoptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="litai roots ")
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name).resolve()
        self.root = base / "project"
        self.root.mkdir()
        self.selection_file = base / "selection.json"
        for name in ("kit", "runtime"):
            (self.root / name).mkdir()
            (self.root / name / "Makefile").write_text(
                "all:\n\t@echo build\ntest:\n\t@echo test\n", encoding="utf-8"
            )
            (self.root / name / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.root / "shared").mkdir()
        (self.root / "shared" / "api.py").write_text("API = 1\n", encoding="utf-8")
        (self.root / "repo.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.selection = {
            "schema": SELECTION_SCHEMA,
            "components": [
                {
                    "name": name,
                    "root": name,
                    "commands": [
                        {
                            "id": phase,
                            "command": command,
                            "cwd": name,
                            "evidence": f"{name}/Makefile",
                        }
                        for phase, command in (("build", "make"), ("test", "make test"))
                    ],
                }
                for name in ("kit", "runtime")
            ],
            "shared_sources": [
                {"path": "shared", "owner": "kit", "consumers": ["runtime"]},
                {"path": "repo.sh", "owner": "kit", "consumers": ["runtime"]},
            ],
        }

    def plan(self, selection=None):
        self.selection_file.write_text(
            json.dumps(self.selection if selection is None else selection),
            encoding="utf-8",
        )
        return plan_monorepo_adoption(
            self.root, self.selection_file, inspect_harness(self.root), submodules=[]
        )

    def snapshot(self):
        return {
            p.relative_to(self.root).as_posix(): p.read_bytes() if p.is_file() else None
            for p in self.root.rglob("*")
        }

    def test_candidates_preserve_independent_markers_without_selecting_components(self):
        candidates = build_root_candidates(inspect_harness(self.root))
        self.assertEqual([c["root"] for c in candidates], [".", "kit", "runtime"])
        self.assertTrue(
            all(
                c["authority"] == "candidate-only" and not c["selected"]
                for c in candidates
            )
        )
        self.assertEqual(candidates[1]["markers"], ["kit/Makefile"])
        self.assertEqual(candidates[1]["stages"], [])  # No invented command.

    def test_cli_refined_plan_requires_explicit_baseline_execution_for_apply(self):
        self.plan()
        before = self.snapshot()
        args = [
            "--json",
            "onboard",
            "adopt",
            str(self.root),
            "--default-branch",
            "main",
            "--root-plan",
            str(self.selection_file),
        ]
        with (
            mock.patch(
                "literate_ai.cli.operator.inspect_host_preflight", return_value={}
            ),
            mock.patch("literate_ai.cli.operator.init_project_from_args") as mutator,
            mock.patch(
                "literate_ai.cli.dispatch._perf_build_root",
                return_value=self.root / "_build",
            ),
        ):
            output, errors = io.StringIO(), io.StringIO()
            self.assertEqual(
                main(args, stdout=output, stderr=errors), 0, errors.getvalue()
            )
            result = json.loads(output.getvalue())["result"]
            self.assertFalse(result["apply_supported"])
            self.assertIn("monorepo.baseline_execution_required", result["blockers"])
            self.assertIsNone(result["landing_stage"])
            self.assertEqual(result["next_commands"], [])
            for flags, code in (
                (["--apply"], "operator.onboard_acknowledgement_required"),
                (
                    ["--apply", "--acknowledge", "--allow-unready"],
                    "operator.onboard_plan_blocked",
                ),
                (
                    [
                        "--apply",
                        "--acknowledge",
                        "--expect-plan",
                        result["plan_identity"],
                    ],
                    "operator.onboard_plan_blocked",
                ),
            ):
                errors = io.StringIO()
                self.assertNotEqual(
                    main(args + flags, stdout=io.StringIO(), stderr=errors), 0
                )
                self.assertEqual(json.loads(errors.getvalue())["error"]["code"], code)
            (self.root / "kit" / "app.py").write_text("changed\n", encoding="utf-8")
            errors = io.StringIO()
            self.assertNotEqual(
                main(
                    args
                    + [
                        "--apply",
                        "--acknowledge",
                        "--expect-plan",
                        result["plan_identity"],
                    ],
                    stdout=io.StringIO(),
                    stderr=errors,
                ),
                0,
            )
            self.assertEqual(
                json.loads(errors.getvalue())["error"]["code"],
                "operator.onboard_plan_changed",
            )
            mutator.assert_not_called()
        after = self.snapshot()
        self.assertEqual(set(before), set(after))
        self.assertEqual(
            {k: v for k, v in before.items() if k != "kit/app.py"},
            {k: v for k, v in after.items() if k != "kit/app.py"},
        )

    def test_cli_refined_apply_routes_the_reviewed_plan_to_the_mutator(self):
        self.plan()
        args = [
            "--json",
            "onboard",
            "adopt",
            str(self.root),
            "--default-branch",
            "main",
            "--root-plan",
            str(self.selection_file),
            "--run-baseline",
        ]
        initialized = {"schema": "literate-ai/project-initialization@6"}
        with (
            mock.patch(
                "literate_ai.cli.operator.inspect_host_preflight", return_value={}
            ),
            mock.patch(
                "literate_ai.cli.operator.init_project_from_args",
                return_value=initialized,
            ) as mutator,
            mock.patch(
                "literate_ai.cli.dispatch._perf_build_root",
                return_value=self.root / "_build",
            ),
        ):
            output, errors = io.StringIO(), io.StringIO()
            self.assertEqual(
                main(args, stdout=output, stderr=errors), 0, errors.getvalue()
            )
            plan = json.loads(output.getvalue())["result"]
            self.assertTrue(plan["apply_supported"])
            self.assertEqual(plan["blockers"], [])
            output, errors = io.StringIO(), io.StringIO()
            self.assertEqual(
                main(
                    args
                    + [
                        "--apply",
                        "--acknowledge",
                        "--expect-plan",
                        plan["plan_identity"],
                    ],
                    stdout=output,
                    stderr=errors,
                ),
                0,
                errors.getvalue(),
            )
            namespace = mutator.call_args.args[0]
            self.assertEqual(namespace.root_plan, str(self.selection_file))
            self.assertTrue(namespace.run_baseline)


class DotnetAdoptionTests(unittest.TestCase):
    def test_native_manifests_plan_read_only_with_existing_driver_precedence(self):
        with tempfile.TemporaryDirectory(prefix="litai dotnet ") as directory:
            for suffix in ("csproj", "fsproj", "vbproj"):
                with self.subTest(suffix=suffix):
                    root = Path(directory) / suffix
                    project = root / "src" / f"Library Space.{suffix}"
                    project.parent.mkdir(parents=True)
                    project.write_text('<Project Sdk="Microsoft.NET.Sdk"/>')
                    for generated in ("bin", "obj"):
                        output = project.parent / generated
                        output.mkdir()
                        (output / "Generated.csproj").write_text("<Project/>")
                    before = {
                        p.relative_to(root).as_posix(): p.read_bytes()
                        for p in root.rglob("*")
                        if p.is_file()
                    }
                    plan = plan_convert(root)
                    self.assertEqual(plan["readiness"], "ready")
                    self.assertEqual(plan, plan_convert(root))
                    stages = plan["proposed_wrapper_stages"]
                    self.assertEqual([stage["id"] for stage in stages], ["build"])
                    self.assertEqual(
                        stages[0]["command"],
                        f"dotnet build './Library Space.{suffix}'",
                    )
                    self.assertEqual(
                        stages[0]["evidence"], f"src/Library Space.{suffix}"
                    )
                    self.assertEqual(stages[0]["cwd"], "src")
                    self.assertEqual(
                        before,
                        {
                            p.relative_to(root).as_posix(): p.read_bytes()
                            for p in root.rglob("*")
                            if p.is_file()
                        },
                    )
                    if suffix == "csproj":
                        for declaration, expected in (
                            (
                                "<PropertyGroup><IsTestProject>true</IsTestProject></PropertyGroup>",
                                ["build", "test"],
                            ),
                            (
                                "<PropertyGroup><IsTestProject>false</IsTestProject></PropertyGroup>",
                                ["build"],
                            ),
                            (
                                '<PropertyGroup Condition="false">'
                                "<IsTestProject>true</IsTestProject></PropertyGroup>",
                                ["build"],
                            ),
                            ("<!-- <IsTestProject>true</IsTestProject> -->", ["build"]),
                        ):
                            with self.subTest(declaration=declaration):
                                project.write_text(f"<Project>{declaration}</Project>")
                                self.assertEqual(
                                    [
                                        stage["id"]
                                        for stage in plan_convert(root)[
                                            "proposed_wrapper_stages"
                                        ]
                                    ],
                                    expected,
                                )
                    (root / "Makefile").write_text("all:\n\t@echo built\n")
                    self.assertEqual(
                        plan_convert(root)["proposed_wrapper_stages"][0]["command"],
                        "make -f Makefile",
                    )
                    self.assertEqual(
                        len(plan_convert(root)["proposed_wrapper_stages"]), 1
                    )

    def test_solution_members_prove_tests_and_ambiguity_never_selects_a_peer(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("sln", "slnx"):
                with self.subTest(suffix=suffix):
                    root = Path(directory) / suffix
                    root.mkdir()
                    (root / "App.csproj").write_text(
                        '<Project Sdk="Microsoft.NET.Sdk"/>'
                    )
                    tests = root / "tests" / "Tests.csproj"
                    tests.parent.mkdir()
                    tests.write_text(
                        "<Project><ItemGroup>"
                        '<PackageReference Include="Microsoft.NET.Test.Sdk"/>'
                        "</ItemGroup></Project>"
                    )
                    solution = root / f"App.{suffix}"
                    solution.write_text(
                        '<Solution><Project Path="App.csproj"/>'
                        '<Project Path="tests/Tests.csproj"/></Solution>'
                        if suffix == "slnx"
                        else (
                            'Project("{ID}") = "App", "App.csproj", "{APP}"\n'
                            "EndProject\n"
                            'Project("{ID}") = "Tests", '
                            '"tests\\Tests.csproj", "{TEST}"\n'
                            "EndProject\n"
                        )
                    )
                    (root / "global.json").write_text('{"sdk":{"version":"8.0.100"}}')
                    plan = plan_convert(root)
                    self.assertEqual(plan["readiness"], "ready")
                    self.assertEqual(
                        [stage["command"] for stage in plan["proposed_wrapper_stages"]],
                        [f"dotnet build ./App.{suffix}", f"dotnet test ./App.{suffix}"],
                    )
                    self.assertIn(
                        "toolchain.dotnet-sdk",
                        [finding["detector_id"] for finding in plan["findings"]],
                    )
                    # A test project outside the solution cannot authorize a test stage.
                    solution.write_text(
                        '<Solution><Project Path="App.csproj"/></Solution>'
                        if suffix == "slnx"
                        else (
                            'Project("{ID}") = "App", "App.csproj", "{APP}"\n'
                            "EndProject\n"
                        )
                    )
                    self.assertEqual(
                        [
                            stage["id"]
                            for stage in plan_convert(root)["proposed_wrapper_stages"]
                        ],
                        ["build"],
                    )
                    (root / "Other.sln").write_text("")
                    ambiguous = plan_convert(root)
                    self.assertEqual(ambiguous["readiness"], "blocked-no-driver")
                    self.assertEqual(ambiguous["proposed_wrapper_stages"], [])
                    self.assertEqual(ambiguous, plan_convert(root))
                    self.assertIn(
                        "build-system.dotnet-ambiguous",
                        [finding["detector_id"] for finding in ambiguous["findings"]],
                    )
                    output, errors = TtyStringIO(), io.StringIO()
                    self.assertEqual(
                        main(
                            ["onboard", "adopt", str(root)],
                            stdout=output,
                            stderr=errors,
                        ),
                        0,
                        errors.getvalue(),
                    )
                    self.assertIn("  - blocked-no-driver", output.getvalue())
                    solution.unlink()
                    (root / "Other.sln").unlink()
                    self.assertEqual(
                        plan_convert(root)["readiness"], "blocked-no-driver"
                    )
                    tests.unlink()
                    (root / "App.csproj").unlink()
                    self.assertEqual(
                        plan_convert(root)["readiness"], "blocked-no-driver"
                    )


if __name__ == "__main__":
    unittest.main()
