"""Shared test fixtures extracted from test_retained_harness_remote."""

from __future__ import annotations

import tempfile
from pathlib import Path

from literate_ai.adapters.retained_harness_remote import (
    RetainedHarnessRemoteRequest,
    _runtime_archive,
    _source_archive,
    execute_retained_harness_receiver,
    validate_retained_remote_result,
)
from literate_ai.contracts import (
    ExecutionRequirements,
    ExecutionWorker,
    ExecutionWorkerKind,
    canonical_identity,
    canonical_json_bytes,
)


class LoopbackRetainedExecutor:
    """Replace SSH transport while exercising the real receiver and source custody."""

    def __init__(self) -> None:
        self.results = []

    def execute(self, worker, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = kwargs["inventory"]
            archive = root / "source.tar.gz"
            manifest, archive_identity, archive_size = _source_archive(
                kwargs["source_root"], inventory, archive
            )
            runtime = root / "runtime.zip"
            runtime_identity, runtime_size = _runtime_archive(
                Path(__file__).parents[2] / "src/literate_ai",
                runtime,
                inventory=inventory,
                source_root=kwargs["source_root"],
            )
            request = RetainedHarnessRemoteRequest(
                kwargs["project_id"],
                kwargs["project_revision_identity"],
                worker.identity,
                kwargs["worker_catalog_identity"],
                kwargs["runner_identity"],
                kwargs["lifecycle_request_identity"],
                canonical_identity(inventory),
                kwargs["source_identity"],
                manifest,
                archive_identity,
                archive_size,
                runtime_identity,
                runtime_size,
                kwargs["timeout_seconds"],
            )
            controls = {
                "worker": worker.to_dict(),
                "request": request.to_dict(),
                "inventory": inventory,
            }
            for name, value in controls.items():
                (root / f"{name}.json").write_bytes(canonical_json_bytes(value))
            result = execute_retained_harness_receiver(
                worker_file=root / "worker.json",
                request_file=root / "request.json",
                inventory_file=root / "inventory.json",
                archive=archive,
                runtime_archive=runtime,
                workspace=root / "workspace",
            )
            validate_retained_remote_result(result, request, worker)
            self.results.append(result)
            return result


def _worker(*, os_family: str = "linux") -> ExecutionWorker:
    return ExecutionWorker(
        os_family,
        ExecutionWorkerKind.SSH,
        endpoint=f"runner@{os_family}.example",
        workspace="~/literate-ai",
        requirements=ExecutionRequirements(os_family=os_family),
        lifecycle_executable="~/.local/bin/litai",
    )
