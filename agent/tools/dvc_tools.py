"""Read-only, bounded DVC and Git operations with explicit dataset provenance."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


class DVCTools:
    """Inspect real DVC metadata without allowing arbitrary shell commands."""

    def __init__(self, repo_path: str | Path = ".", timeout: float = 15):
        self.repo_path = Path(repo_path).resolve()
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def _run(self, argv: list[str]) -> str:
        approved = {"git": {"rev-parse", "show"}, "dvc": {"list", "status", "diff"}}
        if len(argv) < 2 or argv[0] not in approved or argv[1] not in approved[argv[0]]:
            raise ValueError("Only approved Git and DVC operations are permitted")
        env = os.environ.copy()
        env.setdefault("DVC_SITE_CACHE_DIR", str(self.repo_path / ".dvc/tmp/site-cache"))
        env.setdefault("DVC_NO_ANALYTICS", "1")
        result = subprocess.run(
            argv,
            cwd=self.repo_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return result.stdout.strip()

    def _revision(self, rev: str) -> str:
        if not isinstance(rev, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._/~^-]{0,199}", rev
        ):
            raise ValueError("Invalid Git revision")
        if ".." in rev:
            raise ValueError("A single commit reference is required")
        return self._run(["git", "rev-parse", "--verify", f"{rev}^{{commit}}"])

    def _path(self, path: str) -> str:
        if not isinstance(path, str) or not path or "\\" in path:
            raise ValueError("A relative .dvc metadata path is required")
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or not path.endswith(".dvc"):
            raise ValueError("A relative .dvc metadata path is required")
        if any(part.startswith("-") for part in relative.parts):
            raise ValueError("Option-like path components are not permitted")
        resolved = (self.repo_path / path).resolve()
        if not resolved.is_relative_to(self.repo_path):
            raise ValueError("Metadata path must stay inside the repository")
        return str(relative)

    def list_datasets(self) -> dict[str, Any]:
        datasets = json.loads(
            self._run(["dvc", "list", ".", "--dvc-only", "--recursive", "--json"])
        )
        return {"datasets": datasets, "git_commit": self._revision("HEAD")}

    def dataset_version(
        self, path: str = "data/iris.csv.dvc", rev: str | None = None
    ) -> dict[str, Any]:
        path = self._path(path)
        commit = self._revision(rev or "HEAD")
        committed = self._run(["git", "show", f"{commit}:{path}"])
        metadata_text = committed if rev is not None else (self.repo_path / path).read_text()
        metadata = yaml.safe_load(metadata_text)
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("outs"), list)
            or not metadata["outs"]
        ):
            raise ValueError("DVC metadata contains no outputs")
        outputs = []
        for output in metadata["outs"]:
            if not isinstance(output, dict) or "path" not in output:
                raise ValueError("Invalid DVC output metadata")
            dataset_path = PurePosixPath(path).parent / str(output["path"])
            if dataset_path.is_absolute() or ".." in dataset_path.parts:
                raise ValueError("External DVC outputs are not permitted")
            algorithm = output.get("hash", "md5")
            checksum = output.get(algorithm)
            if not checksum:
                raise ValueError("DVC output is missing its recorded content hash")
            detail = {
                "dataset_path": str(dataset_path),
                "dvc_hash": checksum,
                "hash_algorithm": algorithm,
                "size": output.get("size"),
            }
            if rev is None and algorithm == "md5" and not str(checksum).endswith(".dir"):
                actual = (self.repo_path / dataset_path).resolve()
                if not actual.is_relative_to(self.repo_path):
                    raise ValueError("Dataset path must stay inside the repository")
                if actual.is_file():
                    digest = hashlib.md5()  # noqa: S324 -- DVC content identity, not security
                    with actual.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    detail["data_matches_metadata"] = digest.hexdigest() == checksum
                else:
                    detail["data_matches_metadata"] = False
            outputs.append(detail)
        result = {
            "git_commit": commit,
            "dvc_file": path,
            "metadata_dirty": rev is None and metadata_text.strip() != committed.strip(),
            "outputs": outputs,
            "source": "recorded DVC metadata at Git revision" if rev else "workspace DVC metadata",
        }
        if len(outputs) == 1:
            result.update(outputs[0])
        return result

    def compare_versions(self, old_rev: str, new_rev: str = "HEAD") -> dict[str, Any]:
        old_commit, new_commit = self._revision(old_rev), self._revision(new_rev)
        diff = json.loads(self._run(["dvc", "diff", "--json", old_commit, new_commit]))
        return {"old_commit": old_commit, "new_commit": new_commit, "diff": diff}

    def data_status(self) -> dict[str, Any]:
        status = json.loads(self._run(["dvc", "status", "--json"]))
        return {"changed": bool(status), "status": status, "scope": "local DVC workspace and cache"}

    def reproduction_instructions(
        self, rev: str, path: str = "data/iris.csv.dvc"
    ) -> dict[str, Any]:
        """Return reviewable commands; never change the caller's checkout."""
        provenance = self.dataset_version(path=path, rev=rev)
        commit = provenance["git_commit"]
        destination = f"/tmp/mlops-reproduce-{commit[:12]}"
        commands = [
            shlex.join(["git", "worktree", "add", "--detach", destination, commit]),
            shlex.join(["dvc", "--cd", destination, "pull", provenance["dvc_file"]]),
            shlex.join(["dvc", "--cd", destination, "checkout", provenance["dvc_file"]]),
            shlex.join(["dvc", "--cd", destination, "status", "--json"]),
        ]
        return {
            **provenance,
            "commands": commands,
            "executed": False,
            "note": (
                "Run from this Git repository. Use an unused worktree directory. The configured DVC "
                "remote must be accessible; relative local remote URLs must be configured as absolute "
                "paths in the new worktree before pulling. No existing checkout is changed."
            ),
        }
