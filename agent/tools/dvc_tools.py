"""Read-only DVC operations; no shell interpolation or arbitrary checkout."""
import json
import re
import subprocess
from pathlib import Path

import yaml


class DVCTools:
    operations = {"datasets", "version", "diff", "status", "reproduce"}

    def __init__(self, root="."):
        self.root = Path(root).resolve()

    def _run(self, *args):
        return subprocess.run(args, cwd=self.root, text=True, capture_output=True,
                              check=True, timeout=30).stdout.strip()

    def _ref(self, ref):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./~^\-]*", ref):
            raise ValueError("Invalid Git reference")
        return self._run("git", "rev-parse", "--verify", ref + "^{commit}")

    def execute(self, a):
        if a.operation == "datasets":
            files = self._run("git", "ls-files", "*.dvc").splitlines()
            return {"datasets": files}
        if a.operation == "status":
            return json.loads(self._run("dvc", "status", "--json"))
        ref = self._ref(a.reference)
        if a.operation == "diff":
            return json.loads(self._run("dvc", "diff", ref, self._ref(a.other_reference),
                                        "--json"))
        if a.operation == "version":
            # Read committed metadata, never infer a version from the current data file.
            metadata = yaml.safe_load(self._run("git", "show", f"{ref}:data/iris.csv.dvc"))
            return {"git_commit": ref, "dataset": "data/iris.csv", "dvc": metadata}
        if a.operation == "reproduce":
            return {"git_commit": ref, "commands": [
                f"git worktree add --detach ../dataset-reproduction {ref}",
                "cd ../dataset-reproduction",
                "dvc pull data/iris.csv.dvc"],
                "prerequisite": "Configure access to the same DVC remote before dvc pull."}
        raise ValueError("Unsupported DVC operation")
