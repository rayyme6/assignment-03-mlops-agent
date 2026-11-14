import hashlib
import json
import subprocess

import pytest

from agent.tools.dvc_tools import DVCTools


@pytest.fixture
def repository(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/iris.csv").write_text("feature,target\n1,0\n")
    checksum = hashlib.md5((tmp_path / "data/iris.csv").read_bytes()).hexdigest()
    (tmp_path / "data/iris.csv.dvc").write_text(
        f"outs:\n- md5: {checksum}\n  path: iris.csv\n  hash: md5\n"
    )
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "Test"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "add", "data/iris.csv.dvc"],
        ["git", "commit", "-qm", "Track fixture dataset"],
    ):
        subprocess.run(argv, cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def test_dataset_version_uses_recorded_commit_and_actual_content(repository):
    result = DVCTools(repository).dataset_version()
    assert len(result["git_commit"]) == 40
    assert result["dataset_path"] == "data/iris.csv"
    assert result["data_matches_metadata"] is True
    assert result["metadata_dirty"] is False


def test_dataset_version_reports_changed_content(repository):
    (repository / "data/iris.csv").write_text("corrupt")
    assert DVCTools(repository).dataset_version()["data_matches_metadata"] is False


def test_historical_version_is_independent_of_dirty_workspace(repository):
    tools = DVCTools(repository)
    expected = tools.dataset_version()["dvc_hash"]
    (repository / "data/iris.csv.dvc").write_text("broken: metadata")
    assert tools.dataset_version(rev="HEAD")["dvc_hash"] == expected
    with pytest.raises(ValueError, match="no outputs"):
        tools.dataset_version()


@pytest.mark.parametrize(
    "path", ["../secret.dvc", "/tmp/secret.dvc", "--help.dvc", "data.csv", "data/../../x.dvc"]
)
def test_invalid_metadata_paths_rejected_before_subprocess(path, monkeypatch):
    tools = DVCTools()
    monkeypatch.setattr(tools, "_run", lambda _: pytest.fail("Unsafe input executed a subprocess"))
    with pytest.raises(ValueError):
        tools.dataset_version(path)


@pytest.mark.parametrize(
    "rev", ["--help", "HEAD; touch /tmp/unsafe", "HEAD:secret", "HEAD..main", "$(whoami)"]
)
def test_invalid_git_revisions_rejected(rev):
    with pytest.raises(ValueError, match="revision|reference"):
        DVCTools().compare_versions(rev)


def test_metadata_symlink_outside_repository_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "secret.dvc"
    external.write_text("secret")
    (repo / "secret.dvc").symlink_to(external)
    with pytest.raises(ValueError, match="inside"):
        DVCTools(repo).dataset_version("secret.dvc")


def test_dvc_status_and_diff_use_json_tool_evidence(monkeypatch):
    tools = DVCTools()
    monkeypatch.setattr(tools, "_revision", lambda rev: "a" * 40 if rev == "v1" else "b" * 40)
    calls = []

    def command(argv):
        calls.append(argv)
        return json.dumps({"data/iris.csv.dvc": [{"changed outs": {"data/iris.csv": "modified"}}]})

    monkeypatch.setattr(tools, "_run", command)
    assert tools.data_status()["changed"] is True
    assert tools.compare_versions("v1")["old_commit"] == "a" * 40
    assert calls[0] == ["dvc", "status", "--json"]
    assert calls[1][:3] == ["dvc", "diff", "--json"]


def test_timeout_propagates_without_inventing_status(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 0.5
        assert kwargs["env"]["DVC_SITE_CACHE_DIR"].endswith(".dvc/tmp/site-cache")
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        DVCTools(timeout=0.5).data_status()


def test_reproduction_instructions_never_modify_original_checkout(repository):
    before = (repository / "data/iris.csv.dvc").read_bytes()
    instructions = DVCTools(repository).reproduction_instructions("HEAD")
    assert instructions["executed"] is False
    assert "worktree add --detach" in instructions["commands"][0]
    assert instructions["git_commit"] in instructions["commands"][0]
    assert (repository / "data/iris.csv.dvc").read_bytes() == before


def test_list_datasets_preserves_recorded_dvc_results(monkeypatch):
    tools = DVCTools()
    monkeypatch.setattr(tools, "_revision", lambda _: "a" * 40)
    monkeypatch.setattr(
        tools, "_run", lambda _: '[{"path":"data/iris.csv","md5":"abc","isout":true}]'
    )
    assert tools.list_datasets()["datasets"][0]["md5"] == "abc"
