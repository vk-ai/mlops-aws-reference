"""Runs the real DVC pipeline in a scratch git repo. Skipped when the dvc CLI isn't installed."""
import hashlib
import json
import shutil
import subprocess

import pytest

from conftest import ROOT

pytestmark = [pytest.mark.dvc, pytest.mark.skipif(shutil.which("dvc") is None, reason="pip install '.[dvc]'")]


def sh(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def test_repro_metrics_push_pull(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("src", "data/raw", "dvc.yaml", "params.yaml"):
        src = ROOT / name
        dst = repo / name
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        else:
            shutil.copy(src, dst)
    sh("git", "init", "-q", cwd=repo)
    sh("dvc", "init", "-q", cwd=repo)
    sh("dvc", "remote", "add", "-d", "local", str(tmp_path / "remote"), cwd=repo)
    sh("dvc", "repro", cwd=repo)
    metrics = json.loads((repo / "metrics" / "test_metrics.json").read_text())
    assert metrics["auc"] >= 0.70
    assert "up to date" in sh("dvc", "status", cwd=repo).lower()
    model = repo / "models" / "model.pt"
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    sh("dvc", "push", cwd=repo)
    model.unlink()
    shutil.rmtree(repo / "data" / "processed")
    sh("dvc", "pull", cwd=repo)
    assert hashlib.sha256(model.read_bytes()).hexdigest() == digest
    assert (repo / "data" / "processed" / "train.parquet").exists()
