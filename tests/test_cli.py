import json

from mlops_ref.cli import main


def test_dvc_stage_commands(tmp_path, capsys):
    w = ["--workdir", str(tmp_path)]
    for cmd in (["prepare"], ["featurize"], ["train"], ["evaluate"], ["drift", "--strength", "1"]):
        assert main(w + cmd) == 0
    out = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert out[0]["validate"]["rows"] == 1000
    assert out[3]["auc"] >= 0.70 and (tmp_path / "metrics" / "test_metrics.json").exists()
    assert out[4]["alert"] is True
