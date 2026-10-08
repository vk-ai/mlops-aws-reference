"""PySpark feature job parity. Skipped without pyspark or a Java runtime."""
import shutil

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
pytestmark = [pytest.mark.spark, pytest.mark.skipif(shutil.which("java") is None, reason="needs a Java runtime")]


def test_spark_features_match_pandas(raw, tmp_path):
    from mlops_ref import pipeline
    from mlops_ref.features import add_features

    src = tmp_path / "in" / "train.parquet"
    src.parent.mkdir()
    raw.to_parquet(src, index=False)
    out = pipeline.featurize(str(src), str(src), tmp_path, engine="spark")
    got = pd.read_parquet(out["train"])
    pd.testing.assert_frame_equal(got, add_features(raw), check_exact=False, rtol=1e-12)


def test_spark_job_writes_parquet(raw, tmp_path):
    from mlops_ref.spark_features import run

    src = tmp_path / "raw.parquet"
    raw.to_parquet(src, index=False)
    assert run(src, tmp_path / "out") == 1000
    assert len(pd.read_parquet(tmp_path / "out")) == 1000
