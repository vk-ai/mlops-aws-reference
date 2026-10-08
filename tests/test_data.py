import io

import pytest

from mlops_ref import data
from mlops_ref.data import FEATURES, TARGET, ChecksumError, fetch, load_raw, verify
from mlops_ref.validate import ValidationError, validate


def test_bundled_file_matches_pinned_checksum():
    assert verify(data.RAW_PATH) == data.SHA256


def test_load_raw_shape_names_and_target(raw):
    assert raw.shape == (1000, 25)
    assert {"duration_months", "credit_amount", "age_years"} <= set(raw.columns)
    assert raw[TARGET].sum() == 300  # Statlog: 700 good, 300 bad
    assert raw["age_years"].between(19, 75).all()


def test_tampered_file_is_refused(tmp_path):
    bad = tmp_path / "german.data-numeric"
    bad.write_bytes(data.RAW_PATH.read_bytes().replace(b" 1 ", b" 2 ", 1))
    with pytest.raises(ChecksumError):
        load_raw(bad)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_verifies_before_replacing(tmp_path, monkeypatch):
    good = data.RAW_PATH.read_bytes()
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda url, timeout: _Resp(good))
    dest = fetch(tmp_path / "raw" / "f")
    assert dest.read_bytes() == good

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda url, timeout: _Resp(b"not the dataset"))
    with pytest.raises(ChecksumError):
        fetch(dest)
    assert dest.read_bytes() == good  # the good copy is untouched
    assert not dest.with_suffix(".part").exists()


def test_validate_passes_on_real_data(raw):
    assert validate(raw) == {"rows": 1000, "default_rate": 0.3}


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.drop(columns=["age_years"]), "missing columns"),
    (lambda d: d.head(100), "only 100 rows"),
    (lambda d: d.assign(a01=None), "nulls"),
    (lambda d: d.assign(age_years=d["age_years"] - 30), "age_years outside"),
    (lambda d: d.assign(credit_amount=0.0), "must be > 0"),
    (lambda d: d.assign(**{TARGET: 0}), "default rate"),
])
def test_validate_rejects_bad_batches(raw, mutate, msg):
    with pytest.raises(ValidationError, match=msg):
        validate(mutate(raw.copy()))


def test_feature_list_is_24_columns():
    assert len(FEATURES) == 24 and len(set(FEATURES)) == 24
