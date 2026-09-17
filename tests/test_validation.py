import pandas as pd, pytest
from common.validation import Schema, validate

def test_validation_catches_duplicates_ranges_orphans():
    df = pd.DataFrame({"id": ["a", "a", "b"], "x": [1.0, 50.0, 2.0], "ref": ["r1", "r2", "zz"]})
    sch = Schema(required={"id": "O", "x": "f"}, key="id", ranges={"x": (0, 10)})
    rep = validate(df, sch, "t", references={"ref": pd.Series(["r1", "r2"])})
    names = {c.name for c in rep.errors}
    assert {"unique_key", "range:x", "referential:ref"} <= names
    with pytest.raises(ValueError):
        rep.raise_if_failed()

def test_validation_passes_clean():
    df = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0]})
    rep = validate(df, Schema(required={"id": "O", "x": "f"}, key="id", ranges={"x": (0, 10)}), "t")
    assert not rep.errors

def test_passing_dtype_check_has_no_mismatch_detail():
    """A passing check must not read 'expected object, got object'."""
    df = pd.DataFrame({"a": ["x", "y"], "n": [1, 2]})
    rep = validate(df, Schema(required={"a": "O", "n": "i"}), "t")
    for c in rep.checks:
        if c.name.startswith("dtype:"):
            assert c.passed and c.detail == ""

def test_failing_dtype_check_still_explains_itself():
    df = pd.DataFrame({"a": [1.5, 2.5]})
    rep = validate(df, Schema(required={"a": "O"}), "t")
    c = next(c for c in rep.checks if c.name == "dtype:a")
    assert not c.passed and "expected object" in c.detail
