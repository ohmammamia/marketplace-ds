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
