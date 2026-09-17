import pandas as pd
from project2_feedback_nlp.src import privacy, quality, emerging, taxonomy

def test_redaction_removes_identifiers():
    t, c = privacy.redact("Call me on 07123 456 789 or tom.x@example.com, I'm in E17 4AB. Provider Ahmed was late")
    assert "07123" not in t and "@" not in t and "4AB" not in t and "Ahmed" not in t
    assert c["PHONE"] == 1 and c["EMAIL"] == 1 and c["POSTCODE"] == 1 and c["NAME"] >= 1

def test_quality_flags_reasons():
    df = pd.DataFrame({"text": ["good job thanks", "good job thanks", "", "ok", "Zażółć gęślą jaźń ćma żółw"]})
    d, rep = quality.apply(df)
    assert rep["exclusions"] == {"exact_duplicate": 1, "empty": 1, "too_short": 1, "non_english": 1}

def test_emerging_flags_spike_but_not_volume_lift():
    weeks = pd.date_range("2025-01-06", periods=16, freq="7D")
    a = [10] * 12 + [10] * 4; b = [10] * 12 + [30] * 4; c = [10] * 12 + [12] * 4
    counts = pd.DataFrame({"a": a, "b": b, "c": c}, index=weeks)
    out = emerging.detect(counts).set_index("category")
    assert out.loc["b", "emerging"] and not out.loc["a", "emerging"] and not out.loc["c", "emerging"]
    lifted = counts * 2                                       # everything doubles: no *relative* emergence except b
    out2 = emerging.detect(lifted).set_index("category")
    assert not out2.loc["a", "emerging"]

def test_rule_label_and_severity_cover_taxonomy():
    assert set(taxonomy.SEVERITY) == set(taxonomy.CATEGORIES)
    assert taxonomy.rule_label("requested a refund for the credits") == "credits_refund"
    assert taxonomy.rule_label("zzz") == "unassigned"

import pytest

@pytest.mark.parametrize("text,label", [
    ("call me on 07700-900-123", "PHONE"),        # hyphen-separated
    ("07700.900.123 is mine", "PHONE"),           # dot-separated
    ("ring +44 20 7946 0958", "PHONE"),           # international landline
    ("+44 (0) 7700 900123 please", "PHONE"),      # +44(0) form
    ("I live at sw1a 1aa", "POSTCODE"),           # lower case
])
def test_redaction_covers_realistic_pii_formats(text, label):
    """Each of these survived both the redactor and the old `\\d{10}|@` gate."""
    out, counts = privacy.redact(text)
    assert label in counts, f"{text!r} left un-redacted: {out!r}"
    assert not privacy.residual_pii(pd.Series([out])).any()

def test_possessive_name_is_redacted():
    out, counts = privacy.redact("Tom's work was poor")
    assert "Tom" not in out and counts.get("NAME") == 1

def test_residual_pii_gate_catches_what_the_patterns_catch():
    """The gate must use the same patterns as the redactor, not a weaker proxy."""
    leaked = pd.Series(["call me on 07700-900-123", "I live at sw1a 1aa"])
    assert privacy.residual_pii(leaked).all()
    assert not privacy.residual_pii(pd.Series(["no identifiers here", "paid 250 pounds"])).any()

def test_emerging_applies_fdr_correction():
    """A category significant on its raw p-value but not after BH must not flag."""
    import numpy as np
    counts = pd.DataFrame({f"c{i}": [5] * 12 + [6] * 4 for i in range(8)})
    counts["c0"] = [5] * 12 + [9] * 4                     # a mild, borderline lift
    out = emerging.detect(counts)
    assert {"p_value", "q_value", "emerging"} <= set(out.columns)
    assert (out.q_value >= out.p_value - 1e-9).all()      # BH never lowers a p-value
    assert not out.loc[out.q_value >= 0.05, "emerging"].any()
