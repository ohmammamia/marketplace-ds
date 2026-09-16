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
