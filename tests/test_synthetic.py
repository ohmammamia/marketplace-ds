"""The generator is the root of every committed artefact, so its determinism
and stream independence are load-bearing."""
import numpy as np
import pandas as pd
from common import synthetic


def _gen(tmp_path, name, **kw):
    d = tmp_path / name
    return synthetic.generate_all(str(d), n_providers=40, n_leads=120, n_feedback=80, **kw)


def test_generator_is_deterministic(tmp_path):
    a, b = _gen(tmp_path, "a"), _gen(tmp_path, "b")
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_tables_have_independent_rng_streams(tmp_path, monkeypatch):
    """An edit to an earlier generator must not shift the later tables.

    A single shared `rng` threaded through all four made every table depend on
    how many draws the previous one took, which is how the committed data
    drifted out of step with this file.
    """
    base = _gen(tmp_path, "base")
    orig = synthetic.make_providers
    monkeypatch.setattr(synthetic, "make_providers",
                        lambda n, rng: (rng.random(64), orig(n, rng))[1])
    after = _gen(tmp_path, "after")
    # `feedback` draws on nothing that providers produce, so it must be untouched
    pd.testing.assert_frame_equal(base["feedback"], after["feedback"])
    pd.testing.assert_frame_equal(base["leads"], after["leads"])
