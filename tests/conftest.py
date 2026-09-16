import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd, pytest
from common import synthetic

@pytest.fixture(scope="session")
def small_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    return synthetic.generate_all(str(d), seed=1, n_providers=60, n_leads=400, n_feedback=300), str(d)
