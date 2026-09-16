import numpy as np, pandas as pd
from project1_matching.src import data, candidates, features, evaluation, ranker

def test_clean_and_candidates(small_data):
    _, d = small_data
    ins, leads, offers, reports = data.clean(*data.load(d))
    assert ins.job_type.isin({"repair", "install", "both"}).all()
    assert not reports["offers_clean"].errors
    c = candidates.generate(leads.iloc[0], ins)
    assert (c.job_type.isin({"both", leads.iloc[0].job_type})).all()
    assert (c.distance_km <= 8.0).all() or c.coverage.str.contains(leads.iloc[0].zone).all()

def test_history_has_no_leakage(small_data):
    _, d = small_data
    ins, leads, offers, _ = data.clean(*data.load(d))
    as_of = offers.offered_at.quantile(0.5)
    h = features.provider_history(offers, as_of, prior_rate=0.4)
    manual = offers[offers.offered_at < as_of].groupby("provider_id").purchased.agg(["size", "sum"])
    assert (h.hist_n_offers == manual["size"]).all()

def test_ranking_metrics_perfect_and_random():
    df = pd.DataFrame({"lead_id": ["l"] * 4, "provider_id": list("abcd"), "purchased": [1, 0, 1, 0], "s": [4, 3, 2, 1]})
    m = evaluation.ranking_metrics(df, "s", k=2)
    assert m["precision@2"] == 0.5 and m["recall@2"] == 0.5
    df["s2"] = [4, 1, 3, 0]
    assert evaluation.ranking_metrics(df, "s2", k=2)["ndcg@2"] == 1.0

def test_ranker_scores_with_rank_fixed(small_data):
    _, d = small_data
    ins, leads, offers, _ = data.clean(*data.load(d))
    df = features.build_training_set(leads, ins, offers)
    tr, te = ranker.temporal_split(df)
    m = ranker.fit(ranker.make_models(), tr)["logreg"]
    p = ranker.score(m, te)
    assert p.shape == (len(te),) and (0 <= p).all() and (p <= 1).all()
