"""Unsupervised discovery: TF-IDF + NMF. Chosen over LDA (short texts,
sparse counts) and over embedding clustering (harder to explain, no evident
gain on this corpus size). Topics are *evidence for taxonomy design*, not the
final categories — see taxonomy.py."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

EXTRA_STOP = {"name", "phone", "email", "postcode", "url", "marketplace", "job", "jobs"}


def vectorise(texts: pd.Series):
    vec = TfidfVectorizer(min_df=3, max_df=0.6, ngram_range=(1, 2), sublinear_tf=True,
                          stop_words=list(__import__("sklearn.feature_extraction.text", fromlist=["ENGLISH_STOP_WORDS"]).ENGLISH_STOP_WORDS | EXTRA_STOP))
    return vec, vec.fit_transform(texts)


def fit_topics(X, vocab: np.ndarray, n_topics: int, seed: int = 0, top: int = 8):
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=seed, max_iter=400)
    W = nmf.fit_transform(X)
    H = nmf.components_
    terms = [[vocab[i] for i in np.argsort(H[k])[::-1][:top]] for k in range(n_topics)]
    return nmf, W, terms


def coherence_proxy(X, vocab_index: dict, terms: list[list[str]]) -> float:
    """Mean pairwise document co-occurrence (PMI-like) of top terms; higher is better."""
    Xb = (X > 0).astype(int)
    N = Xb.shape[0]
    scores = []
    for tl in terms:
        idx = [vocab_index[t] for t in tl if t in vocab_index]
        M = Xb[:, idx]
        df = np.asarray(M.sum(0)).ravel() + 1
        co = (M.T @ M).toarray() + 1
        s = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                s.append(np.log(co[i, j] * N / (df[i] * df[j])))
        scores.append(np.mean(s) if s else 0)
    return float(np.mean(scores))


def stability(X, vocab, n_topics: int, seeds=(0, 1, 2), top: int = 8) -> float:
    """Mean best-match Jaccard overlap of top terms between seeds."""
    runs = [fit_topics(X, vocab, n_topics, s, top)[2] for s in seeds]
    js = []
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            for ta in runs[a]:
                js.append(max(len(set(ta) & set(tb)) / len(set(ta) | set(tb)) for tb in runs[b]))
    return float(np.mean(js))


def choose_k(X, vocab, ks=(4, 6, 8, 10, 12)) -> pd.DataFrame:
    vi = {t: i for i, t in enumerate(vocab)}
    rows = []
    for k in ks:
        _, _, terms = fit_topics(X, vocab, k)
        rows.append({"k": k, "coherence": coherence_proxy(X, vi, terms), "stability": stability(X, vocab, k)})
    return pd.DataFrame(rows)
