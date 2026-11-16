"""End-to-end check on a small synthetic dataset with low-rank structure."""

import numpy as np
import pytest
import scipy.sparse as sps

from recsys2025.base_models import BASE_MODELS
from recsys2025.data import holdout_split
from recsys2025.two_stage import TwoStageRecommender, recall_at_k

FAST = {
    "SLIM": dict(topK=50), "EASER": dict(topK=50), "IALS": dict(iterations=5, factors=16),
    "UserKNN": dict(topK=50), "SLIMBPR": dict(epochs=5), "PureSVD": dict(num_factors=8),
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n_users, n_items, k = 500, 300, 6
    affinity = rng.normal(size=(n_users, k)) @ rng.normal(size=(k, n_items))
    affinity += 2.0 * np.log1p(np.arange(n_items))[::-1] / np.log(n_items)  # popularity skew
    affinity += rng.gumbel(scale=0.3, size=affinity.shape)
    urm = sps.csr_matrix((affinity > np.quantile(affinity, 0.88, axis=1, keepdims=True)).astype(np.float32))
    trainval, test = holdout_split(urm, 0.8, 1)
    train, val = holdout_split(trainval, 0.8, 2)
    return trainval, train, val, test


@pytest.fixture(scope="module")
def fitted(data):
    trainval, train, val, _ = data
    rec = TwoStageRecommender(n_candidates=60, batch_size=128, base_params=FAST,
                              xgb_params=dict(n_estimators=150, early_stopping_rounds=20), verbose=False)
    rec.fit(train, val)
    rec.fit_base(trainval)
    return rec


def test_all_base_models_trained(fitted):
    assert set(fitted.models) == set(BASE_MODELS)


def test_feature_columns_match_between_train_and_inference(fitted):
    df = fitted.build_features(np.arange(20))
    assert [c for c in df.columns if c not in ("UserID", "ItemID")] == fitted.feature_cols


def test_candidates_are_unseen_unique_and_capped(fitted):
    df = fitted.build_features(np.arange(50))
    assert df.groupby("UserID").size().max() <= 60
    assert not df.duplicated(["UserID", "ItemID"]).any()
    seen = fitted.URM[df["UserID"].to_numpy(), df["ItemID"].to_numpy()]
    assert np.asarray(seen).sum() == 0


def test_recommendations_shape_and_cold_users(fitted, data):
    trainval = data[0]
    cold = trainval.shape[0] + 5  # user id beyond the matrix
    recs = fitted.recommend(np.array([0, 1, 2, cold]))
    for u, items in recs.items():
        assert len(items) == 20 and len(set(items)) == 20
        if u < trainval.shape[0]:
            assert not set(items) & set(trainval[u].indices)
    assert recs[cold] == fitted.top_pop[:20].tolist()


def test_reranker_beats_its_own_base_models(fitted, data):
    test = data[3]
    users = np.flatnonzero(np.ediff1d(test.indptr) > 0)
    two_stage = recall_at_k(fitted.recommend(users), test)
    for name in ("TopPop", "SLIM", "EASER", "RP3"):
        single = dict(zip(users.tolist(), fitted.models[name].recommend(users, cutoff=20)))
        assert two_stage > recall_at_k(single, test), name
