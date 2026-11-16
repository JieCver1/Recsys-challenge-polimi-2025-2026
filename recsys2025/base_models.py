"""First-stage collaborative-filtering models and their tuned hyperparameters.

Hyperparameters were tuned with Optuna on a train/validation holdout during the
competition (see README). All models except iALS come from the course framework.
"""

import os

import numpy as np
import scipy.sparse as sps

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender


class IALSRecommender:
    """Implicit ALS from the `implicit` library, exposing the framework's scoring interface."""

    RECOMMENDER_NAME = "IALSRecommender"

    def __init__(self, URM_train, verbose=False):
        self.URM_train = sps.csr_matrix(URM_train, dtype=np.float32)
        self.verbose = verbose

    def fit(self, factors=100, regularization=0.01, alpha=1.0, iterations=15, num_threads=0, random_seed=42):
        from implicit.cpu.als import AlternatingLeastSquares

        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            use_cg=True,
            num_threads=num_threads,
            dtype=np.float32,
            random_state=random_seed,
        )
        self.model.fit(self.URM_train, show_progress=self.verbose)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        scores = self.model.user_factors[np.atleast_1d(user_id_array)] @ self.model.item_factors.T
        return np.asarray(scores, dtype=np.float32)


# name -> (class, fit kwargs). Values found with Optuna during the competition.
BASE_MODELS = {
    "TopPop": (TopPop, {}),
    "SLIM": (MultiThreadSLIM_SLIMElasticNetRecommender, dict(
        l1_ratio=1.5190356919784374e-06, alpha=0.005419614628252354, topK=561, positive_only=True,
        workers=max(1, (os.cpu_count() or 2) // 2))),
    "RP3": (RP3betaRecommender, dict(
        topK=40, alpha=1.4692126186652583, beta=0.36750406285425163, normalize_similarity=True)),
    "EASER": (EASE_R_Recommender, dict(l2_norm=423.6711768137149, topK=649)),
    "IALS": (IALSRecommender, dict(
        factors=101, regularization=0.6025168879722137, alpha=9.465921675498674, iterations=31)),
    "ItemKNN": (ItemKNNCFRecommender, dict(
        topK=50, shrink=952, similarity="asymmetric", feature_weighting="TF-IDF",
        asymmetric_alpha=0.024247509583039298)),
    "UserKNN": (UserKNNCFRecommender, dict(
        topK=362, shrink=109, similarity="asymmetric", feature_weighting="TF-IDF",
        asymmetric_alpha=0.3496302728900016)),
    "P3alpha": (P3alphaRecommender, dict(topK=155, alpha=1.3351300340675043, normalize_similarity=True)),
    "SLIMBPR": (SLIM_BPR_Cython, dict(
        topK=125, epochs=91, lambda_i=1.9719285330802604e-06, lambda_j=3.4367676913044208e-06,
        learning_rate=0.0821757946413263)),
    "PureSVD": (PureSVDRecommender, dict(num_factors=28)),
}


def train_base_models(URM, params=None, names=None, verbose=False):
    """Fit every base model on URM. `params` overrides the default fit kwargs per model."""
    params = params or {}
    models = {}
    for name in names or BASE_MODELS:
        cls, kwargs = BASE_MODELS[name]
        model = cls(URM) if cls is TopPop else cls(URM, verbose=verbose)
        model.fit(**{**kwargs, **params.get(name, {})})
        models[name] = model
    return models
