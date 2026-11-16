"""Two-stage pipeline: candidate generation from 4 CF models, XGBoost pairwise reranking."""

import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm
from xgboost import XGBRanker

from recsys2025.base_models import train_base_models

# Stage 1: how many items each generator contributes before fusion.
CANDIDATE_GENERATORS = {"SLIM": 100, "ItemKNN": 50, "RP3": 80, "IALS": 80}
RRF_K = 60  # reciprocal-rank-fusion constant
# Hand-tuned blend used as one reranker feature.
ENSEMBLE_WEIGHTS = {"SLIM": 0.6, "IALS": 0.2, "RP3": 0.1, "ItemKNN": 0.1}

DEFAULT_XGB_PARAMS = dict(
    objective="rank:pairwise",
    eval_metric="ndcg@20",
    learning_rate=0.03,
    max_depth=6,
    n_estimators=5000,
    subsample=0.9,
    colsample_bytree=0.8,
    early_stopping_rounds=120,
    tree_method="hist",
    random_state=42,
)


def recall_at_k(recommendations: dict, URM_test: sps.csr_matrix, k: int = 20) -> float:
    """Mean over users with at least one test item of |hits in top-k| / |test items|."""
    URM_test = sps.csr_matrix(URM_test)
    recalls = []
    for u, items in recommendations.items():
        relevant = URM_test.indices[URM_test.indptr[u]:URM_test.indptr[u + 1]]
        if len(relevant):
            recalls.append(np.isin(items[:k], relevant).sum() / len(relevant))
    return float(np.mean(recalls)) if recalls else 0.0


class TwoStageRecommender:
    def __init__(self, n_candidates=150, batch_size=1000, xgb_params=None, base_params=None, verbose=True):
        self.n_candidates = n_candidates
        self.batch_size = batch_size
        self.xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.base_params = base_params or {}
        self.verbose = verbose
        self.ranker = None
        self.feature_cols = None

    # ------------------------------------------------------------------ base models
    def fit_base(self, URM):
        """(Re)train every first-stage model on URM. Call with the full data before submitting."""
        self.URM = sps.csr_matrix(URM, dtype=np.float32)
        self.models = train_base_models(self.URM, self.base_params)
        self.item_pop = np.ediff1d(self.URM.tocsc().indptr).astype(np.float32)
        self.user_len = np.ediff1d(self.URM.indptr).astype(np.float32)
        self.top_pop = np.argsort(-self.item_pop, kind="stable")
        return self

    # ------------------------------------------------------- candidates + features
    def build_features(self, users):
        """One row per (user, candidate item). Used identically for training and inference."""
        users = np.asarray(users)
        n_items = self.URM.shape[1]
        names = list(self.models)
        chunks = []
        batches = range(0, len(users), self.batch_size)
        for start in tqdm(batches, disable=not self.verbose, desc="candidates+features"):
            batch = users[start:start + self.batch_size]
            seen = self.URM[batch]
            rows_seen = np.repeat(np.arange(len(batch)), np.ediff1d(seen.indptr))
            scores = {}
            for m in names:
                s = np.asarray(self.models[m]._compute_item_score(batch), dtype=np.float32)
                s[rows_seen, seen.indices] = -np.inf
                scores[m] = s

            # Reciprocal rank fusion over the generators, then keep the best n_candidates.
            fused = np.zeros((len(batch), n_items), dtype=np.float32)
            rows = np.arange(len(batch))[:, None]
            for m, k in CANDIDATE_GENERATORS.items():
                top = np.argpartition(-scores[m], k - 1, axis=1)[:, :k]
                top = np.take_along_axis(top, np.argsort(-scores[m][rows, top], axis=1), axis=1)
                fused[rows, top] += 1.0 / (RRF_K + np.arange(1, k + 1, dtype=np.float32))
            fused[rows_seen, seen.indices] = 0.0
            n = min(self.n_candidates, n_items - 1)
            cand = np.argpartition(-fused, n - 1, axis=1)[:, :n]
            r_idx, c_idx = np.nonzero(fused[rows, cand] > 0)
            items = cand[r_idx, c_idx]

            df = pd.DataFrame({"UserID": batch[r_idx], "ItemID": items, "rrf": fused[r_idx, items]})
            for m in names:
                df[f"S_{m}"] = scores[m][r_idx, items]
            chunks.append(df)

        df = pd.concat(chunks, ignore_index=True)
        by_user = df.groupby("UserID", sort=False)
        for m in names:
            col = f"S_{m}"
            df[f"R_{m}"] = by_user[col].rank(ascending=False, method="first").astype(np.float32)
            std = by_user[col].transform("std").fillna(0).replace(0, 1)
            df[f"Z_{m}"] = ((df[col] - by_user[col].transform("mean")) / std).astype(np.float32)
        df["item_pop"] = self.item_pop[df["ItemID"].to_numpy()]
        df["log_item_pop"] = np.log1p(df["item_pop"])
        df["user_len"] = self.user_len[df["UserID"].to_numpy()]
        df["log_user_len"] = np.log1p(df["user_len"])
        df["S_ensemble"] = sum(w * df[f"S_{m}"] for m, w in ENSEMBLE_WEIGHTS.items())
        df["S_diff_SLIM_IALS"] = df["S_SLIM"] - df["S_IALS"]
        df["P_SLIM"] = by_user["S_SLIM"].rank(pct=True).astype(np.float32)
        df["P_IALS"] = by_user["S_IALS"].rank(pct=True).astype(np.float32)
        return df.fillna(0)

    # --------------------------------------------------------------------- reranker
    def fit(self, URM_train, URM_labels, val_fraction=0.15, seed=42):
        """Train base models on URM_train, label their candidates with URM_labels, fit XGBRanker."""
        self.fit_base(URM_train)
        URM_labels = sps.csr_matrix(URM_labels)
        users = np.flatnonzero((np.ediff1d(URM_labels.indptr) > 0) & (self.user_len > 0))
        df = self.build_features(users)

        n_items = self.URM.shape[1]
        lab = URM_labels.tocoo()
        positives = lab.row.astype(np.int64) * n_items + lab.col
        df["Label"] = np.isin(df["UserID"].to_numpy(np.int64) * n_items + df["ItemID"], positives).astype(np.float32)
        self.candidate_recall = df["Label"].sum() / URM_labels[users].nnz
        # Users without any positive candidate give no pairwise signal.
        df = df[df.groupby("UserID")["Label"].transform("max") > 0]

        self.feature_cols = [c for c in df.columns if c not in ("UserID", "ItemID", "Label")]
        rng = np.random.default_rng(seed)
        all_users = df["UserID"].unique()
        val_users = rng.choice(all_users, int(len(all_users) * val_fraction), replace=False)
        is_val = df["UserID"].isin(val_users).to_numpy()
        tr, va = df[~is_val], df[is_val]

        self.ranker = XGBRanker(**self.xgb_params)
        self.ranker.fit(
            tr[self.feature_cols], tr["Label"], qid=tr["UserID"],
            eval_set=[(va[self.feature_cols], va["Label"])], eval_qid=[va["UserID"]],
            verbose=100 if self.verbose else False,
        )
        if self.verbose:
            print(f"candidate recall: {self.candidate_recall:.4f}  "
                  f"best ndcg@20: {self.ranker.best_score:.4f} at iteration {self.ranker.best_iteration}")
        return self

    def recommend(self, users, cutoff=20, chunk_size=5000):
        """Return {user: [cutoff item ids]}; users unknown to URM or with few candidates get TopPop padding."""
        users = np.asarray(users)
        known = users[(users < self.URM.shape[0])]
        known = known[self.user_len[known] > 0]
        recs = {}
        for start in range(0, len(known), chunk_size):
            df = self.build_features(known[start:start + chunk_size])
            df["pred"] = self.ranker.predict(df[self.feature_cols])
            df = df.sort_values(["UserID", "pred"], ascending=[True, False])
            for u, items in df.groupby("UserID", sort=False)["ItemID"]:
                recs[int(u)] = items.to_numpy()[:cutoff].tolist()

        out = {}
        for u in users:
            items = recs.get(int(u), [])
            if len(items) < cutoff:
                seen = set(self.URM[u].indices) if u < self.URM.shape[0] else set()
                pad = (i for i in self.top_pop if i not in seen and i not in items)
                items = items + [int(next(pad)) for _ in range(cutoff - len(items))]
            out[int(u)] = items
        return out
