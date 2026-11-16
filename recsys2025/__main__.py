"""Command line entry point.

    python -m recsys2025 evaluate --data-dir data/
    python -m recsys2025 submit   --data-dir data/ --out submission.csv
"""

import argparse
import os
import time

import numpy as np

from recsys2025.data import holdout_split, load_target_users, load_urm, write_submission
from recsys2025.two_stage import TwoStageRecommender, recall_at_k


def main():
    p = argparse.ArgumentParser(prog="recsys2025")
    p.add_argument("command", choices=["evaluate", "submit"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--n-candidates", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="XGBoost device, e.g. cuda")
    args = p.parse_args()

    t0 = time.time()
    URM_all = load_urm(os.path.join(args.data_dir, "data_train.csv"))
    rec = TwoStageRecommender(n_candidates=args.n_candidates, xgb_params={"device": args.device})

    if args.command == "evaluate":
        # 64% base-model training / 16% reranker labels / 20% test, mirroring the 80/20 competition split.
        URM_trainval, URM_test = holdout_split(URM_all, 0.8, args.seed)
        URM_train, URM_val = holdout_split(URM_trainval, 0.8, args.seed + 1)
        rec.fit(URM_train, URM_val)
        rec.fit_base(URM_trainval)  # same step as before submitting: refit stage 1 on more data
        users = np.flatnonzero(np.ediff1d(URM_test.indptr) > 0)
        recs = rec.recommend(users)
        slim = rec.models["SLIM"]
        slim_recs = {int(u): list(r) for u, r in zip(users, slim.recommend(users, cutoff=20))}
        print(f"Recall@20  two-stage: {recall_at_k(recs, URM_test):.5f}  "
              f"SLIM alone: {recall_at_k(slim_recs, URM_test):.5f}")
    else:
        URM_train, URM_val = holdout_split(URM_all, 0.8, args.seed)
        rec.fit(URM_train, URM_val)
        rec.fit_base(URM_all)
        targets = load_target_users(os.path.join(args.data_dir, "data_target_users_test.csv"))
        write_submission(rec.recommend(targets), args.out)
        print(f"Wrote {args.out}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
