"""Loading the competition CSVs, splitting, and writing submissions."""

import numpy as np
import pandas as pd
import scipy.sparse as sps


def load_urm(train_csv: str) -> sps.csr_matrix:
    """Read data_train.csv (user_id, item_id, data) into a binary user-rating matrix."""
    df = pd.read_csv(train_csv)
    # Some versions of the file use row/col headers.
    df = df.rename(columns={"row": "user_id", "col": "item_id"})
    users = df["user_id"].to_numpy()
    items = df["item_id"].to_numpy()
    urm = sps.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (users, items)),
        shape=(users.max() + 1, items.max() + 1),
    )
    urm.data[:] = 1.0  # collapse duplicate interactions
    return urm


def load_target_users(target_csv: str) -> np.ndarray:
    return pd.read_csv(target_csv)["user_id"].to_numpy()


def holdout_split(urm: sps.csr_matrix, train_fraction: float = 0.8, seed: int = 42):
    """Random global holdout over interactions, same protocol as the competition split."""
    coo = urm.tocoo()
    rng = np.random.default_rng(seed)
    in_train = rng.random(coo.nnz) < train_fraction

    def build(mask):
        return sps.csr_matrix(
            (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=urm.shape, dtype=np.float32
        )

    return build(in_train), build(~in_train)


def write_submission(recommendations: dict, path: str) -> None:
    """recommendations: {user_id: list of 20 item ids}."""
    rows = [
        {"user_id": int(u), "item_list": " ".join(map(str, items))}
        for u, items in recommendations.items()
    ]
    pd.DataFrame(rows, columns=["user_id", "item_list"]).to_csv(path, index=False)
