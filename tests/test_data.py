import numpy as np
import torch

import data.colagnn as colagnn
from data import ColaGNNData


def load(synthetic_dataset, horizon=5):
    colagnn.DATASETS["japan"] = ("japan.txt", "japan-adj.txt", 4)  # 4 synthetic regions instead of 47 prefectures
    return ColaGNNData("japan", synthetic_dataset, window=20, horizon=horizon)


def test_split_and_window_alignment(synthetic_dataset):
    d = load(synthetic_dataset, horizon=5)
    assert (d.n_train, d.n_val) == (80, 112)
    assert d.train_idx[0] == 20 + 5 - 1 and d.val_idx == list(range(80, 112)) and d.test_idx == list(range(112, 160))
    X, Y = d.test
    assert X.shape == (48, 20, 4) and Y.shape == (48, 4)
    i = d.test_idx[3]
    assert torch.allclose(X[3], torch.from_numpy(d.dat[i - 5 + 1 - 20:i - 5 + 1]))  # last observed step is i-h
    assert torch.allclose(Y[3], torch.from_numpy(d.dat[i]))


def test_normalisation_uses_training_rows_only(synthetic_dataset):
    d = load(synthetic_dataset)
    stat_rows = np.concatenate([d.raw[d.train_idx[0] - 5 + 1 - 20:][:20], d.raw[d.train_idx]], 0)
    assert np.allclose(d.max, stat_rows.max(0)) and np.allclose(d.min, stat_rows.min(0))
    assert np.allclose(d.denorm(d.dat), d.raw, atol=1e-3)


def test_stream_targets_match_window_protocol(synthetic_dataset):
    d = load(synthetic_dataset, horizon=3)
    x_all, sets = d.stream(3)
    assert x_all.shape == (1, 160, 4)
    pos, tgt = sets["test"]
    assert tgt.tolist() == d.test_idx and torch.equal(pos, tgt - 3)
    assert sets["val"][1].tolist() == d.val_idx
