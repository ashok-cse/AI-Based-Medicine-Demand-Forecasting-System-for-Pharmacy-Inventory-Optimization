"""Unit tests for preprocessing helpers that don't require a database."""
import numpy as np

from ml.preprocessing import make_sequences, SEASON_MAP


def test_make_sequences_shapes():
    series = np.arange(10, dtype="float32")  # 0..9
    X, y = make_sequences(series, seq_len=3)
    # 10 points, window 3 -> 7 samples
    assert X.shape == (7, 3)
    assert y.shape == (7,)
    # first window predicts the 4th element
    assert list(X[0]) == [0, 1, 2]
    assert y[0] == 3
    # last sample
    assert list(X[-1]) == [6, 7, 8]
    assert y[-1] == 9


def test_make_sequences_too_short():
    X, y = make_sequences(np.array([1.0, 2.0]), seq_len=5)
    assert X.shape[0] == 0
    assert y.shape[0] == 0


def test_season_map_covers_four_seasons():
    assert set(SEASON_MAP) == {"Winter", "Spring", "Summer", "Autumn"}
    assert sorted(SEASON_MAP.values()) == [0, 1, 2, 3]
