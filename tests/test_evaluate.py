"""SPEC M2: the LOSO leak guard and causal post-processing."""

import numpy as np
import pytest

from haptic_ai import evaluate


def test_leaked_subject_raises():
    with pytest.raises(evaluate.LeakError):
        evaluate.check_disjoint(np.array(["a", "b", "c"]), np.array(["c"]))


def test_loso_folds_are_disjoint():
    groups = np.repeat(["a", "b", "c"], 20)
    for tr, te in evaluate.loso_folds(groups):
        assert len(set(groups[tr]) & set(groups[te])) == 0
        assert len(set(groups[te])) == 1


def test_moving_average_is_causal_and_per_session():
    rng = np.random.default_rng(0)
    p = rng.random((10, 3))
    sessions = np.array(["s"] * 6 + ["t"] * 4)
    out = evaluate.moving_average(p, sessions, k=3)
    np.testing.assert_allclose(out[0], p[0])
    np.testing.assert_allclose(out[5], p[3:6].mean(0))
    np.testing.assert_allclose(out[6], p[6])  # a new session restarts the average
    p2 = p.copy()
    p2[5:] = 0  # changing later windows leaves earlier outputs alone
    np.testing.assert_allclose(evaluate.moving_average(p2, sessions, k=3)[:5], out[:5])


def test_macro_f1_uses_steps_1_to_5_only():
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    pred = np.array([6, 1, 2, 3, 4, 5, 0])  # NULL and WASH_OTHER swapped: not scored
    assert evaluate.macro_f1(y, pred) == 1.0
