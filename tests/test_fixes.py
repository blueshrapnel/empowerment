"""Capacities against analytic values, and exact deterministic empowerment."""

import numpy as np
import pytest

from empowerment import compute_empowerment
from empowerment.information_theory import blahut_arimoto


def binary_entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


@pytest.mark.parametrize("error", [0.05, 0.1, 0.25])
def test_binary_symmetric_channel(error):
    P = np.array([[1 - error, error], [error, 1 - error]])
    assert blahut_arimoto(P, epsilon=1e-9) == pytest.approx(1 - binary_entropy(error), abs=1e-5)


@pytest.mark.parametrize("n", [2, 3, 8])
def test_symmetric_channel_capacity_is_log_n_minus_row_entropy(n):
    P = 0.7 * np.eye(n) + 0.3 / n
    h = -np.sum(P[:, 0] * np.log2(P[:, 0]))
    assert blahut_arimoto(P, epsilon=1e-9) == pytest.approx(np.log2(n) - h, abs=1e-5)


def line_world(n_states, n_actions=4):
    """Action 0 moves right, action 1 left, the others stay. T[s', a, s]."""
    T = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        T[min(s + 1, n_states - 1), 0, s] = 1
        T[max(s - 1, 0), 1, s] = 1
        for a in range(2, n_actions):
            T[s, a, s] = 1
    return T


def test_deterministic_empowerment_is_exact_beyond_the_old_sampling_limit():
    # 4 ** 12 sequences would have been sampled. From the left end, 13 states are reachable in 12 steps.
    assert compute_empowerment(line_world(30), True, 12, 0) == pytest.approx(np.log2(13))
