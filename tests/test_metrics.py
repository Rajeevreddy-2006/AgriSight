import numpy as np

from agrisight import metrics as M


def test_softmax_rows_sum_to_one():
    p = M.softmax(np.random.randn(5, 4))
    assert np.allclose(p.sum(1), 1)


def test_ece_zero_for_perfect_confident_predictions():
    labels = np.array([0, 1, 2, 1])
    probs = np.eye(3)[labels]
    assert M.ece(probs, labels) == 0
    assert M.accuracy(probs, labels) == 1


def test_ece_detects_overconfidence():
    labels = np.array([0, 0, 0, 0])
    probs = np.tile([0.05, 0.95], (4, 1))  # 95% confident, always wrong
    assert M.ece(probs, labels) > 0.9


def test_risk_coverage_ranks_errors_last():
    labels = np.array([0, 0, 0, 0])
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])  # wrong ones are least confident
    coverage, risk, aurc = M.risk_coverage(probs, labels)
    assert coverage[-1] == 1.0
    assert risk[0] == 0 and risk[1] == 0
    assert risk[-1] == 0.5
    sel = M.selective(probs, labels, threshold=0.75)
    assert sel["coverage"] == 0.5 and sel["accuracy_on_kept"] == 1.0 and sel["errors_caught"] == 1.0
