"""Tests for optional L-BFGS diagonal preconditioner hook."""

import numpy as np

from dasfcn.precond import apply_diag_precond, lbfgs_diag_preconditioner


def test_lbfgs_diag_preconditioner_positive():
    s = np.array([1.0, 0.5, -0.2])
    y = np.array([0.8, 0.4, -0.1])
    d = lbfgs_diag_preconditioner([s], [y])
    assert d is not None
    assert np.all(d > 0)
    v = np.ones(3)
    w = apply_diag_precond(v, d)
    assert w.shape == (3,)
