"""Optimizer tests. No data lake needed."""

import types

import cvxpy as cp
import numpy as np
import pytest

from sf_quant.optimizer.optimizers import _check_inaccurate


def inaccurate(*constraints, status="optimal_inaccurate"):
    return types.SimpleNamespace(status=status, constraints=list(constraints))


def test_inaccurate_solution_within_tolerance_is_accepted():
    x = cp.Variable(2)
    x.value = np.array([0.5, -0.5 + 1e-9])
    _check_inaccurate(inaccurate(cp.sum(x) == 0, x >= -0.5))


def test_inaccurate_solution_breaking_a_constraint_is_rejected():
    x = cp.Variable(2)
    x.value = np.array([0.5, -0.5 + 1e-3])
    with pytest.raises(ValueError, match="violated by 1.0e-03"):
        _check_inaccurate(inaccurate(cp.sum(x) == 0))


def test_risk_constraint_is_judged_relative_to_its_bound():
    """Variance 1e-5 over a 0.05^2 target: 2.5e-8 absolute, 1e-5 relative."""
    x = cp.Variable(1)
    x.value = np.array([0.05 * np.sqrt(1 + 1e-5)])
    risk = cp.sum_squares(x) <= 0.05**2
    _check_inaccurate(inaccurate(risk))
    with pytest.raises(ValueError, match="inaccurate"):
        _check_inaccurate(inaccurate(risk), scale={risk.id: 0.05**2})


def test_optimal_solutions_are_not_rechecked():
    x = cp.Variable(2)
    x.value = np.array([1.0, 1.0])
    _check_inaccurate(inaccurate(cp.sum(x) == 0, status="optimal"))
