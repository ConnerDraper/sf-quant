"""Run the docstring examples that can run.

Only modules whose examples need no data belong here. The data-layer loaders
document calls against the cluster's parquet store, so their examples show the
call and no output: there is nothing to execute off the cluster, and an
invented result is worse than none.
"""

from __future__ import annotations

import doctest

import pytest

import sf_quant.optimizer.mandate as mandate

RUNNABLE = [mandate]


@pytest.mark.parametrize("module", RUNNABLE, ids=lambda m: m.__name__)
def test_docstring_examples(module):
    results = doctest.testmod(module, verbose=False, report=True)
    assert results.attempted > 0, f"{module.__name__} has no runnable examples"
    assert results.failed == 0
