import datetime as dt

import pytest

from sf_backtester import BacktestConfig, ConfigError

BASE = {"name": "x", "start": "2010-01-01", "end": "2012-12-31"}


def test_defaults():
    cfg = BacktestConfig.from_dict(BASE)
    assert cfg.start == dt.date(2010, 1, 1)
    assert cfg.factor_control == "neutral"
    assert cfg.aum == 1_000_000
    assert cfg.bps_grid == [0.0, 5.0, 10.0, 20.0]


@pytest.mark.parametrize(
    "override",
    [
        {"nmae": "typo"},
        {"profile": "experimental"},  # the mandate is not configurable
        {"active_risk_target": 0.04},
        {"rebalance": "montly"},
        {"factor_control": "loose"},
        {"slurm": {"cpus": 4}},
        {"end": "2009-01-01"},
    ],
)
def test_bad_configs_raise(override):
    with pytest.raises(ConfigError):
        BacktestConfig.from_dict(BASE | override)


def test_resolved_config_round_trips_and_hash_ignores_compute():
    cfg = BacktestConfig.from_dict(BASE | {"slurm": {"n_cpus": 4}})
    assert BacktestConfig.from_dict(cfg.resolved()) == cfg
    assert cfg.hash() == BacktestConfig.from_dict(cfg.resolved() | {"n_workers": 8}).hash()
    assert cfg.hash() != BacktestConfig.from_dict(cfg.resolved() | {"bps_grid": [0, 7.5]}).hash()
