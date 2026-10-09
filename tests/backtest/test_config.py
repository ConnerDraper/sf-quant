import datetime as dt

import pytest

from sf_backtester import BacktestConfig, ConfigError

BASE = {"name": "x", "start": "2010-01-01", "end": "2012-12-31"}


def test_defaults_are_the_silver_fund_mandate():
    cfg = BacktestConfig.from_dict(BASE)
    assert cfg.profile == "mandate"
    assert cfg.active_risk_target == 0.05
    assert cfg.start == dt.date(2010, 1, 1)
    assert cfg.aum == 1_000_000
    assert cfg.bps_grid == [0.0, 5.0, 10.0, 20.0]


def test_risk_target_is_locked_under_mandate_profile():
    with pytest.raises(ConfigError, match="locked"):
        BacktestConfig.from_dict(BASE | {"active_risk_target": 0.04})
    assert BacktestConfig.from_dict(BASE | {"profile": "experimental", "active_risk_target": 0.04})


@pytest.mark.parametrize(
    "override",
    [
        {"nmae": "typo"},
        {"rebalance": "montly"},
        {"on_failure": "skip"},
        {"slurm": {"cpus": 4}},
        {"end": "2009-01-01"},
        {"factor_bounds": {"USSLOWL_SIZE": -0.1}},
    ],
)
def test_bad_configs_raise(override):
    with pytest.raises(ConfigError):
        BacktestConfig.from_dict(BASE | override)


def test_resolved_config_round_trips_and_hash_ignores_compute():
    cfg = BacktestConfig.from_dict(BASE | {"factor_bounds": {"USSLOWL_SIZE": 0.1}, "slurm": {"n_cpus": 4}})
    assert BacktestConfig.from_dict(cfg.resolved()) == cfg
    assert cfg.hash() == BacktestConfig.from_dict(cfg.resolved() | {"n_workers": 8}).hash()
    assert cfg.hash() != BacktestConfig.from_dict(cfg.resolved() | {"bps_grid": [0, 7.5]}).hash()
