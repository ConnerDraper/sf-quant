import datetime as dt

import pytest

from sf_quant.slurm import BacktestConfig, SlurmConfig
from sf_quant.slurm.scripts.utils import get_constraints
from sf_quant.slurm.slurm import generate_sbatch_script

BASE = dict(
    signal_name="idio_mom",
    data_path="/tmp/idio_mom_alphas.parquet",
    start=dt.date(2020, 1, 1),
    end=dt.date(2022, 12, 31),
    project_root="/home/me/project",
    byu_email="me@byu.edu",
)


def test_defaults_are_the_fund_mandate():
    config = BacktestConfig(**BASE)
    assert config.constraints == ["ZeroInvestment", "ZeroBeta", "LongOnly"]
    assert config.target_active_risk == 0.05
    assert (config.lag, config.rebalance_period) == (0, 1)


def test_years_come_from_the_window_not_the_data():
    assert BacktestConfig(**BASE).years == [2020, 2021, 2022]


def test_baseline_run_has_no_path_suffix():
    config = BacktestConfig(**BASE)
    assert config.output_dir == "/home/me/project/weights/idio_mom/0.05"
    assert config.logs_dir == "logs/idio_mom/0.05"


def test_sweeps_write_to_separate_directories():
    a = BacktestConfig(**BASE, lag=5)
    b = BacktestConfig(**BASE, rebalance_period=21)
    assert a.output_dir.endswith("/idio_mom/0.05/lag5_rebal1")
    assert b.output_dir.endswith("/idio_mom/0.05/lag0_rebal21")
    assert a.output_dir != b.output_dir


def test_iso_date_strings_are_coerced():
    config = BacktestConfig(**{**BASE, "start": "2020-01-01", "end": "2022-12-31"})
    assert config.start == dt.date(2020, 1, 1)
    assert config.years == [2020, 2021, 2022]


def test_slurm_dict_is_coerced():
    config = BacktestConfig(**BASE, slurm={"n_cpus": 4, "mem": "16G"})
    assert isinstance(config.slurm, SlurmConfig)
    assert config.slurm.n_cpus == 4


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"start": dt.date(2023, 1, 1)}, "is after end"),
        ({"lag": -1}, "lag must be >= 0"),
        ({"rebalance_period": 0}, "rebalance_period must be >= 1"),
    ],
)
def test_invalid_config_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        BacktestConfig(**{**BASE, **bad})


def test_roundtrip_through_yaml(tmp_path):
    config = BacktestConfig(**BASE, lag=5, rebalance_period=21)
    path = tmp_path / "config.yml"
    config.to_yaml(path)
    assert BacktestConfig.from_yaml(path) == config


def test_sbatch_script_passes_the_window_and_sweep_settings():
    config = BacktestConfig(**BASE, lag=5, rebalance_period=21)
    script = generate_sbatch_script(config, config.years)

    assert "--array=0-2%31" in script
    assert "years=(2020 2021 2022)" in script
    assert "--start 2020-01-01" in script
    assert "--end 2022-12-31" in script
    assert "--lag 5" in script
    assert "--rebalance_period 21" in script
    assert 'CONSTRAINTS="ZeroInvestment ZeroBeta LongOnly"' in script
    # Gamma is gone from the whole stack.
    assert "gamma" not in script.lower()


def test_constraint_registry_resolves_names():
    assert [type(c).__name__ for c in get_constraints(["ZeroBeta", "LongOnly"])] == [
        "ZeroBeta",
        "LongOnly",
    ]


def test_unknown_constraint_name_is_rejected():
    with pytest.raises(KeyError, match="Unknown constraint 'UnitBeta'"):
        get_constraints(["UnitBeta"])
