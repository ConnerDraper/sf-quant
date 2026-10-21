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


class FakeSbatch:
    """Stands in for subprocess.run; records each sbatch call, hands out ids."""

    def __init__(self):
        self.calls = []

    def __call__(self, args, **kwargs):
        import subprocess

        self.calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=f"{1000 + len(self.calls)};cluster\n", stderr="")


@pytest.fixture
def sbatch(monkeypatch):
    import sf_quant.slurm.slurm as slurm_module

    fake = FakeSbatch()
    monkeypatch.setattr(slurm_module.subprocess, "run", fake)
    return fake


def test_submit_returns_the_job_id_and_chains_dependencies(sbatch):
    from sf_quant.slurm import submit_job

    first = submit_job("#!/bin/bash\necho a\n")
    second = submit_job("#!/bin/bash\necho b\n", after=[first, None, "77"])

    assert (first, second) == ("1001", "1002")
    assert sbatch.calls[0][:2] == ["sbatch", "--parsable"]
    assert "--dependency=afterok:1001:77" in sbatch.calls[1]
    assert not any(a.startswith("--dependency") for a in sbatch.calls[0])


def test_dry_run_submits_nothing(sbatch, capsys):
    from sf_quant.slurm import submit_job

    assert submit_job("#!/bin/bash\necho a\n", dry_run=True, after=["5"]) is None
    assert sbatch.calls == []
    assert "--dependency=afterok:5" in capsys.readouterr().out


def test_sweep_queues_one_job_per_config(sbatch, tmp_path):
    from sf_quant.slurm import submit_sweep

    configs = [
        BacktestConfig(**{**BASE, "project_root": str(tmp_path)}, lag=lag, logs_dir=str(tmp_path / f"logs{lag}"))
        for lag in (0, 5)
    ]
    ids = submit_sweep(configs, after=["9"])
    assert ids == ["1001", "1002"]
    assert all("--dependency=afterok:9" in call for call in sbatch.calls)


def test_command_script_runs_the_command_in_the_workdir(tmp_path):
    from sf_quant.slurm.slurm import generate_command_script

    script = generate_command_script(
        "python -m study analyze", job_name="analyze", logs_dir=str(tmp_path), byu_email="me@byu.edu",
        slurm=SlurmConfig(n_cpus=2, mem="8G", time="00:10:00"), workdir="/home/me/study",
    )
    assert "#SBATCH --cpus-per-task=2" in script
    assert "#SBATCH --mem=8G" in script
    assert "cd /home/me/study\npython -m study analyze" in script
    assert "gamma" not in script.lower()


def test_scripts_export_the_submitting_sessions_data_lake(monkeypatch, tmp_path):
    import sf_quant.data._config as data_config
    from sf_quant.slurm.slurm import generate_command_script

    monkeypatch.setitem(data_config._config, "root", "/lake")
    monkeypatch.setitem(data_config._config, "database", "production")
    config = BacktestConfig(**BASE)
    for script in (
        generate_sbatch_script(config, config.years),
        generate_command_script("true", job_name="x", logs_dir=str(tmp_path), byu_email="me@byu.edu"),
    ):
        assert "export ROOT=/lake DATABASE=production" in script


def test_load_weights_names_missing_years(lake, tmp_path):
    import polars as pl
    from conftest import BARRIDS, DATES
    from sf_quant.slurm import MissingYearsError, expected_years, load_weights

    alphas = pl.DataFrame(
        {"date": [d for d in DATES for _ in BARRIDS], "barrid": BARRIDS * len(DATES), "alpha": 0.01}
    )
    alphas.write_parquet(tmp_path / "alphas.parquet")
    config = BacktestConfig(**{
        **BASE, "data_path": str(tmp_path / "alphas.parquet"), "project_root": str(tmp_path),
        "start": DATES[0], "end": DATES[-1],
    })
    assert expected_years(config) == [2023, 2024]

    out = tmp_path / "weights" / "idio_mom" / "0.05"
    out.mkdir(parents=True)
    alphas.filter(pl.col("date").dt.year() == 2023).write_parquet(out / "2023.parquet")
    with pytest.raises(MissingYearsError, match=r"\[2024\]"):
        load_weights(config)

    alphas.filter(pl.col("date").dt.year() == 2024).write_parquet(out / "2024.parquet")
    assert load_weights(config).height == alphas.height
