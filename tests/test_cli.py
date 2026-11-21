import json
import re

from typer.testing import CliRunner

from trialdesign import __version__
from trialdesign.cli import app

runner = CliRunner()


def test_version_flag_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"trialdesign {__version__}"


def test_design_survival_json() -> None:
    result = runner.invoke(app, ["design", "--endpoint", "survival", "--hr", "0.7", "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["events_ceil"] == 247
    assert out["alpha_two_sided"] == 0.05
    assert out["version"] == __version__


def test_design_binary_table_states_sidedness() -> None:
    result = runner.invoke(app, ["design", "--endpoint", "binary", "--pc", "0.3", "--pt", "0.45"])
    assert result.exit_code == 0, result.output
    assert "two-sided" in result.output
    assert re.search(r"n_control_ceil\s+163\n", result.output)


def test_design_normal_t_method() -> None:
    args = ["design", "--endpoint", "normal", "--delta", "0.5", "--sd", "1", "--method", "t"]
    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["n_control_ceil"] == 64


def test_design_missing_parameter_fails_early() -> None:
    result = runner.invoke(app, ["design", "--endpoint", "binary", "--pc", "0.3"])
    assert result.exit_code != 0
    assert "--pt is required" in result.output


def test_design_invalid_value_fails_with_plain_message() -> None:
    result = runner.invoke(app, ["design", "--endpoint", "survival", "--hr", "1.0"])
    assert result.exit_code == 2
    assert "hazard_ratio must be positive and not equal to 1" in result.output
