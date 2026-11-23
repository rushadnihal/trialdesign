import json
import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from trialdesign import __version__, bayes
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
    assert result.exit_code == 2
    assert "Error: --pt is required for the binary endpoint" in result.output


def test_design_invalid_value_fails_with_plain_message() -> None:
    result = runner.invoke(app, ["design", "--endpoint", "survival", "--hr", "1.0"])
    assert result.exit_code == 2
    assert "hazard_ratio must be positive and not equal to 1" in result.output


# ------------------------------------------------------------------------------ M5

EXAMPLES = Path(__file__).parent.parent / "examples"
SURVIVAL = str(EXAMPLES / "phase3_survival.toml")
BAYES = str(EXAMPLES / "phase2_binary_bayes.toml")
REF = {
    c["id"]: c
    for c in json.loads((Path(__file__).parent / "reference_values.json").read_text())["boundaries"]
}
PNG = b"\x89PNG\r\n\x1a\n"


def _json(args: list[str]) -> dict:
    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


@pytest.mark.parametrize("spending", ["obf", "pocock"])
def test_design_group_sequential_matches_r(spending: str) -> None:
    out = _json(
        ["design", "--endpoint", "survival", "--hr", "0.7", "--looks", "3", "--spending", spending]
    )
    ref = REF[f"{spending}_k3"]
    np.testing.assert_allclose(out["efficacy_z"], ref["efficacy"], atol=1e-3)
    assert out["inflation_factor"] == pytest.approx(ref["inflation_factor"], abs=1e-3)
    events = REF[f"max_events_hr0.7_{spending}_k3"]["events_max"]
    assert out["sample_size"]["max_events"] == pytest.approx(events, abs=0.05)
    assert out["alpha_one_sided"] == 0.025
    assert out["fixed"]["alpha_two_sided"] == 0.05


def test_design_one_look_is_the_fixed_design() -> None:
    out = _json(["design", "--endpoint", "survival", "--hr", "0.7", "--looks", "1"])
    assert out["efficacy_z"][0] == pytest.approx(1.959964, abs=1e-5)
    assert out["inflation_factor"] == pytest.approx(1.0, abs=1e-6)


def test_design_config_futility_matches_gsdesign() -> None:
    out = _json(["design", "--config", SURVIVAL, "--looks", "3"])
    ref = REF["futility_obf_pocock_k3"]
    np.testing.assert_allclose(out["futility_z"], ref["futility"], atol=1e-3)
    assert out["inflation_factor"] == pytest.approx(ref["inflation_factor"], abs=1e-3)
    assert out["binding_futility"] is False


def test_design_flags_override_config() -> None:
    out = _json(["design", "--config", SURVIVAL, "--hr", "0.8", "--futility", "none"])
    assert out["hazard_ratio"] == 0.8
    assert out["futility_z"] is None
    assert out["looks"] == [0.33, 0.67, 1.0]


def test_two_sided_alpha_flag_is_converted() -> None:
    base = ["design", "--endpoint", "binary", "--pc", "0.3", "--pt", "0.45", "--looks", "2"]
    one = _json([*base, "--alpha-one-sided", "0.025"])
    two = _json([*base, "--alpha-two-sided", "0.05"])
    assert one["efficacy_z"] == two["efficacy_z"]
    assert one["sample_size"] == two["sample_size"]


def test_design_group_sequential_table() -> None:
    result = runner.invoke(app, ["design", "--config", SURVIVAL])
    assert result.exit_code == 0, result.output
    assert "group-sequential design | alpha is one-sided" in result.output
    assert __version__ in result.output
    assert re.search(r"max_events\s+307\n", result.output)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--looks", "0.5,0.9"], "looks must be positive, increase and end at 1.0"),
        (["--looks", "3", "--spending", "haybittle"], 'efficacy: spending must be "obf"'),
        (["--futility", "obf"], "--looks is required for a group-sequential design"),
        (["--alpha-one-sided", "0.025", "--alpha-two-sided", "0.05"], "not both"),
    ],
)
def test_design_invalid_sequential_input(args: list[str], message: str) -> None:
    result = runner.invoke(app, ["design", "--endpoint", "survival", "--hr", "0.7", *args])
    assert result.exit_code == 2
    assert message in result.output


def test_binding_futility_needs_futility(tmp_path: Path) -> None:
    p = tmp_path / "d.toml"
    p.write_text(
        '[design]\nendpoint = "survival"\nhr = 0.7\n'
        "[sequential]\nlooks = 3\nbinding_futility = true\n"
    )
    result = runner.invoke(app, ["design", "--config", str(p)])
    assert result.exit_code == 2
    assert "binding futility needs a futility spending" in result.output


def test_design_without_endpoint_fails() -> None:
    result = runner.invoke(app, ["design"])
    assert result.exit_code == 2
    assert "--endpoint is required" in result.output


def test_bad_config_fails_with_plain_message(tmp_path: Path) -> None:
    p = tmp_path / "d.toml"
    p.write_text("[design]\nalpha = 0.05\n")
    result = runner.invoke(app, ["design", "--config", str(p)])
    assert result.exit_code == 2
    assert "use alpha_one_sided or alpha_two_sided" in result.output


def test_simulate_is_reproducible_and_close_to_exact() -> None:
    args = ["simulate", "--config", SURVIVAL, "--nsim", "50000", "--seed", "11"]
    a, b = _json(args), _json(args)
    assert a == b
    sim = a["simulation"]
    assert sim["seed"] == 11 and sim["nsim"] == 50_000
    assert abs(sim["type_i_error"] - 0.025) < 3 * sim["type_i_error_se"]
    assert abs(sim["power"] - a["design"]["exact"]["power"]) < 3 * sim["power_se"]


def test_simulate_prints_seed_and_version() -> None:
    result = runner.invoke(app, ["simulate", "--config", SURVIVAL, "--nsim", "2000"])
    assert result.exit_code == 0, result.output
    header = result.output.splitlines()[0]
    assert header.startswith(f"trialdesign {__version__} | seed 2026 | 2,000 simulated trials")
    assert "z vs exact" in result.output


def test_simulate_draws_and_prints_a_seed() -> None:
    args = ["simulate", "--endpoint", "survival", "--hr", "0.7", "--looks", "2", "--nsim", "1000"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    seed = int(re.search(r"\| seed (\d+) \|", result.output).group(1))
    again = _json([*args, "--seed", str(seed)])
    first = _json([*args, "--seed", str(seed)])
    assert again == first


def test_simulate_needs_looks() -> None:
    result = runner.invoke(app, ["simulate", "--endpoint", "survival", "--hr", "0.7"])
    assert result.exit_code == 2
    assert "give --looks" in result.output


def test_bayes_flags() -> None:
    args = [
        "bayes",
        "--pc",
        "0.3",
        "--pt",
        "0.45",
        "--n",
        "60",
        "--interim",
        "30",
        "--go",
        "0.9",
        "--nogo",
        "0.1",
        "--xc",
        "6",
        "--xt",
        "12",
    ]
    out = _json(args)["bayes"]
    b = bayes.bayesian_interim_design(30, 60)
    assert out["h1"]["prob_success"] == pytest.approx(
        b.operating_characteristics(0.3, 0.45).prob_success
    )
    assert out["interim"]["posterior_prob_superiority"] == pytest.approx(0.951, abs=0.002)
    assert out["interim"]["ppos"] == pytest.approx(b.ppos(6, 12))
    result = runner.invoke(app, args)
    assert "exact (no simulation)" in result.output
    assert f"decision: {b.decide(6, 12)}" in result.output


def test_bayes_config_with_simulation_and_assurance() -> None:
    out = _json(["bayes", "--config", BAYES, "--nsim", "20000"])
    s, sim = out["bayes"], out["simulation"]
    assert sim["seed"] == 2026
    for key in ("h0", "h1", "assurance"):
        assert (
            abs(sim[key]["prob_success"] - s[key]["prob_success"]) < 4 * sim[key]["prob_success_se"]
        )
    assert s["assurance"]["design_prior_treatment"] == [4.5, 5.5]


def test_bayes_flags_override_config() -> None:
    out = _json(["bayes", "--config", BAYES, "--success", "0.99", "--prior-t", "2,3"])
    assert out["bayes"]["success_threshold"] == 0.99
    assert out["bayes"]["prior_treatment"] == [2.0, 3.0]
    assert out["simulation"]["nsim"] == 200_000  # [simulation] in the file still applies


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--pc", "0.3", "--pt", "0.45", "--interim", "30"], "--n is required"),
        (
            ["--pc", "0.3", "--pt", "0.45", "--n", "60", "--interim", "30", "--xc", "4"],
            "give both xc and xt",
        ),
        (
            ["--pc", "0.3", "--pt", "0.45", "--n", "60", "--interim", "30", "--prior-c", "1"],
            'prior_c must be a Beta prior "a,b"',
        ),
        (
            ["--pc", "0.3", "--pt", "0.45", "--n", "20", "--interim", "30"],
            "need 1 <= n_interim < n_final",
        ),
    ],
)
def test_bayes_invalid_input(args: list[str], message: str) -> None:
    result = runner.invoke(app, ["bayes", *args])
    assert result.exit_code == 2
    assert message in result.output


def test_report_without_out_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["report", "--config", BAYES])
    assert result.exit_code == 0, result.output
    assert "## Bayesian go/no-go interim" in result.output
    assert "add --out DIR for the plots" in result.output
    assert list(tmp_path.iterdir()) == []


def test_report_group_sequential(tmp_path: Path) -> None:
    out = tmp_path / "results"
    result = runner.invoke(
        app, ["report", "--config", SURVIVAL, "--nsim", "20000", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert {p.name for p in out.iterdir()} == {"report.md", "boundaries.png", "oc.png"}
    for name in ("boundaries.png", "oc.png"):
        assert (out / name).read_bytes()[:8] == PNG
    md = (out / "report.md").read_text()
    assert f"trialdesign {__version__}" in md and "seed 2026" in md
    assert "![Stopping boundaries](boundaries.png)" in md
    assert "Futility is non-binding" in md
    assert "| Type I error (futility ignored) | 0.0250 |" in md


def test_report_bayes(tmp_path: Path) -> None:
    out = tmp_path / "r"
    result = runner.invoke(
        app, ["report", "--config", BAYES, "--xc", "6", "--xt", "12", "--out", str(out), "--json"]
    )
    assert result.exit_code == 0, result.output
    files = json.loads(result.output)["files"]
    assert [Path(f).name for f in files] == ["report.md", "bayes_oc.png", "decision_map.png"]
    md = (out / "report.md").read_text()
    assert "### Interim decision rule" in md and "decision **" in md
    assert "above the conventional" not in md  # the calibrated default controls alpha


def test_report_from_flags(tmp_path: Path) -> None:
    out = tmp_path / "r"
    args = [
        "report",
        "--endpoint",
        "normal",
        "--delta",
        "0.5",
        "--sd",
        "1",
        "--looks",
        "2",
        "--out",
        str(out),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "no simulation (exact results only)" in (out / "report.md").read_text()


def test_report_needs_something() -> None:
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 2
    assert "nothing to report" in result.output


def test_bayes_default_is_calibrated_and_old_default_is_flagged() -> None:
    base = ["bayes", "--pc", "0.3", "--pt", "0.45", "--n", "60", "--interim", "30"]
    assert _json(base)["bayes"]["max_type_i_error"] <= 0.025
    result = runner.invoke(app, [*base, "--success", "0.95"])
    assert "above the conventional one-sided 0.025" in result.output


def test_bayes_calibrate_flag() -> None:
    base = ["bayes", "--pc", "0.3", "--pt", "0.45", "--n", "40", "--interim", "20"]
    out = _json([*base, "--calibrate"])["bayes"]
    assert out["success_threshold"] == pytest.approx(bayes.calibrate_success_threshold(20, 40))
    assert out["success_calibrated_to"] == 0.025
    assert out["max_type_i_error"] <= 0.025
    strict = _json([*base, "--calibrate", "--alpha-one-sided", "0.01"])["bayes"]
    assert strict["max_type_i_error"] <= 0.01 < out["max_type_i_error"]
    result = runner.invoke(app, [*base, "--calibrate", "--success", "0.9"])
    assert result.exit_code == 2
    assert "give success or calibrate, not both" in result.output


def test_deliberate_calibration_to_a_looser_alpha_is_not_flagged() -> None:
    args = [
        "bayes",
        "--pc",
        "0.3",
        "--pt",
        "0.45",
        "--n",
        "60",
        "--interim",
        "30",
        "--calibrate",
        "--alpha-one-sided",
        "0.10",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "calibrated: worst-case type I error <= 0.1 one-sided" in result.output
    assert "above the conventional" not in result.output
