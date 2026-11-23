"""M10: Hwang-Shih-DeCani, Kim-DeMets and user-specified spending, and binding futility.

Boundaries, inflation factors and expected sample sizes are checked against
``rpact::getDesignGroupSequential`` and ``getDesignCharacteristics``
(tests/spending_reference_values.json, written by tests/r/spending_reference_values.R).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from trialdesign import boundaries, simulate, spending, ssr
from trialdesign.cli import app

REF = json.loads((Path(__file__).parent / "spending_reference_values.json").read_text())["cases"]
runner = CliRunner()
TOL = 1e-5


def _design(case: dict) -> boundaries.GroupSequentialDesign:
    return boundaries.group_sequential_design(
        case["looks"],
        alpha_one_sided=case["alpha_one_sided"],
        power=case["power"],
        efficacy_spending=case["efficacy_spending"],
        futility_spending=case["futility_spending"],
        binding_futility=case["binding_futility"],
    )


# ------------------------------------------------------------------------- against rpact


@pytest.mark.parametrize("case", REF, ids=lambda c: c["id"])
def test_cumulative_alpha_matches_rpact(case: dict) -> None:
    got = spending.cumulative_spending(
        case["efficacy_spending"], case["looks"], case["alpha_one_sided"]
    )
    np.testing.assert_allclose(got, case["alpha_spent"], rtol=0, atol=1e-7)


@pytest.mark.parametrize("case", REF, ids=lambda c: c["id"])
def test_boundaries_match_rpact(case: dict) -> None:
    d = _design(case)
    np.testing.assert_allclose(d.efficacy, case["efficacy_z"], rtol=0, atol=TOL)
    if case["futility_z"] is None:
        assert d.futility is None
    else:
        np.testing.assert_allclose(d.futility[:-1], case["futility_z"], rtol=0, atol=TOL)
        assert d.futility[-1] == d.efficacy[-1]


@pytest.mark.parametrize("case", REF, ids=lambda c: c["id"])
def test_inflation_and_expected_size_match_rpact(case: dict) -> None:
    d = _design(case)
    assert d.inflation_factor == pytest.approx(case["inflation_factor"], abs=TOL)
    # rpact's average sample numbers are relative to the fixed design, futility followed.
    assert d.expected_info_fraction(0.0) * d.inflation_factor == pytest.approx(
        case["asn_h0"], abs=TOL
    )
    assert d.expected_info_fraction() * d.inflation_factor == pytest.approx(case["asn_h1"], abs=TOL)


# ------------------------------------------------------------------- spending functions


def test_hsd_gamma_zero_and_kd_rho_one_are_linear() -> None:
    t = np.array([0.0, 0.2, 0.5, 1.0])
    np.testing.assert_allclose(spending.spending_hsd(t, 0.025, 0.0), 0.025 * t)
    np.testing.assert_allclose(spending.spending_kd(t, 0.025, 1.0), 0.025 * t)
    # Small gamma approaches the linear limit continuously.
    np.testing.assert_allclose(spending.spending_hsd(t, 0.025, 1e-8), 0.025 * t, atol=1e-10)


@pytest.mark.parametrize("spec", ["hsd:-4", "hsd:1", "kd:3", "kd:0.5"])
def test_parametric_spending_is_a_spending_function(spec: str) -> None:
    f = spending.get_spending_function(spec)
    t = np.linspace(0, 1, 101)
    out = f(t, 0.025)
    assert out[0] == 0.0
    assert out[-1] == pytest.approx(0.025, abs=1e-15)
    assert np.all(np.diff(out) > 0)


def test_hsd_closed_form() -> None:
    # Hwang, Shih & DeCani (1990): alpha (1 - exp(-gamma t)) / (1 - exp(-gamma)).
    expected = 0.025 * (1 - math.exp(2.0)) / (1 - math.exp(4.0))
    assert spending.spending_hsd(0.5, 0.025, -4.0) == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize(
    ("spec", "label", "name"),
    [
        ("OBF", "obf", "Lan-DeMets O'Brien-Fleming-type"),
        ("hsd:-4.0", "hsd:-4", "Hwang-Shih-DeCani (gamma = -4)"),
        (" kd : 2.5 ", "kd:2.5", "Kim-DeMets (rho = 2.5)"),
        ("user:0.10,0.4,1", "user:0.1,0.4,1", "user-specified (cumulative fractions 0.1, 0.4, 1)"),
    ],
)
def test_spending_strings_round_trip(spec: str, label: str, name: str) -> None:
    parsed = spending.parse_spending(spec)
    assert parsed.label == label
    assert parsed.name == name
    assert spending.parse_spending(parsed.label) == parsed


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("hsd", "needs a parameter"),
        ("kd:0", "rho must be positive"),
        ("hsd:nan", "gamma must be finite"),
        ("hsd:1,2", "spending must be"),
        ("obf:1", "takes no parameter"),
        ("user:0.5,0.2,1", "not decrease and end at 1"),
        ("user:0.5,0.9", "not decrease and end at 1"),
        ("linear", "spending must be"),
    ],
)
def test_bad_spending_strings(spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        spending.parse_spending(spec)


def test_user_spending_needs_one_fraction_per_look() -> None:
    with pytest.raises(ValueError, match="2 fractions but the design has 3 looks"):
        spending.cumulative_spending("user:0.5,1", 3, 0.025)
    with pytest.raises(ValueError, match="defined only at the looks"):
        spending.get_spending_function("user:0.5,1")


def test_user_spending_can_skip_a_look() -> None:
    # Spending nothing at look 1 means no efficacy stop there (an infinite boundary).
    d = boundaries.group_sequential_design(
        [0.5, 0.75, 1.0], efficacy_spending="user:0,0.4,1", futility_spending="obf"
    )
    assert d.efficacy[0] == math.inf
    up, _ = d.crossing_probabilities(0.0, use_futility=False)
    assert up[0] == 0.0
    assert up.sum() == pytest.approx(0.025, abs=1e-8)


def test_design_records_canonical_spending_labels() -> None:
    d = boundaries.group_sequential_design(
        3, efficacy_spending="HSD:-4.0", futility_spending="kd:2"
    )
    assert (d.efficacy_spending, d.futility_spending) == ("hsd:-4", "kd:2")


# ---------------------------------------------------------------------- binding futility


@pytest.mark.parametrize(("eff", "fut"), [("obf", "obf"), ("pocock", "hsd:-2"), ("kd:3", "kd:2")])
def test_binding_futility_properties(eff: str, fut: str) -> None:
    nb = boundaries.group_sequential_design(4, efficacy_spending=eff, futility_spending=fut)
    b = boundaries.group_sequential_design(
        4, efficacy_spending=eff, futility_spending=fut, binding_futility=True
    )
    assert b.binding_futility and not nb.binding_futility
    # Alpha is spent exactly when futility is followed...
    up, _ = b.crossing_probabilities(0.0, use_futility=True)
    assert up.sum() == pytest.approx(0.025, abs=1e-8)
    # ...and exceeded when it is ignored.
    assert b.crossing_probabilities(0.0, use_futility=False)[0].sum() > 0.025
    # Binding boundaries are no higher, and the trial is smaller.
    assert np.all(b.efficacy <= nb.efficacy + 1e-9)
    assert b.inflation_factor < nb.inflation_factor
    # Power is still the target with futility followed.
    assert b.crossing_probabilities()[0].sum() == pytest.approx(0.80, abs=1e-6)


def test_binding_needs_futility() -> None:
    with pytest.raises(ValueError, match="binding futility needs futility boundaries"):
        boundaries.group_sequential_design(3, binding_futility=True)


def test_ssr_refuses_binding_futility() -> None:
    gsd = boundaries.group_sequential_design(
        [0.5, 1.0], futility_spending="obf", binding_futility=True
    )
    with pytest.raises(ValueError, match="non-binding futility only"):
        ssr.SSRDesign(gsd)


def test_simulated_type_i_error_follows_binding_futility() -> None:
    gsd = boundaries.group_sequential_design(
        3, efficacy_spending="hsd:-4", futility_spending="kd:2", binding_futility=True
    )
    sim = simulate.simulate_design(gsd, n_sim=100_000, seed=2026)
    assert sim.type_i_error == sim.h0.reject_prob
    assert sim.type_i_error_se == sim.h0.reject_prob_se
    assert abs(sim.type_i_error - 0.025) < 3 * sim.type_i_error_se


# --------------------------------------------------------------------------------- CLI


def test_cli_design_with_new_spending_and_binding() -> None:
    args = ["design", "--endpoint", "normal", "--delta", "3", "--sd", "10", "--looks", "3"]
    args += ["--efficacy", "hsd:-4", "--futility", "kd:2", "--binding-futility", "--json"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["efficacy_spending"] == "hsd:-4"
    assert out["futility_spending"] == "kd:2"
    assert out["binding_futility"] is True
    ex = out["exact"]
    assert ex["type_i_error"] == pytest.approx(0.025, abs=1e-8)
    assert ex["type_i_error_futility_ignored"] > 0.025


def test_cli_table_names_spending_and_binding() -> None:
    args = ["design", "--endpoint", "normal", "--delta", "3", "--sd", "10", "--looks", "3"]
    result = runner.invoke(
        app, [*args, "--efficacy", "kd:3", "--futility", "obf", "--binding-futility"]
    )
    assert result.exit_code == 0, result.output
    assert "Kim-DeMets (rho = 3) alpha spending" in result.output
    assert "beta spending, binding" in result.output
    assert "Type I error if futility is ignored" in result.output


def test_config_binding_and_user_spending(tmp_path: Path) -> None:
    p = tmp_path / "d.toml"
    p.write_text(
        '[design]\nendpoint = "survival"\nhr = 0.7\n'
        '[sequential]\nlooks = [0.3, 0.6, 1.0]\nefficacy = "user:0.05,0.3,1"\n'
        'futility = "hsd:1"\nbinding_futility = true\n'
    )
    result = runner.invoke(app, ["design", "--config", str(p), "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["binding_futility"] is True
    case = next(c for c in REF if c["id"] == "user_hsd_b")
    np.testing.assert_allclose(out["efficacy_z"], case["efficacy_z"], atol=TOL)
    # The command line can switch binding off again.
    result = runner.invoke(app, ["design", "--config", str(p), "--non-binding-futility", "--json"])
    assert json.loads(result.output)["binding_futility"] is False


def test_cli_bad_spending_and_binding_messages(tmp_path: Path) -> None:
    base = ["design", "--endpoint", "normal", "--delta", "3", "--sd", "10", "--looks", "3"]
    result = runner.invoke(app, [*base, "--efficacy", "user:0.5,1"])
    assert result.exit_code == 2
    assert "2 fractions but the design has 3 looks" in result.output
    p = tmp_path / "d.toml"
    p.write_text(
        '[design]\nendpoint = "survival"\nhr = 0.7\n[sequential]\nlooks = 3\n'
        'futility = "obf"\nbinding_futility = "yes"\n'
    )
    result = runner.invoke(app, ["design", "--config", str(p)])
    assert result.exit_code == 2
    assert "binding_futility must be true or false" in result.output


def test_report_explains_binding(tmp_path: Path) -> None:
    p = tmp_path / "d.toml"
    p.write_text(
        '[design]\nendpoint = "normal"\ndelta = 3\nsd = 10\n'
        '[sequential]\nlooks = 3\nefficacy = "hsd:-4"\nfutility = "kd:2"\nbinding_futility = true\n'
    )
    result = runner.invoke(app, ["report", "--config", str(p)])
    assert result.exit_code == 0, result.output
    md = result.output
    assert "Futility is binding" in md
    assert "Hwang-Shih-DeCani (gamma = -4) alpha spending" in md
