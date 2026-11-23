"""M7: non-inferiority margins, validated against rpact (tests/ni_reference_values.json).

The reference values come from ``tests/r/ni_reference_values.R``. Non-inferiority changes
only the fixed-design sample size (the effect under H0 and, for binary endpoints, the
variance under H0); the group-sequential maximum is still the fixed size times the inflation
factor, which the ``gsd`` cases check end to end.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize
from typer.testing import CliRunner

from trialdesign import fixed, report
from trialdesign.boundaries import group_sequential_design
from trialdesign.cli import app

REF = json.loads((Path(__file__).parent / "ni_reference_values.json").read_text())["ni"]
FIXED = [c for c in REF if c["kind"] == "fixed"]
GSD = [c for c in REF if c["kind"] == "gsd"]
EXAMPLES = Path(__file__).parents[1] / "examples"
runner = CliRunner()


def _fixed_size(case: dict) -> float:
    """Control-arm size (or events) for a reference case."""
    common = {
        "alpha_two_sided": 2 * case["alpha_one_sided"],
        "power": case["power"],
        "allocation": case["allocation"],
    }
    if case["endpoint"] == "normal":
        res = fixed.sample_size_normal(case["delta"], case["sd"], delta0=case["delta0"], **common)
        return res.n_control
    if case["endpoint"] == "binary":
        res = fixed.sample_size_binary(
            case["p_control"], case["p_treatment"], delta0=case["delta0"], **common
        )
        return res.n_control
    ev = fixed.events_survival(case["hazard_ratio"], hazard_ratio0=case["hazard_ratio0"], **common)
    return ev.events


@pytest.mark.parametrize("case", FIXED, ids=lambda c: c["id"])
def test_fixed_matches_rpact(case: dict) -> None:
    expected = case["n_control"] if "n_control" in case else case["events"]
    assert _fixed_size(case) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("case", GSD, ids=lambda c: c["id"])
def test_group_sequential_maximum_matches_rpact(case: dict) -> None:
    d = group_sequential_design(
        case["looks"],
        alpha_one_sided=case["alpha_one_sided"],
        power=case["power"],
        efficacy_spending=case["efficacy"],
    )
    if case["endpoint"] == "survival":
        assert d.max_sample_size(_fixed_size(case)) == pytest.approx(case["max_events"], abs=0.05)
    else:
        n_total = _fixed_size(case) * (1 + case["allocation"])
        assert d.max_sample_size(n_total) == pytest.approx(case["max_n_total"], abs=0.05)


# ------------------------------------------------------------ restricted MLE (binary)


def _restricted_mle_numeric(pc: float, pt: float, r: float, d0: float) -> tuple[float, float]:
    """Maximise the expected log-likelihood over p_C with p_T = p_C + d0, by brute force."""

    def negll(p: float) -> float:
        q = p + d0
        return -(pc * math.log(p) + (1 - pc) * math.log(1 - p)) - r * (
            pt * math.log(q) + (1 - pt) * math.log(1 - q)
        )

    lo, hi = max(0.0, -d0) + 1e-12, min(1.0, 1 - d0) - 1e-12
    p = optimize.minimize_scalar(negll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    return float(p.x), float(p.x + d0)


@pytest.mark.parametrize(
    ("pc", "pt", "r", "d0"),
    [
        (0.85, 0.85, 1.0, -0.10),
        (0.80, 0.85, 1.0, -0.10),
        (0.20, 0.20, 1.0, 0.10),
        (0.70, 0.65, 0.5, -0.15),
        (0.30, 0.45, 2.0, 0.05),
        (0.05, 0.08, 3.0, -0.02),
    ],
)
def test_farrington_manning_closed_form_maximises_likelihood(pc, pt, r, d0) -> None:
    pc0, pt0 = fixed.farrington_manning_rates(pc, pt, r, d0)
    assert pt0 - pc0 == pytest.approx(d0, abs=1e-12)
    num = _restricted_mle_numeric(pc, pt, r, d0)
    assert (pc0, pt0) == pytest.approx(num, abs=1e-6)


def test_farrington_manning_reduces_to_pooled_rate() -> None:
    pc0, pt0 = fixed.farrington_manning_rates(0.3, 0.45, 2.0, 0.0)
    assert pc0 == pt0 == pytest.approx((0.3 + 2 * 0.45) / 3)
    # and is continuous as delta0 -> 0
    near = fixed.farrington_manning_rates(0.3, 0.45, 2.0, 1e-9)
    assert near == pytest.approx((pc0, pt0), abs=1e-6)


# ----------------------------------------------------------------------- properties


def test_zero_margin_reproduces_superiority() -> None:
    kw = {"alpha_two_sided": 0.05, "power": 0.8, "allocation": 2.0}
    assert fixed.sample_size_normal(0.5, 1, delta0=0.0, **kw) == fixed.sample_size_normal(
        0.5, 1, **kw
    )
    assert fixed.sample_size_binary(0.3, 0.45, delta0=0.0, **kw) == fixed.sample_size_binary(
        0.3, 0.45, **kw
    )
    assert fixed.events_survival(0.7, hazard_ratio0=1.0, **kw) == fixed.events_survival(0.7, **kw)


@pytest.mark.parametrize("allocation", [1.0, 0.5, 2.0])
def test_power_inverts_sample_size(allocation: float) -> None:
    n = fixed.sample_size_normal(0.0, 10, delta0=-3, power=0.9, allocation=allocation)
    assert fixed.power_normal(
        n.n_control, 0.0, 10, delta0=-3, allocation=allocation
    ) == pytest.approx(0.9)
    n = fixed.sample_size_binary(0.8, 0.85, delta0=-0.1, allocation=allocation)
    assert fixed.power_binary(
        n.n_control, 0.8, 0.85, delta0=-0.1, allocation=allocation
    ) == pytest.approx(0.8)
    ev = fixed.events_survival(0.9, hazard_ratio0=1.25, allocation=allocation)
    assert fixed.power_survival(
        ev.events, 0.9, hazard_ratio0=1.25, allocation=allocation
    ) == pytest.approx(0.8)


def test_power_at_margin_is_one_sided_alpha() -> None:
    # The alternative sitting exactly on the margin rejects with probability alpha / 2.
    assert fixed.power_normal(100, -3.0, 10, delta0=-3.0 + 1e-12) == pytest.approx(0.025)
    assert fixed.power_survival(400, 1.3, hazard_ratio0=1.3 + 1e-12) == pytest.approx(0.025)


def test_wider_margin_needs_fewer_patients() -> None:
    n = [fixed.sample_size_binary(0.85, 0.85, delta0=-m).n_control for m in (0.05, 0.10, 0.15)]
    assert n[0] > n[1] > n[2]
    t_n = fixed.sample_size_normal(0.0, 10, delta0=-3, method="t").n_control
    assert t_n > fixed.sample_size_normal(0.0, 10, delta0=-3).n_control


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: fixed.sample_size_normal(0.2, 1, delta0=0.2), "delta must differ from delta0"),
        (
            lambda: fixed.sample_size_binary(0.8, 0.7, delta0=-0.1),
            "p_treatment - p_control must differ from delta0",
        ),
        (
            lambda: fixed.events_survival(1.3, hazard_ratio0=1.3),
            "hazard_ratio must differ from hazard_ratio0",
        ),
        (lambda: fixed.events_survival(1.0, hazard_ratio0=0.0), "must be positive"),
        (lambda: fixed.farrington_manning_rates(0.5, 0.5, 1.0, 1.0), "delta0"),
    ],
)
def test_invalid_input_fails_with_plain_message(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


# --------------------------------------------------------------- report, CLI, config


@pytest.mark.parametrize(
    ("summary", "text"),
    [
        (
            {"endpoint": "binary", "p_control": 0.85, "p_treatment": 0.85, "delta0": -0.1},
            "non-inferiority, H0: p_T - p_C <= -0.1",
        ),
        (
            {"endpoint": "binary", "p_control": 0.2, "p_treatment": 0.2, "delta0": 0.1},
            "non-inferiority, H0: p_T - p_C >= 0.1",
        ),
        (
            {"endpoint": "normal", "delta": 5.0, "sd": 10, "delta0": 2.0},
            "superiority by a margin, H0: mean_T - mean_C <= 2",
        ),
        ({"endpoint": "survival", "hazard_ratio": 1.0, "hazard_ratio0": 1.3}, "HR >= 1.3"),
        ({"endpoint": "survival", "hazard_ratio": 0.7}, "superiority, H0: HR = 1"),
        ({"endpoint": "normal", "delta": 0.5, "sd": 1}, "superiority, H0: mean_T - mean_C = 0"),
    ],
)
def test_hypothesis_text(summary: dict, text: str) -> None:
    assert text in report.hypothesis_text(summary)


def test_superiority_summary_has_no_null_keys() -> None:
    s = report.fixed_summary("binary", pc=0.3, pt=0.45)
    assert "delta0" not in s
    s = report.fixed_summary("survival", hr=0.7)
    assert "hazard_ratio0" not in s


def test_cli_fixed_noninferiority_json() -> None:
    args = ["design", "--endpoint", "binary", "--pc", "0.85", "--pt", "0.85", "--delta0", "-0.1"]
    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["delta0"] == -0.1
    assert out["n_control_ceil"] == 207


def test_cli_group_sequential_survival_noninferiority() -> None:
    args = ["design", "--endpoint", "survival", "--hr", "1", "--hr0", "1.3", "--looks", "3"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "non-inferiority, H0: HR >= 1.3" in result.output
    out = json.loads(runner.invoke(app, [*args, "--json"]).output)
    case = next(c for c in GSD if c["id"] == "ni_gsd_survival_obf3")
    assert out["sample_size"]["max_events"] == pytest.approx(case["max_events"], abs=0.05)
    assert out["hazard_ratio0"] == 1.3


def test_cli_rejects_null_for_the_wrong_endpoint() -> None:
    result = runner.invoke(
        app, ["design", "--endpoint", "survival", "--hr", "0.7", "--delta0", "0"]
    )
    assert result.exit_code == 2
    assert "--delta0 does not apply to the survival endpoint; use --hr0" in result.output


def test_example_config_runs_and_reports(tmp_path: Path) -> None:
    cfg = str(EXAMPLES / "phase3_ni_binary.toml")
    result = runner.invoke(app, ["design", "--config", cfg, "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["delta0"] == -0.1
    assert np.isclose(out["exact"]["type_i_error"], 0.025)
    result = runner.invoke(app, ["report", "--config", cfg, "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "non-inferiority, H0: p_T - p_C <= -0.1" in (tmp_path / "report.md").read_text()
