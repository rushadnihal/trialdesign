import math

import numpy as np
import pytest

from trialdesign import bayes, boundaries, report


def _gsd(endpoint: str = "normal", futility: str | None = "obf", **effect):
    d = boundaries.group_sequential_design(3, 0.025, 0.8, "obf", futility)
    f = report.fixed_summary(endpoint, alpha_two_sided=0.05, **effect)
    return d, f, report.sequential_summary(f, d)


def test_max_sample_size_is_fixed_times_inflation_factor() -> None:
    d, f, s = _gsd(delta=0.5, sd=1.0)
    size = s["sample_size"]
    assert size["max_n_control"] == pytest.approx(f["n_control"] * d.inflation_factor)
    assert size["max_n_control_ceil"] == math.ceil(f["n_control"] * d.inflation_factor)
    assert size["look_total_ceil"][-1] == size["max_n_total_ceil"]
    assert np.all(np.diff(size["look_total_ceil"]) > 0)


def test_survival_max_events_and_patients() -> None:
    d, f, s = _gsd("survival", hr=0.7, prob_event=0.5)
    size = s["sample_size"]
    assert size["unit"] == "events"
    assert size["max_events"] == pytest.approx(f["events"] * d.inflation_factor)
    assert size["max_n_total_ceil"] == math.ceil(size["max_events_ceil"] / 0.5)


def test_exact_operating_characteristics() -> None:
    _, _, s = _gsd(delta=0.5, sd=1.0)
    ex = s["exact"]
    assert ex["type_i_error"] == pytest.approx(0.025, abs=1e-6)  # futility ignored
    assert ex["power"] == pytest.approx(0.8, abs=1e-4)
    assert sum(ex["prob_efficacy_h0"]) < 0.025  # following futility only lowers it
    assert sum(ex["prob_efficacy_h1"]) + sum(ex["prob_futility_h1"]) == pytest.approx(1, 1e-6)
    assert ex["expected_total_h0"] < s["sample_size"]["max_n_total_ceil"]
    assert s["alpha_spent_cumulative"][-1] == pytest.approx(0.025)


def test_sequential_summary_rejects_mismatched_alpha() -> None:
    d = boundaries.group_sequential_design(3)
    f = report.fixed_summary("survival", hr=0.7, alpha_two_sided=0.025)
    with pytest.raises(ValueError, match="2 \\* alpha_one_sided"):
        report.sequential_summary(f, d)


def test_format_table_plain_and_markdown() -> None:
    plain = report.format_table(["a", "b"], [["x", 0.5], ["yy", None]])
    assert plain.splitlines() == ["a        b", "x   0.5000", "yy       -"]
    md = report.format_table(["a"], [["P(x | y)"], [1.5e-5]], markdown=True)
    assert md.splitlines() == ["| a |", "| --- |", "| P(x \\| y) |", "| 1.50e-05 |"]


def test_bayes_summary_matches_module() -> None:
    b = bayes.bayesian_interim_design(30, 60)
    s = report.bayes_summary(b, 0.3, 0.45, design_prior_t=(4.5, 5.5), xc=6, xt=12)
    assert s["h0"]["prob_success"] == pytest.approx(
        b.operating_characteristics(0.3, 0.3).prob_success
    )
    assert s["assurance"]["prob_success"] == pytest.approx(
        b.assurance(0.3, (4.5, 5.5)).prob_success
    )
    assert s["max_type_i_error"] >= s["h0"]["prob_success"]
    assert s["interim"]["posterior_prob_superiority"] == pytest.approx(0.951, abs=0.002)
    assert s["interim"]["decision"] == b.decide(6, 12)
    # The rule table reproduces the decision table.
    for row in s["decision_rule"]:
        d = b.decision_table[row["xc"]]
        if row["go_if_xt_at_least"] is not None:
            assert np.all(d[row["go_if_xt_at_least"] :] == 1)
        if row["nogo_if_xt_at_most"] is not None:
            assert np.all(d[: row["nogo_if_xt_at_most"] + 1] == -1)


def test_bayes_summary_needs_both_interim_counts() -> None:
    b = bayes.bayesian_interim_design(10, 20)
    with pytest.raises(ValueError, match="give both xc and xt"):
        report.bayes_summary(b, 0.3, 0.45, xc=3)


def test_write_report_needs_a_design(tmp_path) -> None:
    with pytest.raises(ValueError, match="nothing to report"):
        report.write_report(tmp_path, "0")
