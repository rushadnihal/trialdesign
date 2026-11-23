r"""Design summaries, plain and Markdown tables, PNG plots and the Markdown design report.

The ``*_summary`` functions turn the objects of :mod:`trialdesign.fixed`,
:mod:`trialdesign.boundaries`, :mod:`trialdesign.simulate` and :mod:`trialdesign.bayes` into
plain dictionaries (JSON-ready, no numpy types) that the command line prints and the report
renders. Only :func:`write_report` and the ``plot_*`` functions touch the disk.

Group-sequential designs use a one-sided alpha throughout. The fixed design they are
inflated from uses the two-sided alpha ``2 * alpha_one_sided``, so the two rest on the same
critical value. Operating characteristics are exact (recursive numerical integration,
Jennison & Turnbull 2000, chapter 19) and, when a simulation is run, are shown next to the
Monte Carlo estimates and their standard errors :math:`\sqrt{p(1-p)/n_{sim}}`.

References
----------
Jennison, C. and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
Clinical Trials*. Chapman & Hall/CRC.
Lan, K. K. G. and DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials.
*Biometrika* 70, 659-663.
O'Hagan, A., Stevens, J. W. and Campbell, M. J. (2005). Assurance in clinical trial design.
*Pharmaceutical Statistics* 4, 187-201.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from trialdesign import accrual, bayes, fixed, ssr
from trialdesign.boundaries import (
    GroupSequentialDesign,
    crossing_probabilities,
    expected_info_fraction,
)
from trialdesign.simulate import DesignSimulation, OperatingCharacteristics
from trialdesign.spending import cumulative_spending

__all__ = [
    "CONVENTIONAL_ALPHA_ONE_SIDED",
    "SPENDING_NAMES",
    "accrual_summary",
    "accrual_text",
    "alpha_note",
    "bayes_markdown",
    "bayes_simulation_summary",
    "bayes_summary",
    "bayes_tables",
    "effect_text",
    "fixed_summary",
    "format_table",
    "futility_text",
    "gsd_markdown",
    "gsd_tables",
    "hypothesis_text",
    "needs_alpha_note",
    "plot_bayes_oc",
    "plot_boundaries",
    "plot_decision_map",
    "plot_gsd_oc",
    "plot_ssr",
    "sequential_summary",
    "simulation_summary",
    "ssr_markdown",
    "ssr_summary",
    "ssr_tables",
    "success_text",
    "write_report",
]

# Reference palette (validated categorical slots 1-3, diverging poles and neutral midpoint).
_BLUE, _ORANGE, _AQUA = "#2a78d6", "#eb6834", "#1baf7a"
_RED, _NEUTRAL = "#e34948", "#e1e0d9"
_INK, _INK_2, _MUTED, _GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

SPENDING_NAMES = {"obf": "O'Brien-Fleming-type", "pocock": "Pocock-type"}
_ENDPOINT_NAMES = {
    "normal": "Continuous (normal)",
    "binary": "Binary",
    "survival": "Time to event (log-rank)",
}

Summary = dict[str, Any]


def _floats(x: NDArray | Sequence[float]) -> list[float]:
    return [float(v) for v in x]


def _ceil(x: float) -> int:
    return math.ceil(x - 1e-9)


# ------------------------------------------------------------------------ summaries


def fixed_summary(
    endpoint: str,
    *,
    delta: float | None = None,
    sd: float | None = None,
    pc: float | None = None,
    pt: float | None = None,
    hr: float | None = None,
    prob_event: float | None = None,
    alpha_two_sided: float = 0.05,
    power: float = 0.80,
    allocation: float = 1.0,
    method: fixed.NormalMethod = "z",
    delta0: float | None = None,
    hr0: float | None = None,
    accrual_inputs: Summary | None = None,
) -> Summary:
    """Fixed-design sample size (or events) as a flat dictionary.

    The endpoint's inputs must already be present; see ``trialdesign design --help``.
    ``delta0`` (normal, binary) or ``hr0`` (survival) set the effect under H0 for a
    non-inferiority design; leave them out for superiority. They appear in the summary only
    when given, so superiority output is unchanged.

    For survival, ``accrual_inputs`` holds the keyword arguments of
    :func:`trialdesign.accrual.survival_study` other than the events and hazard ratio
    (``median_control`` plus two of ``accrual_time``, ``accrual_rate`` and ``followup``,
    and optionally ``dropout`` and ``dropout_time``). It turns events into patients and
    calendar time, in place of a single ``prob_event``.
    """
    common = {"alpha_two_sided": alpha_two_sided, "power": power, "allocation": allocation}
    if endpoint == "survival":
        assert hr is not None
        if delta0 is not None:
            raise ValueError("delta0 is for normal and binary endpoints; use hr0 for survival")
        if accrual_inputs is not None and prob_event is not None:
            raise ValueError(
                "give either prob_event or an accrual model (median_control, accrual and "
                "follow-up), not both"
            )
        ev = fixed.events_survival(
            hr, prob_event=prob_event, hazard_ratio0=1.0 if hr0 is None else hr0, **common
        )
        out: Summary = {
            "endpoint": endpoint,
            "design": "fixed",
            "hazard_ratio": hr,
            **({} if hr0 is None else {"hazard_ratio0": hr0}),
            "prob_event": prob_event,
            **common,
            "events": ev.events,
            "events_ceil": ev.events_ceil,
        }
        if ev.n_total is not None:
            out["n_total"] = ev.n_total
            out["n_total_ceil"] = ev.n_total_ceil
        if accrual_inputs is not None:
            study = _study(ev.events, hr, allocation, accrual_inputs)
            out["accrual_inputs"] = dict(accrual_inputs)
            out["accrual"] = accrual_summary(study)
            out["n_total"] = study.n_total
            out["n_total_ceil"] = _ceil(study.n_total)
        return out
    if hr0 is not None:
        raise ValueError("hr0 is for the survival endpoint; use delta0 for normal and binary")
    d0 = 0.0 if delta0 is None else delta0
    if endpoint == "normal":
        assert delta is not None and sd is not None
        res = fixed.sample_size_normal(delta, sd, method=method, delta0=d0, **common)
        inputs: Summary = {"delta": delta, "sd": sd, "method": method}
    elif endpoint == "binary":
        assert pc is not None and pt is not None
        res = fixed.sample_size_binary(pc, pt, delta0=d0, **common)
        inputs = {"p_control": pc, "p_treatment": pt}
    else:
        raise ValueError(f"unknown endpoint {endpoint!r}")
    if delta0 is not None:
        inputs["delta0"] = delta0
    return {
        "endpoint": endpoint,
        "design": "fixed",
        **inputs,
        **common,
        "n_control": res.n_control,
        "n_treatment": res.n_treatment,
        "n_control_ceil": res.n_control_ceil,
        "n_treatment_ceil": res.n_treatment_ceil,
        "n_total_ceil": res.n_total_ceil,
    }


_INPUT_KEYS = (
    "delta",
    "sd",
    "method",
    "p_control",
    "p_treatment",
    "delta0",
    "hazard_ratio",
    "hazard_ratio0",
    "prob_event",
    "accrual_inputs",
)


def _study(events: float, hr: float, allocation: float, inputs: Summary) -> accrual.SurvivalStudy:
    return accrual.survival_study(events, hr, allocation=allocation, **inputs)


def accrual_summary(
    study: accrual.SurvivalStudy, info_rates: Sequence[float] | None = None
) -> Summary:
    """Accrual, follow-up and calendar times of a :class:`~trialdesign.accrual.SurvivalStudy`.

    With ``info_rates``, also the expected calendar time and patients randomised at each look.
    Times are in the unit of the inputs (months by convention).
    """
    out: Summary = {
        "n_total": study.n_total,
        "n_total_ceil": _ceil(study.n_total),
        "accrual_times": _floats(study.accrual_times),
        "accrual_rates": _floats(study.accrual_rates),
        "accrual_duration": study.accrual_duration,
        "followup": study.followup,
        "study_duration": study.study_duration,
        "prob_event": study.prob_event,
        "hazard_control": study.hazard_control,
        "dropout_hazard": study.dropout,
    }
    if info_rates is not None:
        times = study.look_times(info_rates)
        out["look_time"] = _floats(times)
        out["look_patients"] = [study.patients_at(t) for t in times]
    return out


def accrual_text(s: Summary) -> str:
    """One sentence on accrual and timing, e.g. for the report and the CLI."""
    a, inputs = s["accrual"], s["accrual_inputs"]
    rates = a["accrual_rates"]
    if len(rates) == 1:
        pace = f"at {rates[0]:.1f} patients per month for {a['accrual_duration']:.1f} months"
    else:
        segs = ", ".join(
            f"{r:.1f}/month to month {t:.1f}"
            for r, t in zip(rates, a["accrual_times"], strict=True)
        )
        pace = f"with piecewise accrual ({segs})"
    drop = inputs.get("dropout", 0.0) or 0.0
    dropout = (
        f" ({drop:.0%} dropout per {inputs.get('dropout_time', 12.0):g} months)" if drop else ""
    )
    return (
        f"Recruiting {a['n_total_ceil']} patients {pace} and following each for at least "
        f"{a['followup']:.1f} months{dropout}, with a control median of "
        f"{inputs['median_control']:g} months, puts the final analysis at month "
        f"{a['study_duration']:.1f}."
    )


def sequential_summary(fixed_design: Summary, design: GroupSequentialDesign) -> Summary:
    """Group-sequential design built on a fixed design with the same one-sided alpha and power.

    Parameters
    ----------
    fixed_design : dict
        From :func:`fixed_summary` with ``alpha_two_sided = 2 * design.alpha_one_sided``.
    design : GroupSequentialDesign
        From :func:`trialdesign.boundaries.group_sequential_design`.

    Returns
    -------
    dict
        Inputs, boundaries, maximum and per-look sample size (or events), and exact operating
        characteristics. ``type_i_error`` ignores futility (non-binding); everything under H1,
        and the futility and expected-size entries under H0, follow futility.
    """
    if not math.isclose(fixed_design["alpha_two_sided"], 2 * design.alpha_one_sided):
        raise ValueError("the fixed design must use alpha_two_sided = 2 * alpha_one_sided")
    t, c, b = design.info_rates, design.efficacy, design.futility
    infl = design.inflation_factor
    survival = fixed_design["endpoint"] == "survival"
    size: Summary
    if survival:
        unit = "events"
        max_unrounded = fixed_design["events"] * infl
        max_total = _ceil(max_unrounded)
        size = {"unit": unit, "max_events": max_unrounded, "max_events_ceil": max_total}
        if fixed_design.get("accrual_inputs") is not None:
            study = _study(
                max_unrounded,
                fixed_design["hazard_ratio"],
                fixed_design["allocation"],
                fixed_design["accrual_inputs"],
            )
            acc = accrual_summary(study, design.info_rates)
            size["accrual"] = acc
            size["max_n_total"] = acc["n_total"]
            size["max_n_total_ceil"] = acc["n_total_ceil"]
        elif fixed_design.get("prob_event") is not None:
            n = max_total / fixed_design["prob_event"]
            size["max_n_total"] = n
            size["max_n_total_ceil"] = _ceil(n)
    else:
        unit = "patients"
        mc = fixed_design["n_control"] * infl
        mt = fixed_design["n_treatment"] * infl
        max_total = _ceil(mc) + _ceil(mt)
        size = {
            "unit": unit,
            "max_n_control": mc,
            "max_n_treatment": mt,
            "max_n_control_ceil": _ceil(mc),
            "max_n_treatment_ceil": _ceil(mt),
            "max_n_total_ceil": max_total,
        }
    looks = [_ceil(v) for v in t * max_total]
    looks[-1] = max_total
    size["look_total_ceil"] = looks
    size["fixed_total"] = fixed_design["events" if survival else "n_total_ceil"]

    eff0, _ = crossing_probabilities(t, c, None, 0.0)
    eff0_f, fut0_f = crossing_probabilities(t, c, b, 0.0)
    eff1, fut1 = crossing_probabilities(t, c, b, design.drift)
    e0 = expected_info_fraction(t, c, b, 0.0)
    e1 = expected_info_fraction(t, c, b, design.drift)
    return {
        "endpoint": fixed_design["endpoint"],
        "design": "group-sequential",
        **{k: fixed_design[k] for k in _INPUT_KEYS if k in fixed_design},
        "allocation": fixed_design["allocation"],
        "alpha_one_sided": design.alpha_one_sided,
        "power": design.power,
        "looks": _floats(t),
        "efficacy_spending": design.efficacy_spending,
        "futility_spending": design.futility_spending,
        "binding_futility": False,
        "efficacy_z": _floats(c),
        "futility_z": None if b is None else _floats(b),
        "nominal_p_one_sided": _floats(stats.norm.sf(c)),
        "alpha_spent_cumulative": _floats(
            cumulative_spending(design.efficacy_spending, t, design.alpha_one_sided)
        ),
        "drift": design.drift,
        "inflation_factor": infl,
        "fixed": fixed_design,
        "sample_size": size,
        "exact": {
            "type_i_error": float(eff0.sum()),
            "power": float(eff1.sum()),
            "prob_efficacy_h0": _floats(eff0_f),
            "prob_futility_h0": _floats(fut0_f),
            "prob_efficacy_h1": _floats(eff1),
            "prob_futility_h1": _floats(fut1),
            "expected_total_h0": e0 * max_total,
            "expected_total_h1": e1 * max_total,
        },
    }


def _oc_dict(oc: OperatingCharacteristics, n_max: float) -> Summary:
    en, en_se = oc.expected_sample_size(n_max)
    return {
        "reject": oc.reject_prob,
        "reject_se": oc.reject_prob_se,
        "futility": oc.futility_prob,
        "futility_se": oc.futility_prob_se,
        "prob_efficacy": _floats(oc.prob_efficacy),
        "prob_efficacy_se": _floats(oc.prob_efficacy_se),
        "prob_futility": _floats(oc.prob_futility),
        "prob_futility_se": _floats(oc.prob_futility_se),
        "expected_total": en,
        "expected_total_se": en_se,
    }


def simulation_summary(sim: DesignSimulation, n_max: float) -> Summary:
    """Simulated operating characteristics, each with its Monte Carlo SE.

    ``h0_ignore_futility`` gives the type I error of the non-binding design; ``h0`` and
    ``h1`` follow futility. ``n_max`` is the maximum total sample size (or events).
    """
    return {
        "nsim": sim.h1.n_sim,
        "seed": sim.seed,
        "type_i_error": sim.type_i_error,
        "type_i_error_se": sim.h0_ignore_futility.reject_prob_se,
        "power": sim.power,
        "power_se": sim.h1.reject_prob_se,
        "h0_ignore_futility": _oc_dict(sim.h0_ignore_futility, n_max),
        "h0": _oc_dict(sim.h0, n_max),
        "h1": _oc_dict(sim.h1, n_max),
    }


SSR_EFFECTS = (1.0, 0.85, 0.7)  # true effect / design effect for the SSR OC table


def _ssr_oc_dict(oc: ssr.SSROperatingCharacteristics, n_max: float) -> Summary:
    return {
        "reject_prob": oc.reject_prob,
        "prob_efficacy_interim": oc.prob_efficacy_interim,
        "prob_futility_interim": oc.prob_futility_interim,
        "prob_zone": dict(oc.prob_zone),
        "expected_n_total": oc.expected_n_fraction * n_max,
    }


def ssr_summary(
    planned: Summary,
    design: ssr.SSRDesign,
    z1: float | None = None,
    sim: Sequence[ssr.SSRSimulation] | None = None,
) -> Summary:
    """Promising-zone sample size re-estimation as a dictionary.

    Parameters
    ----------
    planned : dict
        :func:`sequential_summary` of the planned two-look design (``design.gsd``).
    design : SSRDesign
        The adaptive design.
    z1 : float, optional
        Observed interim z-statistic: adds the conditional power, the zone and the
        re-estimated sample size.
    sim : sequence of SSRSimulation, optional
        Simulations under H0 (futility ignored) and the design alternative, as checks.

    Returns
    -------
    dict
        Sample sizes are in patients: the planned maximum ``n_planned``, the cap
        ``n_cap``, and expected totals. ``exact`` holds the type I error (CHW and naive
        tests, futility ignored) and, for each true effect in :data:`SSR_EFFECTS` as a
        fraction of the design effect, the adaptive and the planned design's power and
        expected sample size (futility followed).
    """
    size = planned["sample_size"]
    if size["unit"] != "patients":
        raise ValueError("sample size re-estimation is for normal and binary endpoints")
    n_max = size["max_n_total_ceil"]
    n1 = size["look_total_ceil"][0]
    n_cap = _ceil(n_max * design.max_factor)
    drift = design.gsd.drift
    zones = {
        "cp_min": design._z_at_cp(design.cp_min),
        "cp_target": design._z_at_cp(design.cp_target),
    }
    h0 = design.operating_characteristics(0.0, use_futility=False)
    h0_naive = design.operating_characteristics(0.0, use_futility=False, test="naive")
    effects = []
    for theta in SSR_EFFECTS:
        oc = design.operating_characteristics(theta * drift)
        eff, _ = design.gsd.crossing_probabilities(theta * drift)
        effects.append(
            {
                "effect_ratio": theta,
                **_ssr_oc_dict(oc, n_max),
                "planned_reject_prob": float(eff.sum()),
                "planned_expected_n_total": design.gsd.expected_info_fraction(theta * drift)
                * n_max,
            }
        )
    out: Summary = {
        "planned": planned,
        "interim_fraction": design.t1,
        "cp_min": design.cp_min,
        "cp_target": design.cp_target,
        "max_factor": design.max_factor,
        "n_planned": n_max,
        "n_interim": n1,
        "n_cap": n_cap,
        "c1": design.c1,
        "c2": design.c2,
        "b1": design.b1,
        "zone_z1": zones,
        "zone_effect_ratio": {
            k: None if z is None else z / math.sqrt(design.t1) / drift for k, z in zones.items()
        },
        "exact": {
            "type_i_error": h0.reject_prob,
            "type_i_error_naive": h0_naive.reject_prob,
            "h0": _ssr_oc_dict(h0, n_max),
            "effects": effects,
        },
    }
    if z1 is not None:
        m = float(design.stage2_multiplier(z1))
        zone = str(design.zone(z1))
        n2_planned = n_max - n1
        n_total = n1 if zone in ("efficacy", "futility") else n1 + _ceil(m * n2_planned)
        out["interim"] = {
            "z1": z1,
            "zone": zone,
            "observed_effect_ratio": z1 / math.sqrt(design.t1) / drift,
            "cp_observed": float(design.conditional_power(z1)),
            "cp_design": float(design.conditional_power(z1, drift=drift)),
            "stage2_multiplier": m,
            "n_total": n_total,
            "cp_observed_after": float(design.conditional_power(z1, m=m)),
        }
    if sim:
        out["simulation"] = [
            {
                "effect_ratio": x.drift / drift,
                "nsim": x.n_sim,
                "seed": x.seed,
                "reject_prob": x.reject_prob,
                "reject_prob_se": x.reject_prob_se,
                "reject_prob_naive": x.reject_prob_naive,
                "reject_prob_naive_se": x.reject_prob_naive_se,
                "expected_n_total": x.expected_n_fraction * n_max,
                "expected_n_total_se": x.expected_n_fraction_se * n_max,
                "prob_zone": dict(x.prob_zone),
            }
            for x in sim
        ]
    return out


def ssr_tables(s: Summary, markdown: bool = False) -> list[str]:
    """Tables for an SSR summary: the zones, then operating characteristics."""

    def z_or_dash(z: float | None) -> str:
        return "-" if z is None else f"{z:.3f}"

    zl, zh = s["zone_z1"]["cp_min"], s["zone_z1"]["cp_target"]
    el, eh = s["zone_effect_ratio"]["cp_min"], s["zone_effect_ratio"]["cp_target"]
    lo_txt = "-inf" if s["b1"] is None else f"{s['b1']:.3f}"
    zone_rows: list[list[Any]] = []
    if s["b1"] is not None:
        zone_rows.append(["Futility (stop)", f"< {lo_txt}", "", s["n_interim"]])
    zone_rows += [
        ["Unfavourable", f"< {z_or_dash(zl)}", f"< {z_or_dash(el)}", s["n_planned"]],
        [
            "Promising",
            f"{z_or_dash(zl)} to {z_or_dash(zh)}",
            f"{z_or_dash(el)} to {z_or_dash(eh)}",
            f"{s['n_planned']} to {s['n_cap']}",
        ],
        ["Favourable", f"{z_or_dash(zh)} to {s['c1']:.3f}", f">= {z_or_dash(eh)}", s["n_planned"]],
        ["Efficacy (stop)", f">= {s['c1']:.3f}", "", s["n_interim"]],
    ]
    zones = format_table(
        ["Zone", "Interim z1", "Observed / design effect", "Total patients"], zone_rows, markdown
    )
    ex = s["exact"]
    rows: list[list[Any]] = [
        [
            "H0 (futility ignored), CHW test",
            ex["type_i_error"],
            "",
            ex["h0"]["expected_n_total"],
            "",
            ex["h0"]["prob_zone"]["promising"],
        ],
        ["H0, naive pooled test", ex["type_i_error_naive"], "", "", "", ""],
    ]
    for e in ex["effects"]:
        rows.append(
            [
                f"Effect {e['effect_ratio']:g} x design",
                e["reject_prob"],
                e["planned_reject_prob"],
                e["expected_n_total"],
                e["planned_expected_n_total"],
                e["prob_zone"]["promising"],
            ]
        )
    oc = format_table(
        [
            "Scenario",
            "P(reject) adaptive",
            "P(reject) planned",
            "E[patients] adaptive",
            "E[patients] planned",
            "P(promising)",
        ],
        rows,
        markdown,
    )
    out = [zones, oc]
    if "simulation" in s:
        sim_rows = []
        exact_by_ratio = {e["effect_ratio"]: e for e in ex["effects"]}
        for x in s["simulation"]:
            r = round(x["effect_ratio"], 12)
            exact = ex["type_i_error"] if r == 0 else exact_by_ratio.get(r, {}).get("reject_prob")
            sim_rows.append(
                [
                    "H0 (futility ignored)" if r == 0 else f"Effect {r:g} x design",
                    exact,
                    x["reject_prob"],
                    x["reject_prob_se"],
                    _z(x["reject_prob"], exact, x["reject_prob_se"]) if exact is not None else "",
                    x["reject_prob_naive"],
                ]
            )
        out.append(
            format_table(
                ["Scenario", "Exact", "Simulated", "MC SE", "z vs exact", "Simulated, naive test"],
                sim_rows,
                markdown,
            )
        )
    return out


def _bayes_oc_dict(oc: bayes.BayesianOC) -> Summary:
    return {
        "prob_success": oc.prob_success,
        "prob_go": oc.prob_go,
        "prob_nogo": oc.prob_nogo,
        "prob_continue": oc.prob_continue,
        "prob_final_success": oc.prob_final_success,
        "expected_n_total": oc.expected_n_control + oc.expected_n_treatment,
    }


def _decision_rule(design: bayes.BayesianInterimDesign) -> list[Summary]:
    """For each interim control count: no-go up to, and go from, these treatment counts."""
    rows = []
    for xc, row in enumerate(design.decision_table):
        nogo = np.flatnonzero(row == -1)
        go = np.flatnonzero(row == 1)
        rows.append(
            {
                "xc": xc,
                "nogo_if_xt_at_most": int(nogo.max()) if nogo.size else None,
                "go_if_xt_at_least": int(go.min()) if go.size else None,
            }
        )
    return rows


def _max_type_i_error(design: bayes.BayesianInterimDesign, pc: float) -> tuple[float, float]:
    return bayes.max_type_i_error(design, np.unique(np.append(bayes.NULL_RATES, pc)))


def bayes_summary(
    design: bayes.BayesianInterimDesign,
    pc: float,
    pt: float,
    *,
    design_prior_c: bayes.DesignPrior | None = None,
    design_prior_t: bayes.DesignPrior | None = None,
    xc: int | None = None,
    xt: int | None = None,
) -> Summary:
    """Exact operating characteristics of a Bayesian interim design, as a dictionary.

    ``h0`` is at :math:`p_T = p_C = ` ``pc`` (its ``prob_success`` is the type I error) and
    ``h1`` at ``(pc, pt)``. ``max_type_i_error`` is the largest type I error over a grid of
    common response rates from 0.05 to 0.95. Assurance is given when either design prior is
    set; an arm without one is fixed at ``pc`` or ``pt``. Interim results are given when
    ``xc`` and ``xt`` are set.
    """
    h0 = design.operating_characteristics(pc, pc)
    h1 = design.operating_characteristics(pc, pt)
    max_err, at_p = _max_type_i_error(design, pc)
    out: Summary = {
        "endpoint": "binary",
        "design": "bayesian-interim",
        "p_control": pc,
        "p_treatment": pt,
        "n_interim_control": design.n_interim_control,
        "n_interim_treatment": design.n_interim_treatment,
        "n_final_control": design.n_final_control,
        "n_final_treatment": design.n_final_treatment,
        "prior_control": list(design.prior_control),
        "prior_treatment": list(design.prior_treatment),
        "success_threshold": design.success_threshold,
        "go_threshold": design.go_threshold,
        "nogo_threshold": design.nogo_threshold,
        "h0": _bayes_oc_dict(h0),
        "h1": _bayes_oc_dict(h1),
        "max_type_i_error": max_err,
        "max_type_i_error_at": at_p,
        "decision_rule": _decision_rule(design),
        "assurance": None,
        "interim": None,
    }
    if design_prior_c is not None or design_prior_t is not None:
        law_c = pc if design_prior_c is None else design_prior_c
        law_t = pt if design_prior_t is None else design_prior_t
        a = design.assurance(law_c, law_t)
        out["assurance"] = {
            "design_prior_control": list(law_c) if isinstance(law_c, tuple) else law_c,
            "design_prior_treatment": list(law_t) if isinstance(law_t, tuple) else law_t,
            **_bayes_oc_dict(a),
        }
    if (xc is None) != (xt is None):
        raise ValueError("give both xc and xt (interim responders per arm), or neither")
    if xc is not None and xt is not None:
        out["interim"] = {
            "xc": xc,
            "xt": xt,
            "posterior_prob_superiority": bayes.prob_superiority(
                xc,
                design.n_interim_control,
                xt,
                design.n_interim_treatment,
                design.prior_control,
                design.prior_treatment,
            ),
            "ppos": design.ppos(xc, xt),
            "decision": design.decide(xc, xt),
        }
    return out


def _bayes_sim_dict(oc: OperatingCharacteristics, n_total: int) -> Summary:
    return {
        "prob_success": oc.reject_prob,
        "prob_success_se": oc.reject_prob_se,
        "prob_go": float(oc.prob_efficacy[0]),
        "prob_go_se": float(oc.prob_efficacy_se[0]),
        "prob_nogo": float(oc.prob_futility[0]),
        "prob_nogo_se": float(oc.prob_futility_se[0]),
        "expected_n_total": oc.expected_info_fraction * n_total,
        "expected_n_total_se": oc.expected_info_fraction_se * n_total,
    }


def bayes_simulation_summary(
    design: bayes.BayesianInterimDesign,
    pc: float,
    pt: float,
    nsim: int,
    seed: int,
    *,
    design_prior_c: bayes.DesignPrior | None = None,
    design_prior_t: bayes.DesignPrior | None = None,
) -> Summary:
    """Simulated check of :func:`bayes_summary` through the M3 engine, with MC SEs.

    H0, H1 and assurance reuse the same seed (common random numbers), as
    :func:`trialdesign.simulate.simulate_design` does.
    """
    n_total = design.n_final_control + design.n_final_treatment
    out: Summary = {
        "nsim": nsim,
        "seed": seed,
        "h0": _bayes_sim_dict(bayes.simulate_bayesian(design, pc, pc, nsim, seed), n_total),
        "h1": _bayes_sim_dict(bayes.simulate_bayesian(design, pc, pt, nsim, seed), n_total),
        "assurance": None,
    }
    if design_prior_c is not None or design_prior_t is not None:
        law_c = pc if design_prior_c is None else design_prior_c
        law_t = pt if design_prior_t is None else design_prior_t
        oc = bayes.simulate_assurance(design, law_c, law_t, nsim, seed)
        out["assurance"] = _bayes_sim_dict(oc, n_total)
    return out


# --------------------------------------------------------------------------- tables


def _z(estimate: float, exact: float, se: float) -> str:
    """(simulated - exact) / MC SE, as text; |z| > 2 is flagged with an asterisk."""
    if se == 0:
        return "-"
    z = (estimate - exact) / se
    return f"{z:+.2f}" + ("*" if abs(z) > 2 else "")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if 0 < abs(value) < 10**-digits:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def format_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], markdown: bool = False
) -> str:
    """A plain-text (aligned) or Markdown table; floats are shown to 4 decimals."""
    cells = [[_fmt(v) for v in row] for row in rows]
    if markdown:
        cells = [[c.replace("|", "\\|") for c in row] for row in cells]
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in cells]
        return "\n".join(lines)
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
    return "\n".join(fmt.format(*r).rstrip() for r in [list(headers), *cells])


def effect_text(s: Summary) -> str:
    if s["endpoint"] == "normal":
        return f"difference {s['delta']:g}, SD {s['sd']:g} ({s['method']} method)"
    if s["endpoint"] == "binary":
        return f"p_C {s['p_control']:g} vs p_T {s['p_treatment']:g}"
    text = f"hazard ratio {s['hazard_ratio']:g}"
    if s.get("prob_event") is not None:
        text += f", P(event) {s['prob_event']:g}"
    if s.get("accrual_inputs") is not None:
        text += f", control median {s['accrual_inputs']['median_control']:g} months"
    return text


def hypothesis_text(s: Summary) -> str:
    """The null hypothesis in words, e.g. ``non-inferiority, H0: p_T - p_C <= -0.1``.

    The test is one-sided in the direction of the design alternative. A null effect on the
    harmful side of no difference is non-inferiority; on the beneficial side it is
    superiority by a margin.
    """
    if s["endpoint"] == "survival":
        h0 = s.get("hazard_ratio0")
        if h0 is None or h0 == 1:
            return "superiority, H0: HR = 1"
        gap = math.log(s["hazard_ratio"]) - math.log(h0)
        kind = "non-inferiority" if gap * math.log(h0) < 0 else "superiority by a margin"
        return f"{kind}, H0: HR {'>=' if gap < 0 else '<='} {h0:g}"
    d0 = s.get("delta0")
    name = "mean_T - mean_C" if s["endpoint"] == "normal" else "p_T - p_C"
    if d0 is None or d0 == 0:
        return f"superiority, H0: {name} = 0"
    effect = s["delta"] if s["endpoint"] == "normal" else s["p_treatment"] - s["p_control"]
    gap = effect - d0
    kind = "non-inferiority" if gap * d0 < 0 else "superiority by a margin"
    return f"{kind}, H0: {name} {'<=' if gap > 0 else '>='} {d0:g}"


def futility_text(s: Summary) -> str:
    if s["futility_spending"] is None:
        return "none"
    return f"Lan-DeMets {SPENDING_NAMES[s['futility_spending']]} beta spending, non-binding"


def gsd_tables(s: Summary, sim: Summary | None = None, markdown: bool = False) -> list[str]:
    """Tables for a group-sequential summary: boundaries, then operating characteristics."""
    unit = s["sample_size"]["unit"]
    fut = s["futility_z"] or [None] * len(s["looks"])
    acc = s["sample_size"].get("accrual")
    headers = [
        "Look",
        "Info fraction",
        f"Cumulative {unit}",
        "Efficacy z",
        "Nominal p (1-sided)",
        "Cumulative alpha",
        "Futility z",
    ]
    rows_b = [
        [k + 1, *row]
        for k, row in enumerate(
            zip(
                s["looks"],
                s["sample_size"]["look_total_ceil"],
                s["efficacy_z"],
                s["nominal_p_one_sided"],
                s["alpha_spent_cumulative"],
                fut,
                strict=True,
            )
        )
    ]
    if acc is not None:
        headers[3:3] = ["Month", "Patients"]
        for row, t, n in zip(rows_b, acc["look_time"], acc["look_patients"], strict=True):
            row[3:3] = [f"{t:.1f}", _ceil(n)]
    boundaries = format_table(headers, rows_b, markdown)
    ex = s["exact"]
    per_look_headers = ["Look", "P(eff) H0", "P(fut) H0", "P(eff) H1", "P(fut) H1"]
    per_look = format_table(
        per_look_headers,
        [
            [k + 1, *row]
            for k, row in enumerate(
                zip(
                    ex["prob_efficacy_h0"],
                    ex["prob_futility_h0"],
                    ex["prob_efficacy_h1"],
                    ex["prob_futility_h1"],
                    strict=True,
                )
            )
        ],
        markdown,
    )
    rows: list[list[Any]] = [
        ["Type I error (futility ignored)", ex["type_i_error"]],
        ["Power (futility followed)", ex["power"]],
        [f"Expected {unit} under H0", ex["expected_total_h0"]],
        [f"Expected {unit} under H1", ex["expected_total_h1"]],
    ]
    headers = ["Quantity", "Exact"]
    if sim is not None:
        headers += ["Simulated", "MC SE", "z vs exact"]
        estimates = [
            (sim["type_i_error"], sim["type_i_error_se"]),
            (sim["power"], sim["power_se"]),
            (sim["h0"]["expected_total"], sim["h0"]["expected_total_se"]),
            (sim["h1"]["expected_total"], sim["h1"]["expected_total_se"]),
        ]
        exact = [ex["type_i_error"], ex["power"], ex["expected_total_h0"], ex["expected_total_h1"]]
        for row, (est, se), ref in zip(rows, estimates, exact, strict=True):
            row += [est, se, _z(est, ref, se)]
    return [boundaries, format_table(headers, rows, markdown), per_look]


def bayes_tables(s: Summary, sim: Summary | None = None, markdown: bool = False) -> list[str]:
    """Tables for a Bayesian interim summary: OCs (exact and simulated) and the decision rule."""
    h0, h1 = s["h0"], s["h1"]
    cols = ["Quantity", f"H0 (p_T = p_C = {s['p_control']:g})", f"H1 (p_T = {s['p_treatment']:g})"]
    keys = [
        ("P(success)", "prob_success"),
        ("P(go at interim)", "prob_go"),
        ("P(no-go at interim)", "prob_nogo"),
        ("P(continue)", "prob_continue"),
        ("Expected N (total)", "expected_n_total"),
    ]
    if s["assurance"] is not None:
        cols.append("Design prior (assurance)")
    rows = []
    for label, key in keys:
        row: list[Any] = [label, h0[key], h1[key]]
        if s["assurance"] is not None:
            row.append(s["assurance"][key])
        rows.append(row)
    tables = [format_table(cols, rows, markdown)]
    if sim is not None:
        sim_rows: list[list[Any]] = []
        scen = [("H0", "h0"), ("H1", "h1")] + (
            [("Assurance", "assurance")] if sim["assurance"] else []
        )
        for name, key in scen:
            for label, k in [("P(success)", "prob_success"), ("P(go)", "prob_go")]:
                est, se = sim[key][k], sim[key][f"{k}_se"]
                exact = s[key][k]
                sim_rows.append([f"{name} {label}", exact, est, se, _z(est, exact, se)])
        tables.append(
            format_table(
                ["Quantity", "Exact", "Simulated", "MC SE", "z vs exact"], sim_rows, markdown
            )
        )
    rule = format_table(
        ["Interim control responders", "No-go if treatment responders <=", "Go if >="],
        [[r["xc"], r["nogo_if_xt_at_most"], r["go_if_xt_at_least"]] for r in s["decision_rule"]],
        markdown,
    )
    tables.append(rule)
    return tables


# ---------------------------------------------------------------------------- plots


def _figure(width: float, height: float):
    from matplotlib.figure import Figure  # imported here so the CLI starts fast

    return Figure(figsize=(width, height), dpi=150)


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_INK_2, labelsize=9)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(_INK_2)
    ax.yaxis.label.set_color(_INK_2)


def plot_boundaries(design: GroupSequentialDesign, path: str | Path) -> Path:
    """Efficacy and futility z-boundaries against information fraction, saved as PNG."""
    t, c, b = design.info_rates, design.efficacy, design.futility
    fig = _figure(7, 4.2)
    ax = fig.add_subplot()
    _style(ax)
    eff_label = f"Efficacy ({SPENDING_NAMES[design.efficacy_spending]})"
    ax.plot(t, c, color=_BLUE, linewidth=2, marker="o", markersize=7, label=eff_label)
    ax.fill_between(t, c, max(c.max(), 4.0) + 0.5, color=_BLUE, alpha=0.08, linewidth=0)
    for x, y in zip(t, c, strict=True):
        ax.annotate(
            f"{y:.3f}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=_INK,
        )
    if b is not None:
        fut_label = f"Futility ({SPENDING_NAMES[design.futility_spending or 'obf']}, non-binding)"
        ax.plot(t, b, color=_ORANGE, linewidth=2, marker="s", markersize=7, label=fut_label)
        ax.fill_between(t, min(b.min(), -1.0) - 0.5, b, color=_ORANGE, alpha=0.08, linewidth=0)
        for x, y in zip(t[:-1], b[:-1], strict=True):
            ax.annotate(
                f"{y:.3f}",
                (x, y),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=_INK,
            )
    ax.axhline(0, color=_MUTED, linewidth=0.8)
    ax.set_xlim(0, 1.05)
    lo = min(-1.0, b.min()) if b is not None else -1.0
    ax.set_ylim(lo - 0.5, max(4.0, c.max()) + 0.5)
    ax.set_xticks(t, [f"{v:.2f}" for v in t])
    ax.set_xlabel("Information fraction")
    ax.set_ylabel("z-statistic boundary")
    title = f"Stopping boundaries, one-sided alpha {design.alpha_one_sided:g}"
    ax.set_title(title, loc="left", fontsize=11, color=_INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out)
    return out


def plot_gsd_oc(
    design: GroupSequentialDesign, path: str | Path, n_max: float, unit: str = "patients"
) -> Path:
    """Power curve and expected sample size against the true effect, saved as PNG.

    The effect is on the scale of the design alternative (1 = the effect the trial is
    powered for). Curves are exact, with futility followed and with futility ignored.
    """
    t, c, b = design.info_rates, design.efficacy, design.futility
    theta = np.linspace(0.0, 1.5, 31)
    drifts = theta * design.drift
    scen = [("Futility followed", b, _BLUE, "-")] if b is not None else []
    scen.append(("Futility ignored" if b is not None else "No futility", None, _AQUA, "--"))
    fig = _figure(9, 3.8)
    ax1, ax2 = fig.subplots(1, 2)
    for ax in (ax1, ax2):
        _style(ax)
    for label, fut, color, ls in scen:
        rej = [crossing_probabilities(t, c, fut, d)[0].sum() for d in drifts]
        en = [expected_info_fraction(t, c, fut, d) * n_max for d in drifts]
        ax1.plot(theta, rej, color=color, linestyle=ls, linewidth=2, label=label)
        ax2.plot(theta, en, color=color, linestyle=ls, linewidth=2, label=label)
    ax1.axhline(design.power, color=_MUTED, linewidth=0.8, linestyle=":")
    ax1.axhline(design.alpha_one_sided, color=_MUTED, linewidth=0.8, linestyle=":")
    ax1.set_xlabel("True effect / design effect")
    ax1.set_ylabel("P(reject H0)")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Power curve", loc="left", fontsize=11, color=_INK)
    ax2.axhline(n_max, color=_MUTED, linewidth=0.8, linestyle=":")
    ax2.annotate(
        "maximum",
        (1.5, n_max),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=_INK_2,
    )
    ax2.set_xlabel("True effect / design effect")
    ax2.set_ylabel(f"Expected {unit}")
    ax2.set_ylim(0, n_max * 1.08)
    ax2.set_title("Expected size at stopping", loc="left", fontsize=11, color=_INK)
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out)
    return out


def plot_ssr(design: ssr.SSRDesign, path: str | Path, n_planned: float) -> Path:
    """The promising-zone rule and its effect on power, saved as PNG.

    Left: total patients against the interim z-statistic, with the zones shaded. Right:
    power of the adaptive and the planned design against the true effect (futility
    followed), on the scale of the design effect.
    """
    t1 = design.t1
    lo = design.b1 if design.b1 is not None else -1.0
    z = np.linspace(min(lo, -1.0), design.c1 + 0.5, 600)
    zones = design.zone(z)
    cont = ~np.isin(zones, ("efficacy", "futility"))
    n_tot = np.where(
        cont, (t1 + (1 - t1) * design.stage2_multiplier(z)) * n_planned, t1 * n_planned
    )
    fig = _figure(9, 3.8)
    ax1, ax2 = fig.subplots(1, 2)
    for ax in (ax1, ax2):
        _style(ax)
    prom = zones == "promising"
    if prom.any():
        ax1.axvspan(z[prom][0], z[prom][-1], color=_ORANGE, alpha=0.12, linewidth=0)
        ax1.annotate(
            "promising",
            (0.5 * (z[prom][0] + z[prom][-1]), n_planned * design.max_factor),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=_INK_2,
        )
    ax1.axvline(design.c1, color=_MUTED, linewidth=0.8, linestyle=":")
    if design.b1 is not None:
        ax1.axvline(design.b1, color=_MUTED, linewidth=0.8, linestyle=":")
    ax1.plot(z, n_tot, color=_BLUE, linewidth=2)
    ax1.set_xlabel("Interim z-statistic")
    ax1.set_ylabel("Total patients")
    ax1.set_ylim(0, n_planned * design.max_factor * 1.12)
    ax1.set_title("Re-estimated sample size", loc="left", fontsize=11, color=_INK)
    theta = np.linspace(0.0, 1.5, 31)
    adaptive = [design.operating_characteristics(th * design.gsd.drift).reject_prob for th in theta]
    planned = [design.gsd.crossing_probabilities(th * design.gsd.drift)[0].sum() for th in theta]
    ax2.plot(theta, adaptive, color=_BLUE, linewidth=2, label="Adaptive (CHW test)")
    ax2.plot(theta, planned, color=_AQUA, linewidth=2, linestyle="--", label="Planned design")
    ax2.axhline(design.gsd.power, color=_MUTED, linewidth=0.8, linestyle=":")
    ax2.set_xlabel("True effect / design effect")
    ax2.set_ylabel("P(reject H0)")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Power curve", loc="left", fontsize=11, color=_INK)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out)
    return out


def plot_decision_map(design: bayes.BayesianInterimDesign, path: str | Path) -> Path:
    """Interim go / continue / no-go regions over the responder counts, saved as PNG."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    d = design.decision_table.T  # rows: treatment responders, columns: control responders
    fig = _figure(6, 5)
    ax = fig.add_subplot()
    cmap = ListedColormap([_RED, _NEUTRAL, _BLUE])
    ax.imshow(
        d + 1, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest"
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=_INK_2, labelsize=9)
    ax.set_xlabel(f"Control responders at interim (of {design.n_interim_control})", color=_INK_2)
    ax.set_ylabel(
        f"Treatment responders at interim (of {design.n_interim_treatment})", color=_INK_2
    )
    ax.set_title(
        f"Interim decision: go if PPoS >= {design.go_threshold:g}, "
        f"no-go if PPoS <= {design.nogo_threshold:g}",
        loc="left",
        fontsize=10,
        color=_INK,
    )
    handles = [
        Patch(facecolor=_BLUE, label="Go (stop for success)"),
        Patch(facecolor=_NEUTRAL, label="Continue"),
        Patch(facecolor=_RED, label="No-go (stop for futility)"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
    )
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out)
    return out


def plot_bayes_oc(
    design: bayes.BayesianInterimDesign, pc: float, pt: float, path: str | Path
) -> Path:
    """Exact P(success), P(go) and P(no-go) against the true treatment rate, saved as PNG."""
    grid = np.linspace(0.0, 1.0, 101)
    grid = grid[(grid >= max(0.0, pc - 0.15)) & (grid <= min(1.0, pt + 0.25))]
    ocs = [design.operating_characteristics(pc, p) for p in grid]
    fig = _figure(7, 4.2)
    ax = fig.add_subplot()
    _style(ax)
    series = [
        ("P(success)", [o.prob_success for o in ocs], _BLUE, "-"),
        ("P(go at interim)", [o.prob_go for o in ocs], _AQUA, "--"),
        ("P(no-go at interim)", [o.prob_nogo for o in ocs], _ORANGE, "-."),
    ]
    for label, y, color, ls in series:
        ax.plot(grid, y, color=color, linestyle=ls, linewidth=2, label=label)
    for x, name in ((pc, "H0"), (pt, "H1")):
        ax.axvline(x, color=_MUTED, linewidth=0.8, linestyle=":")
        ax.annotate(
            name, (x, 1.0), xytext=(3, -10), textcoords="offset points", fontsize=8, color=_INK_2
        )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"True treatment response rate (control fixed at {pc:g})")
    ax.set_ylabel("Probability")
    ax.set_title("Operating characteristics (exact)", loc="left", fontsize=11, color=_INK)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out)
    return out


# --------------------------------------------------------------------------- Markdown


def gsd_markdown(
    s: Summary, sim: Summary | None = None, plots: bool = True, header: str = ""
) -> str:
    """Markdown section for a group-sequential design."""
    size = s["sample_size"]
    unit = size["unit"]
    fixed_alpha = s["fixed"]["alpha_two_sided"]
    max_line = (
        f"{size['max_events_ceil']} events"
        if unit == "events"
        else f"{size['max_n_total_ceil']} patients ({size['max_n_control_ceil']} control, "
        f"{size['max_n_treatment_ceil']} treatment)"
    )
    if unit == "events" and "max_n_total_ceil" in size:
        max_line += f", about {size['max_n_total_ceil']} patients"
    if "accrual" in size:
        max_line += ". " + accrual_text({"accrual": size["accrual"], **s})[:-1]
    design_rows = [
        ["Endpoint", _ENDPOINT_NAMES[s["endpoint"]]],
        ["Effect", effect_text(s)],
        ["Hypothesis", hypothesis_text(s)],
        ["Allocation (n_T / n_C)", f"{s['allocation']:g}"],
        ["Alpha", f"{s['alpha_one_sided']:g} one-sided"],
        ["Power", f"{s['power']:g}"],
        ["Looks (information fractions)", ", ".join(f"{v:.2f}" for v in s["looks"])],
        ["Efficacy", f"Lan-DeMets {SPENDING_NAMES[s['efficacy_spending']]} alpha spending"],
        ["Futility", futility_text(s)],
    ]
    boundaries, oc, per_look = gsd_tables(s, sim, markdown=True)
    parts = [
        "## Group-sequential design",
        header,
        format_table(["Item", "Value"], design_rows, markdown=True),
        "### Sample size",
        f"The fixed design needs {_fmt(size['fixed_total'], 2)} {unit} "
        f"(two-sided alpha {fixed_alpha:g}, the same critical value as one-sided "
        f"{s['alpha_one_sided']:g}). The inflation factor is {s['inflation_factor']:.4f}, so "
        f"the maximum is {max_line}.",
        "### Boundaries",
        boundaries,
    ]
    if plots:
        parts.append("![Stopping boundaries](boundaries.png)")
    parts += [
        "### Operating characteristics",
        "Exact values come from recursive numerical integration (Jennison & Turnbull 2000, "
        "chapter 19)."
        + (
            f" Simulated values use {sim['nsim']:,} trials with seed {sim['seed']}; each has a "
            "Monte Carlo SE of sqrt(p(1 - p) / nsim), and z = (simulated - exact) / SE; about "
            "one check in 20 should fall beyond +/-2 (marked *) by chance."
            if sim
            else ""
        ),
        oc,
        "Per-look stopping probabilities (exact; futility followed):",
        per_look,
    ]
    if plots:
        parts.append("![Operating characteristics](oc.png)")
    if s["futility_spending"] is not None:
        parts.append(
            "Futility is non-binding: the efficacy boundaries and the type I error are computed "
            "as if the futility boundary were ignored, so alpha is controlled whether or not the "
            "data monitoring committee stops for futility. Following futility lowers both the "
            "rejection rate and the expected sample size."
        )
    return "\n\n".join(p for p in parts if p)


CONVENTIONAL_ALPHA_ONE_SIDED = 0.025


def success_text(s: Summary) -> str:
    """The final success rule, noting a calibrated threshold."""
    text = f"P(p_T > p_C | data) >= {s['success_threshold']:.4g}"
    alpha = s.get("success_calibrated_to")
    if alpha is not None:
        text += f" (calibrated: worst-case type I error <= {alpha:g} one-sided)"
    return text


def needs_alpha_note(s: Summary) -> bool:
    """True when the type I error exceeds one-sided 0.025 and was not calibrated on purpose."""
    return (
        s["max_type_i_error"] > CONVENTIONAL_ALPHA_ONE_SIDED
        and s.get("success_calibrated_to") is None
    )


def alpha_note(s: Summary) -> str:
    """A warning when the Bayesian design's type I error exceeds one-sided 0.025."""
    return (
        f"Note: the type I error ({s['h0']['prob_success']:.4f} at p_C = {s['p_control']:g}, "
        f"up to {s['max_type_i_error']:.4f}) is above the conventional one-sided "
        f"{CONVENTIONAL_ALPHA_ONE_SIDED:g}. That can be acceptable for a phase 2 go/no-go "
        "decision; if it is not, raise the final success threshold or the go threshold."
    )


def bayes_markdown(
    s: Summary, sim: Summary | None = None, plots: bool = True, header: str = ""
) -> str:
    """Markdown section for a Bayesian go/no-go interim design."""
    n1c, n1t = s["n_interim_control"], s["n_interim_treatment"]
    nc, nt = s["n_final_control"], s["n_final_treatment"]
    pr_c, pr_t = s["prior_control"], s["prior_treatment"]
    rows = [
        ["Endpoint", "Binary"],
        ["Response rates", f"p_C {s['p_control']:g}, p_T {s['p_treatment']:g} under H1"],
        ["Patients at interim (control, treatment)", f"{n1c}, {n1t}"],
        ["Patients at final (control, treatment)", f"{nc}, {nt}"],
        [
            "Analysis priors",
            f"Beta({pr_c[0]:g}, {pr_c[1]:g}) control, Beta({pr_t[0]:g}, {pr_t[1]:g}) treatment",
        ],
        ["Final success", success_text(s)],
        ["Interim go", f"PPoS >= {s['go_threshold']:g} (stop, declare success)"],
        ["Interim no-go", f"PPoS <= {s['nogo_threshold']:g} (stop for futility)"],
    ]
    tables = bayes_tables(s, sim, markdown=True)
    parts = [
        "## Bayesian go/no-go interim",
        header,
        format_table(["Item", "Value"], rows, markdown=True),
        "### Operating characteristics",
        "Exact values sum binomial probabilities over the final success table and the interim "
        "decision table. P(success) under H0 is the type I error of the whole procedure; "
        f"the largest over common response rates from 0.05 to 0.95 is "
        f"{s['max_type_i_error']:.4f} (at {s['max_type_i_error_at']:g}).",
        tables[0],
    ]
    if needs_alpha_note(s):
        parts.append(alpha_note(s))
    if s["assurance"] is not None:
        a = s["assurance"]

        def law(v: Any) -> str:
            return f"Beta({v[0]:g}, {v[1]:g})" if isinstance(v, list) else f"fixed at {v:g}"

        parts.append(
            f"Assurance (O'Hagan, Stevens and Campbell 2005) averages P(success) over the "
            f"design prior: p_C {law(a['design_prior_control'])}, "
            f"p_T {law(a['design_prior_treatment'])}. It is {a['prob_success']:.4f}."
        )
    if sim is not None:
        parts += [
            f"Simulation check: {sim['nsim']:,} trials per scenario, seed {sim['seed']}, "
            "through the M3 engine. z = (simulated - exact) / MC SE; about one check in 20 should "
            "fall beyond +/-2 (marked *) by chance.",
            tables[1],
        ]
    if plots:
        parts.append("![Operating characteristics](bayes_oc.png)")
    parts += ["### Interim decision rule", tables[-1]]
    if plots:
        parts.append("![Interim decision map](decision_map.png)")
    if s["interim"] is not None:
        i = s["interim"]
        parts += [
            "### Observed interim",
            f"{i['xc']}/{n1c} control and {i['xt']}/{n1t} treatment responders: "
            f"P(p_T > p_C | data) = {i['posterior_prob_superiority']:.4f}, "
            f"PPoS = {i['ppos']:.4f}, decision **{i['decision']}**.",
        ]
    return "\n\n".join(p for p in parts if p)


def ssr_markdown(s: Summary, plots: bool = True) -> str:
    """Markdown section for promising-zone sample size re-estimation."""
    zones, oc, *sim = ssr_tables(s, markdown=True)
    ex = s["exact"]
    first = ex["effects"][0]
    parts = [
        "## Sample size re-estimation",
        f"At the interim ({s['interim_fraction']:.2f} of the planned information, "
        f"{s['n_interim']} patients), the conditional power at the observed trend decides the "
        f"second stage. In the promising zone (conditional power {s['cp_min']:g} to "
        f"{s['cp_target']:g}) the sample size is raised to reach {s['cp_target']:g}, up to "
        f"{s['n_cap']} patients ({s['max_factor']:g} times the planned {s['n_planned']}). "
        "Elsewhere the planned sample size is kept (Mehta and Pocock 2011).",
        zones,
        "The final test weights the stages by their *planned* information (Cui, Hung and Wang "
        f"1999), so the type I error stays at {ex['type_i_error']:.4f} whatever the interim "
        f"result. Pooling all patients instead gives {ex['type_i_error_naive']:.4f}. At the "
        f"design effect the adaptive design has power {first['reject_prob']:.3f} "
        f"(planned {first['planned_reject_prob']:.3f}) with "
        f"{first['expected_n_total']:.0f} expected patients "
        f"(planned {first['planned_expected_n_total']:.0f}). Exact values, by integration over "
        "the interim z-statistic:",
        oc,
    ]
    if sim:
        parts += ["Simulated checks:", sim[0]]
    if "interim" in s:
        i = s["interim"]
        parts.append(
            f"**Observed interim z = {i['z1']:.3f}** (observed effect "
            f"{i['observed_effect_ratio']:.2f} x design): {i['zone']} zone, conditional power "
            f"{i['cp_observed']:.3f} at the observed trend ({i['cp_design']:.3f} at the design "
            f"effect). Total patients: {i['n_total']}; conditional power at the observed "
            f"trend after re-estimation {i['cp_observed_after']:.3f}."
        )
    if plots:
        parts.append("![Re-estimated sample size against the interim z, and power curves](ssr.png)")
    return "\n\n".join(parts)


def write_report(
    out_dir: str | Path,
    version: str,
    *,
    source: str | None = None,
    gsd: tuple[GroupSequentialDesign, Summary, Summary | None] | None = None,
    bayes_design: tuple[bayes.BayesianInterimDesign, Summary, Summary | None] | None = None,
    ssr_design: tuple[ssr.SSRDesign, Summary] | None = None,
) -> list[Path]:
    """Write ``report.md`` and its PNG plots into ``out_dir`` (created if needed).

    Parameters
    ----------
    out_dir : str or Path
        Output folder.
    version : str
        Package version, printed in the report.
    source : str, optional
        Name of the design file, printed in the report.
    gsd : (GroupSequentialDesign, summary, simulation summary or None), optional
        Group-sequential part: writes ``boundaries.png`` and ``oc.png``.
    bayes_design : (BayesianInterimDesign, summary, simulation summary or None), optional
        Bayesian part: writes ``bayes_oc.png`` and ``decision_map.png``.
    ssr_design : (SSRDesign, summary), optional
        Sample size re-estimation, reported after the planned design: writes ``ssr.png``.

    Returns
    -------
    list of Path
        The files written, report first.
    """
    if gsd is None and bayes_design is None:
        raise ValueError("nothing to report: give a group-sequential or a Bayesian design")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seeds = [x[2]["seed"] for x in (gsd, bayes_design) if x is not None and x[2] is not None]
    intro = (
        f"Generated by trialdesign {version}"
        + (f" from `{source}`" if source else "")
        + (f", seed {seeds[0]}" if seeds else ", no simulation (exact results only)")
        + ". Group-sequential alpha is one-sided; fixed-design alpha is two-sided."
    )
    parts = ["# Trial design report", intro]
    if gsd is not None:
        design, s, sim = gsd
        n_max = (
            s["sample_size"]["max_events_ceil"]
            if s["sample_size"]["unit"] == "events"
            else s["sample_size"]["max_n_total_ceil"]
        )
        written.append(plot_boundaries(design, out / "boundaries.png"))
        written.append(plot_gsd_oc(design, out / "oc.png", n_max, s["sample_size"]["unit"]))
        parts.append(gsd_markdown(s, sim))
    if ssr_design is not None:
        sd, ss = ssr_design
        written.append(plot_ssr(sd, out / "ssr.png", ss["n_planned"]))
        parts.append(ssr_markdown(ss))
    if bayes_design is not None:
        bd, s, sim = bayes_design
        written.append(plot_bayes_oc(bd, s["p_control"], s["p_treatment"], out / "bayes_oc.png"))
        written.append(plot_decision_map(bd, out / "decision_map.png"))
        parts.append(bayes_markdown(s, sim))
    parts += [
        "## References",
        "- Jennison, C. and Turnbull, B. W. (2000). *Group Sequential Methods with Applications "
        "to Clinical Trials*. Chapman & Hall/CRC.\n"
        "- Lan, K. K. G. and DeMets, D. L. (1983). Discrete sequential boundaries for clinical "
        "trials. *Biometrika* 70, 659-663.\n"
        "- Schoenfeld, D. A. (1983). Sample-size formula for the proportional-hazards "
        "regression model. *Biometrics* 39, 499-503.\n"
        "- O'Hagan, A., Stevens, J. W. and Campbell, M. J. (2005). Assurance in clinical trial "
        "design. *Pharmaceutical Statistics* 4, 187-201.\n"
        "- Lee, J. J. and Liu, D. D. (2008). A predictive probability design for phase II "
        "cancer clinical trials. *Clinical Trials* 5, 93-106.\n"
        "- Cui, L., Hung, H. M. J. and Wang, S.-J. (1999). Modification of sample size in group "
        "sequential clinical trials. *Biometrics* 55, 853-857.\n"
        "- Mehta, C. R. and Pocock, S. J. (2011). Adaptive increase in sample size when interim "
        "results are promising. *Statistics in Medicine* 30, 3267-3284.",
    ]
    md = out / "report.md"
    md.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return [md, *written]
