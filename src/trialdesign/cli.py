"""Command-line interface for trialdesign.

Four commands, ``design``, ``simulate``, ``bayes`` and ``report``. Each reads its inputs from
flags, from a TOML design file given with ``--config`` (see :mod:`trialdesign.config`), or
both; a flag overrides the file. Results print as plain tables, or as JSON with ``--json``.
Only ``report --out DIR`` writes to disk. Every simulation prints its seed and every run
prints the package version, so a result can be reproduced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import typer

from trialdesign import __version__, accrual, bayes, config, fixed, report, ssr
from trialdesign.boundaries import GroupSequentialDesign, group_sequential_design
from trialdesign.patients import simulate_design_patients, simulate_survival_ssr
from trialdesign.simulate import DesignSimulation, simulate_design
from trialdesign.spending import parse_spending

app = typer.Typer(
    name="trialdesign",
    help="Group-sequential and Bayesian design for two-arm clinical trials.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"trialdesign {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the package version and exit.",
        ),
    ] = None,
) -> None:
    """Group-sequential and Bayesian design for two-arm clinical trials."""


class Endpoint(StrEnum):
    normal = "normal"
    binary = "binary"
    survival = "survival"


# ------------------------------------------------------------------------- options

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", help="TOML design file; flags override it.")
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON output.")]
EndpointOpt = Annotated[Endpoint | None, typer.Option(help="Primary endpoint type.")]
DeltaOpt = Annotated[float | None, typer.Option(help="Normal: difference in means (T - C).")]
SdOpt = Annotated[float | None, typer.Option(help="Normal: common standard deviation.")]
PcOpt = Annotated[float | None, typer.Option(help="Binary: control response rate.")]
PtOpt = Annotated[float | None, typer.Option(help="Binary: treatment response rate.")]
HrOpt = Annotated[float | None, typer.Option(help="Survival: hazard ratio (T vs C).")]
Delta0Opt = Annotated[
    float | None,
    typer.Option(
        help="Normal/binary non-inferiority: difference (T - C) under H0, e.g. -0.1 for a "
        "margin of 0.1 when higher is better [default: 0, superiority]."
    ),
]
MedianControlOpt = Annotated[
    float | None, typer.Option(help="Survival accrual: control-arm median survival (months).")
]
AccrualTimeOpt = Annotated[
    str | None,
    typer.Option(
        help='Survival accrual: duration ("24") or end of each segment ("6,24"), in months.'
    ),
]
AccrualRateOpt = Annotated[
    str | None,
    typer.Option(help='Survival accrual: patients per month, one per segment ("10,25").'),
]
FollowupOpt = Annotated[
    float | None,
    typer.Option(
        help="Survival accrual: minimum follow-up after the last patient enters (months). "
        "Give two of --accrual-time, --accrual-rate and --followup."
    ),
]
DropoutOpt = Annotated[
    float | None,
    typer.Option(help="Survival accrual: share lost to dropout by --dropout-time [default: 0]."),
]
DropoutTimeOpt = Annotated[
    float | None, typer.Option(help="Survival accrual: time for --dropout [default: 12].")
]
Hr0Opt = Annotated[
    float | None,
    typer.Option(
        help="Survival non-inferiority: hazard ratio under H0, e.g. 1.3 [default: 1, superiority]."
    ),
]
ProbEventOpt = Annotated[
    float | None, typer.Option(help="Survival: probability of an event by the final analysis.")
]
AlphaOneOpt = Annotated[
    float | None,
    typer.Option(help="One-sided significance level [default: 0.025, two-sided 0.05]."),
]
AlphaTwoOpt = Annotated[float | None, typer.Option(help="Two-sided significance level.")]
PowerOpt = Annotated[float | None, typer.Option(help="Target power [default: 0.80].")]
AllocationOpt = Annotated[
    float | None, typer.Option(help="Allocation ratio n_T / n_C [default: 1].")
]
MethodOpt = Annotated[
    str | None, typer.Option(help='Normal only: "z" (normal approx., default) or "t".')
]
LooksOpt = Annotated[
    str | None,
    typer.Option(
        help='Number of equally spaced looks ("3") or information fractions '
        '("0.33,0.67,1"). Leave out for a fixed design.'
    ),
]
EfficacyOpt = Annotated[
    str | None,
    typer.Option(
        "--efficacy",
        "--spending",
        help='Alpha spending: "obf" (default), "pocock", "hsd:<gamma>" (e.g. "hsd:-4"), '
        '"kd:<rho>" (e.g. "kd:3") or "user:<f1>,...,1" (cumulative fractions of alpha).',
    ),
]
FutilityOpt = Annotated[
    str | None,
    typer.Option(help='Beta spending for futility: "none" (default) or any --efficacy choice.'),
]
BindingOpt = Annotated[
    bool | None,
    typer.Option(
        "--binding-futility/--non-binding-futility",
        help="Make futility binding: smaller trial, but alpha holds only if every futility "
        "stop is made [default: non-binding].",
        show_default=False,
    ),
]
NsimOpt = Annotated[int | None, typer.Option(help="Number of simulated trials.")]
SeedOpt = Annotated[int | None, typer.Option(help="Random seed; drawn and printed if omitted.")]
EngineOpt = Annotated[
    str | None,
    typer.Option(
        help="Simulation engine: 'score' (canonical z-scale process, fast) or 'patient' "
        "(simulated patients and the real test statistic) [default: score]."
    ),
]
NOpt = Annotated[
    int | None, typer.Option("--n", help="Bayes: control patients at the final analysis.")
]
InterimOpt = Annotated[int | None, typer.Option(help="Bayes: control patients at the interim.")]
PriorCOpt = Annotated[
    str | None, typer.Option(help='Bayes: control Beta analysis prior "a,b" [default: 1,1].')
]
PriorTOpt = Annotated[
    str | None, typer.Option(help='Bayes: treatment Beta analysis prior "a,b" [default: 1,1].')
]
SuccessOpt = Annotated[
    float | None,
    typer.Option(help="Bayes: final success if P(p_T > p_C | data) >= this [default: 0.98]."),
]
CalibrateOpt = Annotated[
    bool | None,
    typer.Option(
        "--calibrate/--no-calibrate",
        help="Bayes: choose the success threshold that holds the worst-case type I error at "
        "--alpha-one-sided (default 0.025).",
    ),
]
GoOpt = Annotated[float | None, typer.Option(help="Bayes: go if PPoS >= this [default: 0.9].")]
NogoOpt = Annotated[float | None, typer.Option(help="Bayes: no-go if PPoS <= this [default: 0.1].")]
DesignPriorCOpt = Annotated[
    str | None,
    typer.Option(help='Bayes assurance: control design prior, a rate "0.3" or Beta "a,b".'),
]
DesignPriorTOpt = Annotated[
    str | None,
    typer.Option(help='Bayes assurance: treatment design prior, a rate "0.45" or Beta "a,b".'),
]
InterimFractionOpt = Annotated[
    float | None,
    typer.Option(help="SSR: information fraction of the interim look [default: 0.5]."),
]
CpMinOpt = Annotated[
    float | None,
    typer.Option(help="SSR: lower edge of the promising zone (conditional power) [default: 0.36]."),
]
CpTargetOpt = Annotated[
    float | None,
    typer.Option(help="SSR: conditional power aimed for in the promising zone [default: 0.8]."),
]
MaxFactorOpt = Annotated[
    float | None,
    typer.Option(
        help="SSR: cap on total patients (survival: events), as a multiple of the planned "
        "maximum [default: 2; 1.5 for survival]."
    ),
]
ExtendOpt = Annotated[
    str | None,
    typer.Option(
        help='SSR, survival: find extra events by "followup" (follow the enrolled patients '
        'longer) or "accrual" (enrol more at the last accrual rate) [default: followup].'
    ),
]
Z1Opt = Annotated[
    float | None,
    typer.Option(
        help="SSR: observed interim z-statistic; prints the zone and the new sample size."
    ),
]
XcOpt = Annotated[int | None, typer.Option(help="Bayes: observed control responders at interim.")]
XtOpt = Annotated[int | None, typer.Option(help="Bayes: observed treatment responders at interim.")]


# --------------------------------------------------------------------------- parsing


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def _whole(value: Any, name: str) -> int:
    x = _number(value, name)
    if x != int(x):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    return int(x)


def _numbers(value: Any, name: str) -> list[float]:
    if isinstance(value, str):
        value = [v for v in value.replace(" ", "").split(",") if v]
    if not isinstance(value, list | tuple):
        value = [value]
    return [_number(v, name) for v in value]


def _pair(value: Any, name: str) -> tuple[float, float]:
    v = _numbers(value, name)
    if len(v) != 2:
        raise ValueError(f'{name} must be a Beta prior "a,b"')
    return v[0], v[1]


def _design_prior(value: Any, name: str) -> bayes.DesignPrior | None:
    if value is None:
        return None
    v = _numbers(value, name)
    if len(v) == 1:
        return v[0]
    if len(v) == 2:
        return v[0], v[1]
    raise ValueError(f'{name} must be a response rate or a Beta prior "a,b"')


def _looks(value: Any) -> int | list[float]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    v = _numbers(value, "looks")
    if len(v) == 1 and v[0] == int(v[0]) and v[0] >= 1 and not str(value).startswith("1."):
        return int(v[0])
    return v


def _spending(value: Any, name: str, allow_none: bool) -> str | None:
    s = str(value).strip().lower()
    if allow_none and s in ("none", ""):
        return None
    try:
        return parse_spending(s).label
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from None


def _seed(value: Any) -> int:
    if value is None:  # draw a short seed so it can be printed and reused
        return int(np.random.SeedSequence().generate_state(1)[0])
    return _whole(value, "seed")


@dataclass(frozen=True)
class _DesignInputs:
    endpoint: str
    effect: dict[str, Any]
    alpha_one_sided: float
    power: float
    allocation: float
    method: fixed.NormalMethod
    looks: int | list[float] | None
    efficacy: str
    futility: str | None
    binding_futility: bool = False


_REQUIRED = {"normal": ("delta", "sd"), "binary": ("pc", "pt"), "survival": ("hr",)}


def _resolve_alpha(cfg: config.Config, flags: dict[str, Any]) -> float:
    one, two = flags.get("alpha_one_sided"), flags.get("alpha_two_sided")
    if one is not None and two is not None:
        raise ValueError("give --alpha-one-sided or --alpha-two-sided, not both")
    if one is None and two is None:
        d = cfg.get("design", {})
        one, two = d.get("alpha_one_sided"), d.get("alpha_two_sided")
        if one is not None and two is not None:
            raise ValueError("[design] has both alpha_one_sided and alpha_two_sided; keep one")
    if two is not None:
        return _number(two, "alpha_two_sided") / 2
    return 0.025 if one is None else _number(one, "alpha_one_sided")


def _accrual_inputs(acc: dict[str, Any]) -> dict[str, Any]:
    if acc.get("median_control") is None:
        raise ValueError("--median-control is required for an accrual model")
    out: dict[str, Any] = {"median_control": _number(acc["median_control"], "median_control")}
    for key in ("accrual_time", "accrual_rate"):
        if acc.get(key) is not None:
            v = _numbers(acc[key], key)
            out[key] = v[0] if len(v) == 1 else v
    for key in ("followup", "dropout", "dropout_time"):
        if acc.get(key) is not None:
            out[key] = _number(acc[key], key)
    return out


def _resolve_design(cfg: config.Config, flags: dict[str, Any]) -> _DesignInputs:
    d = config.merge(cfg, "design", flags)
    if d.get("endpoint") is None:
        raise ValueError("--endpoint is required (or endpoint in [design])")
    endpoint = str(d["endpoint"])
    if endpoint not in _REQUIRED:
        raise ValueError(f'endpoint must be "normal", "binary" or "survival", got {endpoint!r}')
    for key in _REQUIRED[endpoint]:
        if d.get(key) is None:
            raise ValueError(f"--{key} is required for the {endpoint} endpoint")
    effect: dict[str, Any] = {k: _number(d[k], k) for k in _REQUIRED[endpoint]}
    if endpoint == "survival" and d.get("prob_event") is not None:
        effect["prob_event"] = _number(d["prob_event"], "prob_event")
    acc = {k: v for k, v in config.merge(cfg, "accrual", flags).items() if v is not None}
    if acc:
        if endpoint != "survival":
            raise ValueError("accrual settings apply only to the survival endpoint")
        effect["accrual_inputs"] = _accrual_inputs(acc)
    null_key, other = ("hr0", "delta0") if endpoint == "survival" else ("delta0", "hr0")
    if d.get(other) is not None:
        raise ValueError(f"--{other} does not apply to the {endpoint} endpoint; use --{null_key}")
    if d.get(null_key) is not None:
        effect[null_key] = _number(d[null_key], null_key)
    method = str(d.get("method", "z"))
    if method not in ("z", "t"):
        raise ValueError('method must be "z" or "t"')
    seq = config.merge(cfg, "sequential", flags)
    binding = seq.get("binding_futility")
    if binding is not None and not isinstance(binding, bool):
        raise ValueError("binding_futility must be true or false")
    futility = _spending(seq.get("futility", "none"), "futility", True)
    if binding and futility is None:
        raise ValueError("binding futility needs a futility spending (--futility)")
    looks = seq.get("looks")
    if looks is None and (seq.get("efficacy") is not None or seq.get("futility") is not None):
        raise ValueError("--looks is required for a group-sequential design")
    return _DesignInputs(
        endpoint=endpoint,
        effect=effect,
        alpha_one_sided=_resolve_alpha(cfg, flags),
        power=_number(d.get("power", 0.80), "power"),
        allocation=_number(d.get("allocation", 1.0), "allocation"),
        method=cast(fixed.NormalMethod, method),
        looks=None if looks is None else _looks(looks),
        efficacy=cast(str, _spending(seq.get("efficacy", "obf"), "efficacy", False)),
        futility=futility,
        binding_futility=bool(binding),
    )


def _fixed(inp: _DesignInputs, alpha_two_sided: float) -> dict[str, Any]:
    return report.fixed_summary(
        inp.endpoint,
        **inp.effect,
        alpha_two_sided=alpha_two_sided,
        power=inp.power,
        allocation=inp.allocation,
        method=inp.method,
    )


def _sequential(inp: _DesignInputs) -> tuple[GroupSequentialDesign, dict[str, Any]]:
    assert inp.looks is not None
    gsd = group_sequential_design(
        inp.looks,
        alpha_one_sided=inp.alpha_one_sided,
        power=inp.power,
        efficacy_spending=inp.efficacy,
        futility_spending=inp.futility,
        binding_futility=inp.binding_futility,
    )
    return gsd, report.sequential_summary(_fixed(inp, 2 * inp.alpha_one_sided), gsd)


def _n_max(s: dict[str, Any]) -> int:
    size = s["sample_size"]
    return size["max_events_ceil"] if size["unit"] == "events" else size["max_n_total_ceil"]


def _simulation_settings(cfg: config.Config, flags: dict[str, Any], default_nsim: int | None):
    sim = config.merge(cfg, "simulation", flags)
    nsim = sim.get("nsim", default_nsim)
    if nsim is None:
        return None, None
    return _whole(nsim, "nsim"), _seed(sim.get("seed"))


def _engine(cfg: config.Config, flags: dict[str, Any]) -> str:
    engine = str(config.merge(cfg, "simulation", flags).get("engine", "score"))
    if engine not in ("score", "patient"):
        raise ValueError(f'engine must be "score" or "patient", got {engine!r}')
    return engine


def _patient_simulation(
    inp: _DesignInputs, gsd: GroupSequentialDesign, s: dict[str, Any], n_sim: int, seed: int
) -> DesignSimulation:
    """Patient-level H0 and H1 simulation of a group-sequential design built by the CLI."""
    size, eff = s["sample_size"], inp.effect
    if inp.endpoint == "normal":
        return simulate_design_patients(
            gsd,
            "normal",
            n_control=size["max_n_control_ceil"],
            n_treatment=size["max_n_treatment_ceil"],
            delta=eff["delta"],
            sd=eff["sd"],
            delta0=eff.get("delta0", 0.0),
            n_sim=n_sim,
            seed=seed,
        )
    if inp.endpoint == "binary":
        return simulate_design_patients(
            gsd,
            "binary",
            n_control=size["max_n_control_ceil"],
            n_treatment=size["max_n_treatment_ceil"],
            p_control=eff["pc"],
            p_treatment=eff["pt"],
            delta0=eff.get("delta0", 0.0),
            n_sim=n_sim,
            seed=seed,
        )
    if eff.get("hr0", 1.0) != 1.0:
        raise ValueError("the patient engine simulates survival superiority only (hr0 = 1)")
    if "accrual_inputs" not in eff:
        raise ValueError(
            "the patient engine needs an accrual model for survival: give --median-control "
            "and two of --accrual-time, --accrual-rate and --followup"
        )
    study = accrual.survival_study(
        size["max_events"], eff["hr"], allocation=inp.allocation, **eff["accrual_inputs"]
    )
    return simulate_design_patients(
        gsd,
        "survival",
        study=study,
        n_total=size["max_n_total_ceil"],
        events=size["max_events_ceil"],
        n_sim=n_sim,
        seed=seed,
    )


def _resolve_bayes(cfg: config.Config, flags: dict[str, Any]):
    b = config.merge(cfg, "bayes", flags)
    for key in ("pc", "pt", "n", "interim"):
        if b.get(key) is None:
            raise ValueError(f"--{key} is required for a Bayesian interim design")
    common = {
        "n_interim": _whole(b["interim"], "interim"),
        "n_final": _whole(b["n"], "n"),
        "allocation": _number(b.get("allocation", 1.0), "allocation"),
        "prior_control": _pair(b.get("prior_c", (1, 1)), "prior_c"),
        "prior_treatment": _pair(b.get("prior_t", (1, 1)), "prior_t"),
        "go_threshold": _number(b.get("go", 0.9), "go"),
        "nogo_threshold": _number(b.get("nogo", 0.1), "nogo"),
    }
    calibrate = b.get("calibrate", False)
    if not isinstance(calibrate, bool):
        raise ValueError("calibrate must be true or false")
    if calibrate:
        if b.get("success") is not None:
            raise ValueError("give success or calibrate, not both")
        alpha = _number(b.get("alpha_one_sided", 0.025), "alpha_one_sided")
        success = bayes.calibrate_success_threshold(alpha_one_sided=alpha, **common)
    else:
        success = _number(b.get("success", bayes.DEFAULT_SUCCESS_THRESHOLD), "success")
    design = bayes.bayesian_interim_design(success_threshold=success, **common)
    extra = {
        "pc": _number(b["pc"], "pc"),
        "pt": _number(b["pt"], "pt"),
        "design_prior_c": _design_prior(b.get("design_prior_c"), "design_prior_c"),
        "design_prior_t": _design_prior(b.get("design_prior_t"), "design_prior_t"),
        "xc": None if b.get("xc") is None else _whole(b["xc"], "xc"),
        "xt": None if b.get("xt") is None else _whole(b["xt"], "xt"),
        "calibrated_to": (
            _number(b.get("alpha_one_sided", 0.025), "alpha_one_sided") if calibrate else None
        ),
    }
    return design, extra


def _bayes_run(design: bayes.BayesianInterimDesign, extra: dict[str, Any], nsim, seed):
    priors = {"design_prior_c": extra["design_prior_c"], "design_prior_t": extra["design_prior_t"]}
    s = report.bayes_summary(
        design, extra["pc"], extra["pt"], xc=extra["xc"], xt=extra["xt"], **priors
    )
    s["success_calibrated_to"] = extra["calibrated_to"]
    sim = None
    if nsim is not None:
        sim = report.bayes_simulation_summary(
            design, extra["pc"], extra["pt"], nsim, seed, **priors
        )
    return s, sim


# --------------------------------------------------------------------------- output


def _resolve_adaptation(
    cfg: config.Config, flags: dict[str, Any], required: bool
) -> dict[str, Any] | None:
    a = {k: v for k, v in config.merge(cfg, "adaptation", flags).items() if v is not None}
    if not a and not required:
        return None
    extend = a.get("extend", "followup")
    if extend not in ("followup", "accrual"):
        raise ValueError(f'extend must be "followup" or "accrual", got {extend!r}')
    return {
        "interim": _number(a.get("interim_fraction", 0.5), "interim_fraction"),
        "cp_min": _number(a.get("cp_min", 0.36), "cp_min"),
        "cp_target": _number(a.get("cp_target", 0.80), "cp_target"),
        "max_factor": None
        if a.get("max_factor") is None
        else _number(a["max_factor"], "max_factor"),
        "z1": None if a.get("z1") is None else _number(a["z1"], "z1"),
        "extend": extend,
        "extend_given": "extend" in a,
    }


def _ssr_flags(cfg: config.Config, flags: dict[str, Any], adapt: dict[str, Any]) -> dict:
    """Flags with looks set to [interim_fraction, 1]; any other looks given is an error."""
    looks = config.merge(cfg, "sequential", flags).get("looks")
    two = [adapt["interim"], 1.0]
    if looks is not None and _looks(looks) not in (2, two):
        raise ValueError(
            "sample size re-estimation uses two looks, [interim_fraction, 1]; "
            "leave out --looks or set interim_fraction in [adaptation]"
        )
    return {**flags, "looks": two}


def _ssr_run(cfg: config.Config, flags: dict[str, Any], adapt: dict[str, Any], nsim, seed):
    """The planned two-look design, the adaptive design and its summary."""
    inp = _resolve_design(cfg, _ssr_flags(cfg, flags, adapt))
    survival = inp.endpoint == "survival"
    if adapt["extend_given"] and not survival:
        raise ValueError("extend applies only to the survival endpoint")
    max_factor = adapt["max_factor"]
    if max_factor is None:
        max_factor = 1.5 if survival else 2.0
    gsd, planned = _sequential(inp)
    design = ssr.SSRDesign(
        gsd, cp_min=adapt["cp_min"], cp_target=adapt["cp_target"], max_factor=max_factor
    )
    if survival:
        return design, _survival_ssr_summary(inp, planned, design, adapt, nsim, seed)
    sims = None
    if nsim is not None:
        rng = np.random.default_rng(seed)
        sims = [
            design.simulate(0.0, nsim, rng, use_futility=False),
            design.simulate(gsd.drift, nsim, rng),
        ]
        sims = [replace(x, seed=seed) for x in sims]
    return design, report.ssr_summary(planned, design, adapt["z1"], sims)


def _survival_ssr_summary(
    inp: _DesignInputs,
    planned: dict[str, Any],
    design: ssr.SSRDesign,
    adapt: dict[str, Any],
    nsim: int | None,
    seed: int | None,
) -> dict[str, Any]:
    """Event re-estimation: needs the accrual model, which sets patients and calendar time."""
    eff, size = inp.effect, planned["sample_size"]
    if eff.get("hr0", 1.0) != 1.0:
        raise ValueError("event re-estimation is for survival superiority only (hr0 = 1)")
    if "accrual_inputs" not in eff:
        raise ValueError(
            "event re-estimation needs an accrual model: give --median-control and two of "
            "--accrual-time, --accrual-rate and --followup"
        )
    study = accrual.survival_study(
        size["max_events"], eff["hr"], allocation=inp.allocation, **eff["accrual_inputs"]
    )
    adaptive = ssr.SurvivalSSR(
        design,
        study,
        events=size["max_events_ceil"],
        n_total=size["max_n_total_ceil"],
        extend=adapt["extend"],
    )
    sims = None
    if nsim is not None:
        if adaptive.extend != "followup":
            raise ValueError(
                'simulated checks of event re-estimation need extend = "followup"; leave out --nsim'
            )
        s0, s1 = (int(x) for x in np.random.SeedSequence(seed).generate_state(2))
        sims = [
            simulate_survival_ssr(adaptive, 1.0, use_futility=False, n_sim=nsim, seed=s0),
            simulate_survival_ssr(adaptive, eff["hr"], n_sim=nsim, seed=s1),
        ]
        sims = [replace(x, seed=seed) for x in sims]
    return report.ssr_summary(planned, design, adapt["z1"], sims, survival=adaptive)


def _print_ssr(s: dict[str, Any]) -> None:
    planned = s["planned"]
    noun = s.get("unit", "patients")
    pairs = {
        "endpoint": planned["endpoint"],
        "effect": report.effect_text(planned),
        "hypothesis": report.hypothesis_text(planned),
        "alpha_one_sided": planned["alpha_one_sided"],
        "power (planned)": planned["power"],
        "efficacy": report.efficacy_text(planned),
        "futility": report.futility_text(planned),
        "interim_fraction": s["interim_fraction"],
        f"{noun} at interim": s["n_interim"],
        f"planned max {noun}": s["n_planned"],
        f"cap on {noun}": s["n_cap"],
        "efficacy z (interim, final)": f"{s['c1']:.4f}, {s['c2']:.4f}",
        "promising zone (CP)": f"{s['cp_min']:g} to {s['cp_target']:g}",
    }
    if "survival" in s:
        sv = s["survival"]
        pairs.update(
            {
                "patients (planned)": sv["n_total"],
                "extra events from": sv["extend"],
                "interim / final month": f"{sv['interim_time']:.1f}, {sv['planned_duration']:.1f}",
                "final month at cap": (
                    f"{sv['cap_duration']:.1f} ({sv['cap_patients']:.0f} patients)"
                ),
            }
        )
    _print_pairs(pairs)
    zones, oc, *sim = report.ssr_tables(s)
    typer.echo(f"\nZones\n{zones}\n\nOperating characteristics (exact)\n{oc}")
    if sim:
        typer.echo(f"\nSimulated checks\n{sim[0]}")
    if "interim" in s:
        i = s["interim"]
        typer.echo(
            f"\nInterim z1 = {i['z1']:.3f}: {i['zone']} zone, CP at observed trend "
            f"{i['cp_observed']:.4f} (at design effect {i['cp_design']:.4f}); total {noun} "
            f"{i['n_total']}, CP at observed trend after re-estimation "
            f"{i['cp_observed_after']:.4f}"
        )
        if "expected_duration" in i:
            typer.echo(
                f"Final analysis expected at month {i['expected_duration']:.1f} with "
                f"{i['expected_patients']:.0f} patients (at the design hazard ratio)"
            )


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"Error: {exc}", err=True)
    return typer.Exit(code=2)


def _print_json(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps({**payload, "version": __version__}, indent=2))


def _print_pairs(pairs: dict[str, Any]) -> None:
    width = max(len(k) for k in pairs)
    for key, value in pairs.items():
        if value is None:
            continue
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        typer.echo(f"  {key:<{width}}  {shown}")


def _print_sequential(s: dict[str, Any], sim: dict[str, Any] | None = None) -> None:
    size = s["sample_size"]
    unit = size["unit"]
    pairs: dict[str, Any] = {
        "endpoint": s["endpoint"],
        "effect": report.effect_text(s),
        "hypothesis": report.hypothesis_text(s),
        "allocation": s["allocation"],
        "alpha_one_sided": s["alpha_one_sided"],
        "power": s["power"],
        "efficacy": report.efficacy_text(s),
        "futility": report.futility_text(s),
        f"fixed {unit} (alpha_two_sided {s['fixed']['alpha_two_sided']:g})": size["fixed_total"],
        "inflation_factor": s["inflation_factor"],
    }
    if unit == "events":
        pairs["max_events"] = size["max_events_ceil"]
        pairs["max_n_total"] = size.get("max_n_total_ceil")
        if "accrual" in size:
            pairs["study_duration (months)"] = size["accrual"]["study_duration"]
    else:
        pairs["max_n_control"] = size["max_n_control_ceil"]
        pairs["max_n_treatment"] = size["max_n_treatment_ceil"]
        pairs["max_n_total"] = size["max_n_total_ceil"]
    _print_pairs(pairs)
    if "accrual" in size:
        typer.echo(f"\n{report.accrual_text({**s, 'accrual': size['accrual']})}")
    boundaries, oc, per_look = report.gsd_tables(s, sim)
    for title, table in (
        ("Boundaries", boundaries),
        ("Operating characteristics", oc),
        ("Per-look stopping probabilities (exact, futility followed)", per_look),
    ):
        typer.echo(f"\n{title}\n{table}")
    if sim is not None:
        h0, h1 = sim["h0"], sim["h1"]
        rows = [
            [k + 1, *row]
            for k, row in enumerate(
                zip(
                    h0["prob_efficacy"],
                    h0["prob_futility"],
                    h1["prob_efficacy"],
                    h1["prob_futility"],
                    strict=True,
                )
            )
        ]
        headers = ["Look", "P(eff) H0", "P(fut) H0", "P(eff) H1", "P(fut) H1"]
        typer.echo("\nPer-look stopping probabilities (simulated, futility followed)")
        typer.echo(report.format_table(headers, rows))


def _print_bayes(s: dict[str, Any], sim: dict[str, Any] | None) -> None:
    pr_c, pr_t = s["prior_control"], s["prior_treatment"]
    _print_pairs(
        {
            "patients at interim (C, T)": f"{s['n_interim_control']}, {s['n_interim_treatment']}",
            "patients at final (C, T)": f"{s['n_final_control']}, {s['n_final_treatment']}",
            "analysis priors (C; T)": f"Beta({pr_c[0]:g}, {pr_c[1]:g}); "
            f"Beta({pr_t[0]:g}, {pr_t[1]:g})",
            "final success": report.success_text(s),
            "interim go / no-go": f"PPoS >= {s['go_threshold']:g} / <= {s['nogo_threshold']:g}",
            "max type I error": f"{s['max_type_i_error']:.4f} "
            f"(over common rates 0.05-0.95, worst at {s['max_type_i_error_at']:g})",
        }
    )
    tables = report.bayes_tables(s, sim)
    typer.echo(f"\nOperating characteristics (exact)\n{tables[0]}")
    if report.needs_alpha_note(s):
        typer.echo(report.alpha_note(s))
    if s["assurance"] is not None:
        typer.echo(f"Assurance = {s['assurance']['prob_success']:.4f}")
    if sim is not None:
        typer.echo(f"\nSimulation check\n{tables[1]}")
    if s["interim"] is not None:
        i = s["interim"]
        typer.echo(
            f"\nInterim {i['xc']}/{s['n_interim_control']} control vs "
            f"{i['xt']}/{s['n_interim_treatment']} treatment: "
            f"P(p_T > p_C | data) = {i['posterior_prob_superiority']:.4f}, "
            f"PPoS = {i['ppos']:.4f}, decision: {i['decision']}"
        )


# -------------------------------------------------------------------------- commands


@app.command()
def design(
    config_file: ConfigOpt = None,
    endpoint: EndpointOpt = None,
    delta: DeltaOpt = None,
    sd: SdOpt = None,
    pc: PcOpt = None,
    pt: PtOpt = None,
    hr: HrOpt = None,
    delta0: Delta0Opt = None,
    hr0: Hr0Opt = None,
    prob_event: ProbEventOpt = None,
    median_control: MedianControlOpt = None,
    accrual_time: AccrualTimeOpt = None,
    accrual_rate: AccrualRateOpt = None,
    followup: FollowupOpt = None,
    dropout: DropoutOpt = None,
    dropout_time: DropoutTimeOpt = None,
    alpha_one_sided: AlphaOneOpt = None,
    alpha_two_sided: AlphaTwoOpt = None,
    power: PowerOpt = None,
    allocation: AllocationOpt = None,
    method: MethodOpt = None,
    looks: LooksOpt = None,
    efficacy: EfficacyOpt = None,
    futility: FutilityOpt = None,
    binding_futility: BindingOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Sample size and boundaries: a fixed design, or group-sequential with --looks.

    A fixed design reports a two-sided alpha (default 0.05). A group-sequential design uses
    the one-sided alpha (default 0.025) and inflates the fixed design with the same critical
    value.
    """
    flags = {k: v for k, v in locals().items() if k not in ("config_file", "as_json")}
    try:
        cfg = config.load_config(config_file)
        inp = _resolve_design(cfg, flags)
        if inp.looks is None:
            out = _fixed(inp, 2 * inp.alpha_one_sided)
        else:
            _, out = _sequential(inp)
    except ValueError as exc:
        raise _fail(exc) from exc
    if as_json:
        _print_json(out)
        return
    if out["design"] == "fixed":
        typer.echo(f"trialdesign {__version__} | fixed design | alpha is two-sided")
        pairs = {"hypothesis": report.hypothesis_text(out), **out}
        acc = pairs.pop("accrual", None)
        pairs.pop("accrual_inputs", None)
        _print_pairs(pairs)
        if acc is not None:
            typer.echo(f"\n{report.accrual_text(out)}")
        return
    typer.echo(f"trialdesign {__version__} | group-sequential design | alpha is one-sided")
    _print_sequential(out)


@app.command()
def simulate(
    config_file: ConfigOpt = None,
    endpoint: EndpointOpt = None,
    delta: DeltaOpt = None,
    sd: SdOpt = None,
    pc: PcOpt = None,
    pt: PtOpt = None,
    hr: HrOpt = None,
    delta0: Delta0Opt = None,
    hr0: Hr0Opt = None,
    prob_event: ProbEventOpt = None,
    median_control: MedianControlOpt = None,
    accrual_time: AccrualTimeOpt = None,
    accrual_rate: AccrualRateOpt = None,
    followup: FollowupOpt = None,
    dropout: DropoutOpt = None,
    dropout_time: DropoutTimeOpt = None,
    alpha_one_sided: AlphaOneOpt = None,
    alpha_two_sided: AlphaTwoOpt = None,
    power: PowerOpt = None,
    allocation: AllocationOpt = None,
    method: MethodOpt = None,
    looks: LooksOpt = None,
    efficacy: EfficacyOpt = None,
    futility: FutilityOpt = None,
    binding_futility: BindingOpt = None,
    nsim: Annotated[
        int | None,
        typer.Option(help="Simulated trials [default: 100000 score, 10000 patient]."),
    ] = None,
    seed: SeedOpt = None,
    engine: EngineOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Monte Carlo operating characteristics of a group-sequential design under H0 and H1.

    Type I error is simulated with futility ignored (non-binding) or followed (binding); power
    and expected sample size follow futility. Every estimate has a Monte Carlo SE and is shown
    next to its exact value. --engine patient simulates patients and the real test (t-test,
    Farrington-Manning z-test or log-rank) instead of the canonical z-scale process.
    """
    flags = {k: v for k, v in locals().items() if k not in ("config_file", "as_json")}
    try:
        cfg = config.load_config(config_file)
        inp = _resolve_design(cfg, flags)
        if inp.looks is None:
            raise ValueError("simulate needs a group-sequential design: give --looks")
        eng = _engine(cfg, flags)
        n_sim, seed_used = _simulation_settings(cfg, flags, 100_000 if eng == "score" else 10_000)
        gsd, s = _sequential(inp)
        if eng == "score":
            design_sim = simulate_design(gsd, n_sim, seed_used)
        else:
            design_sim = _patient_simulation(inp, gsd, s, n_sim, seed_used)
        sim = {**report.simulation_summary(design_sim, _n_max(s)), "engine": eng}
    except ValueError as exc:
        raise _fail(exc) from exc
    if as_json:
        _print_json({"design": s, "simulation": sim})
        return
    typer.echo(
        f"trialdesign {__version__} | seed {seed_used} | {n_sim:,} simulated trials | "
        f"{eng} engine | alpha is one-sided"
    )
    _print_sequential(s, sim)


@app.command(name="bayes")
def bayes_cmd(
    config_file: ConfigOpt = None,
    pc: PcOpt = None,
    pt: PtOpt = None,
    n: NOpt = None,
    interim: InterimOpt = None,
    allocation: AllocationOpt = None,
    prior_c: PriorCOpt = None,
    prior_t: PriorTOpt = None,
    success: SuccessOpt = None,
    calibrate: CalibrateOpt = None,
    alpha_one_sided: AlphaOneOpt = None,
    go: GoOpt = None,
    nogo: NogoOpt = None,
    design_prior_c: DesignPriorCOpt = None,
    design_prior_t: DesignPriorTOpt = None,
    xc: XcOpt = None,
    xt: XtOpt = None,
    nsim: NsimOpt = None,
    seed: SeedOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Bayesian go/no-go interim for a binary endpoint: exact OCs, PPoS and assurance.

    Give --xc and --xt for the posterior, PPoS and decision at an observed interim, a design
    prior for assurance, and --nsim to check the exact results by simulation.
    """
    flags = {k: v for k, v in locals().items() if k not in ("config_file", "as_json")}
    try:
        cfg = config.load_config(config_file)
        bd, extra = _resolve_bayes(cfg, flags)
        n_sim, seed_used = _simulation_settings(cfg, flags, None)
        s, sim = _bayes_run(bd, extra, n_sim, seed_used)
    except ValueError as exc:
        raise _fail(exc) from exc
    if as_json:
        _print_json({"bayes": s, "simulation": sim})
        return
    seed_text = f"seed {seed_used}" if sim is not None else "exact (no simulation)"
    typer.echo(f"trialdesign {__version__} | {seed_text} | Bayesian interim, binary endpoint")
    _print_bayes(s, sim)


@app.command(name="ssr")
def ssr_cmd(
    config_file: ConfigOpt = None,
    endpoint: EndpointOpt = None,
    delta: DeltaOpt = None,
    sd: SdOpt = None,
    pc: PcOpt = None,
    pt: PtOpt = None,
    hr: HrOpt = None,
    delta0: Delta0Opt = None,
    median_control: MedianControlOpt = None,
    accrual_time: AccrualTimeOpt = None,
    accrual_rate: AccrualRateOpt = None,
    followup: FollowupOpt = None,
    dropout: DropoutOpt = None,
    dropout_time: DropoutTimeOpt = None,
    alpha_one_sided: AlphaOneOpt = None,
    alpha_two_sided: AlphaTwoOpt = None,
    power: PowerOpt = None,
    allocation: AllocationOpt = None,
    method: MethodOpt = None,
    efficacy: EfficacyOpt = None,
    futility: FutilityOpt = None,
    interim_fraction: InterimFractionOpt = None,
    cp_min: CpMinOpt = None,
    cp_target: CpTargetOpt = None,
    max_factor: MaxFactorOpt = None,
    extend: ExtendOpt = None,
    z1: Z1Opt = None,
    nsim: NsimOpt = None,
    seed: SeedOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Promising-zone sample size re-estimation at one interim (survival: events).

    Plans a two-look group-sequential design (looks at --interim-fraction and 1), then, in
    the promising zone of conditional power at the observed trend, raises the second-stage
    sample size to reach --cp-target, up to --max-factor times the planned maximum. The final
    test is the Cui-Hung-Wang weighted inverse-normal test, so alpha is controlled exactly.
    For survival the number of events is re-estimated, which needs the accrual model
    (--median-control and two of --accrual-time, --accrual-rate, --followup); --extend says
    how the extra events are found. Prints the zones and exact operating characteristics;
    --z1 applies the rule to an observed interim result, and --nsim adds simulated checks
    (for survival, patient-level).
    """
    flags = {k: v for k, v in locals().items() if k not in ("config_file", "as_json")}
    try:
        cfg = config.load_config(config_file)
        adapt = _resolve_adaptation(cfg, flags, required=True)
        assert adapt is not None
        n_sim, seed_used = _simulation_settings(cfg, flags, None)
        _, s = _ssr_run(cfg, flags, adapt, n_sim, seed_used)
    except ValueError as exc:
        raise _fail(exc) from exc
    if as_json:
        _print_json(s)
        return
    seed_text = f" | seed {seed_used}" if seed_used is not None else ""
    what = "event" if s.get("unit") == "events" else "sample size"
    typer.echo(f"trialdesign {__version__}{seed_text} | {what} re-estimation | alpha is one-sided")
    _print_ssr(s)


@app.command(name="report")
def report_cmd(
    config_file: ConfigOpt = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Folder for report.md and the PNG plots. Without it, print Markdown."),
    ] = None,
    endpoint: EndpointOpt = None,
    delta: DeltaOpt = None,
    sd: SdOpt = None,
    pc: PcOpt = None,
    pt: PtOpt = None,
    hr: HrOpt = None,
    delta0: Delta0Opt = None,
    hr0: Hr0Opt = None,
    prob_event: ProbEventOpt = None,
    median_control: MedianControlOpt = None,
    accrual_time: AccrualTimeOpt = None,
    accrual_rate: AccrualRateOpt = None,
    followup: FollowupOpt = None,
    dropout: DropoutOpt = None,
    dropout_time: DropoutTimeOpt = None,
    alpha_one_sided: AlphaOneOpt = None,
    alpha_two_sided: AlphaTwoOpt = None,
    power: PowerOpt = None,
    allocation: AllocationOpt = None,
    method: MethodOpt = None,
    looks: LooksOpt = None,
    efficacy: EfficacyOpt = None,
    futility: FutilityOpt = None,
    binding_futility: BindingOpt = None,
    n: NOpt = None,
    interim: InterimOpt = None,
    prior_c: PriorCOpt = None,
    prior_t: PriorTOpt = None,
    success: SuccessOpt = None,
    calibrate: CalibrateOpt = None,
    go: GoOpt = None,
    nogo: NogoOpt = None,
    design_prior_c: DesignPriorCOpt = None,
    design_prior_t: DesignPriorTOpt = None,
    xc: XcOpt = None,
    xt: XtOpt = None,
    interim_fraction: InterimFractionOpt = None,
    cp_min: CpMinOpt = None,
    cp_target: CpTargetOpt = None,
    max_factor: MaxFactorOpt = None,
    extend: ExtendOpt = None,
    z1: Z1Opt = None,
    nsim: NsimOpt = None,
    seed: SeedOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Markdown design report with PNG plots of boundaries and operating characteristics.

    Reports the group-sequential design ([design] and [sequential], or --endpoint and
    --looks) and the Bayesian interim ([bayes], or --n and --interim), whichever are given.
    Simulated checks are added when --nsim or [simulation] is set. Nothing is written to
    disk without --out.
    """
    flags = {k: v for k, v in locals().items() if k not in ("config_file", "as_json", "out")}
    try:
        cfg = config.load_config(config_file)
        want_gsd = "design" in cfg or endpoint is not None
        want_bayes = "bayes" in cfg or n is not None or interim is not None
        if not (want_gsd or want_bayes):
            raise ValueError(
                "nothing to report: give --config, --endpoint and --looks, or --n and --interim"
            )
        n_sim, seed_used = _simulation_settings(cfg, flags, None)
        gsd_part = bayes_part = ssr_part = None
        adapt = _resolve_adaptation(cfg, flags, required=False) if want_gsd else None
        if adapt is not None:
            ssr_d, ssr_s = _ssr_run(cfg, flags, adapt, n_sim, seed_used)
            ssr_part = (ssr_d, ssr_s)
        if want_gsd:
            inp = _resolve_design(cfg, flags if adapt is None else _ssr_flags(cfg, flags, adapt))
            if inp.looks is None:
                raise ValueError("the report needs a group-sequential design: give --looks")
            gsd, s = _sequential(inp)
            sim = None
            if n_sim is not None:
                sim = report.simulation_summary(simulate_design(gsd, n_sim, seed_used), _n_max(s))
            gsd_part = (gsd, s, sim)
        if want_bayes:
            bd, extra = _resolve_bayes(cfg, flags)
            bayes_part = (bd, *_bayes_run(bd, extra, n_sim, seed_used))
        source = None if config_file is None else str(config_file)
        files: list[Path] = []
        if out is not None:
            files = report.write_report(
                out,
                __version__,
                source=source,
                gsd=gsd_part,
                bayes_design=bayes_part,
                ssr_design=ssr_part,
            )
    except ValueError as exc:
        raise _fail(exc) from exc
    if as_json:
        payload: dict[str, Any] = {"files": [str(f) for f in files], "seed": seed_used}
        if gsd_part is not None:
            payload["design"], payload["simulation"] = gsd_part[1], gsd_part[2]
        if bayes_part is not None:
            payload["bayes"], payload["bayes_simulation"] = bayes_part[1], bayes_part[2]
        if ssr_part is not None:
            payload["ssr"] = ssr_part[1]
        _print_json(payload)
        return
    if out is not None:
        seed_text = f" | seed {seed_used}" if seed_used is not None else ""
        typer.echo(f"trialdesign {__version__}{seed_text} | report written:")
        for f in files:
            typer.echo(f"  {f}")
        return
    seed_text = f", seed {seed_used}" if seed_used is not None else ""
    typer.echo(f"<!-- trialdesign {__version__}{seed_text}; add --out DIR for the plots -->\n")
    if gsd_part is not None:
        typer.echo(report.gsd_markdown(gsd_part[1], gsd_part[2], plots=False) + "\n")
    if ssr_part is not None:
        typer.echo(report.ssr_markdown(ssr_part[1], plots=False) + "\n")
    if bayes_part is not None:
        typer.echo(report.bayes_markdown(bayes_part[1], bayes_part[2], plots=False))
