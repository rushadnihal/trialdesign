"""Command-line interface for trialdesign.

Four commands, ``design``, ``simulate``, ``bayes`` and ``report``. Each reads its inputs from
flags, from a TOML design file given with ``--config`` (see :mod:`trialdesign.config`), or
both; a flag overrides the file. Results print as plain tables, or as JSON with ``--json``.
Only ``report --out DIR`` writes to disk. Every simulation prints its seed and every run
prints the package version, so a result can be reproduced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import typer

from trialdesign import __version__, bayes, config, fixed, report
from trialdesign.boundaries import GroupSequentialDesign, group_sequential_design
from trialdesign.simulate import simulate_design

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
    typer.Option("--efficacy", "--spending", help='Alpha spending: "obf" (default) or "pocock".'),
]
FutilityOpt = Annotated[
    str | None,
    typer.Option(help='Non-binding beta spending: "obf", "pocock" or "none" (default).'),
]
NsimOpt = Annotated[int | None, typer.Option(help="Number of simulated trials.")]
SeedOpt = Annotated[int | None, typer.Option(help="Random seed; drawn and printed if omitted.")]
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
    s = str(value).lower()
    if allow_none and s in ("none", ""):
        return None
    if s not in ("obf", "pocock"):
        options = '"obf", "pocock"' + (' or "none"' if allow_none else "")
        raise ValueError(f"{name} must be {options}, got {value!r}")
    return s


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
    null_key, other = ("hr0", "delta0") if endpoint == "survival" else ("delta0", "hr0")
    if d.get(other) is not None:
        raise ValueError(f"--{other} does not apply to the {endpoint} endpoint; use --{null_key}")
    if d.get(null_key) is not None:
        effect[null_key] = _number(d[null_key], null_key)
    method = str(d.get("method", "z"))
    if method not in ("z", "t"):
        raise ValueError('method must be "z" or "t"')
    seq = config.merge(cfg, "sequential", flags)
    if seq.get("binding_futility"):
        raise ValueError(
            "binding futility is not supported: futility boundaries are non-binding, so alpha "
            "is controlled whether or not they are followed"
        )
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
        futility=_spending(seq.get("futility", "none"), "futility", True),
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
        "efficacy": f"Lan-DeMets {report.SPENDING_NAMES[s['efficacy_spending']]} alpha spending",
        "futility": report.futility_text(s),
        f"fixed {unit} (alpha_two_sided {s['fixed']['alpha_two_sided']:g})": size["fixed_total"],
        "inflation_factor": s["inflation_factor"],
    }
    if unit == "events":
        pairs["max_events"] = size["max_events_ceil"]
        pairs["max_n_total"] = size.get("max_n_total_ceil")
    else:
        pairs["max_n_control"] = size["max_n_control_ceil"]
        pairs["max_n_treatment"] = size["max_n_treatment_ceil"]
        pairs["max_n_total"] = size["max_n_total_ceil"]
    _print_pairs(pairs)
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
    alpha_one_sided: AlphaOneOpt = None,
    alpha_two_sided: AlphaTwoOpt = None,
    power: PowerOpt = None,
    allocation: AllocationOpt = None,
    method: MethodOpt = None,
    looks: LooksOpt = None,
    efficacy: EfficacyOpt = None,
    futility: FutilityOpt = None,
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
        _print_pairs({"hypothesis": report.hypothesis_text(out), **out})
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
    alpha_one_sided: AlphaOneOpt = None,
    alpha_two_sided: AlphaTwoOpt = None,
    power: PowerOpt = None,
    allocation: AllocationOpt = None,
    method: MethodOpt = None,
    looks: LooksOpt = None,
    efficacy: EfficacyOpt = None,
    futility: FutilityOpt = None,
    nsim: Annotated[int | None, typer.Option(help="Simulated trials [default: 100000].")] = None,
    seed: SeedOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Monte Carlo operating characteristics of a group-sequential design under H0 and H1.

    Type I error is simulated with futility ignored (non-binding); power and expected sample
    size follow futility. Every estimate has a Monte Carlo SE and is shown next to its exact
    value.
    """
    flags = {k: v for k, v in locals().items() if k not in ("config_file", "as_json")}
    try:
        cfg = config.load_config(config_file)
        inp = _resolve_design(cfg, flags)
        if inp.looks is None:
            raise ValueError("simulate needs a group-sequential design: give --looks")
        n_sim, seed_used = _simulation_settings(cfg, flags, 100_000)
        gsd, s = _sequential(inp)
        sim = report.simulation_summary(simulate_design(gsd, n_sim, seed_used), _n_max(s))
    except ValueError as exc:
        raise _fail(exc) from exc
    if as_json:
        _print_json({"design": s, "simulation": sim})
        return
    typer.echo(
        f"trialdesign {__version__} | seed {seed_used} | {n_sim:,} simulated trials | "
        "alpha is one-sided"
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
    alpha_one_sided: AlphaOneOpt = None,
    alpha_two_sided: AlphaTwoOpt = None,
    power: PowerOpt = None,
    allocation: AllocationOpt = None,
    method: MethodOpt = None,
    looks: LooksOpt = None,
    efficacy: EfficacyOpt = None,
    futility: FutilityOpt = None,
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
        gsd_part = bayes_part = None
        if want_gsd:
            inp = _resolve_design(cfg, flags)
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
                out, __version__, source=source, gsd=gsd_part, bayes_design=bayes_part
            )
    except ValueError as exc:
        raise _fail(exc) from exc
    if as_json:
        payload: dict[str, Any] = {"files": [str(f) for f in files], "seed": seed_used}
        if gsd_part is not None:
            payload["design"], payload["simulation"] = gsd_part[1], gsd_part[2]
        if bayes_part is not None:
            payload["bayes"], payload["bayes_simulation"] = bayes_part[1], bayes_part[2]
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
    if bayes_part is not None:
        typer.echo(report.bayes_markdown(bayes_part[1], bayes_part[2], plots=False))
