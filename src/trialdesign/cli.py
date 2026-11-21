"""Command-line interface for trialdesign."""

import json
from enum import StrEnum
from typing import Annotated, cast

import typer

from trialdesign import __version__, fixed

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


def _require(value: float | None, flag: str, endpoint: Endpoint) -> float:
    if value is None:
        raise ValueError(f"{flag} is required for the {endpoint.value} endpoint")
    return value


def _fixed_design(
    endpoint: Endpoint,
    delta: float | None,
    sd: float | None,
    pc: float | None,
    pt: float | None,
    hr: float | None,
    prob_event: float | None,
    alpha_two_sided: float,
    power: float,
    allocation: float,
    method: fixed.NormalMethod,
) -> dict[str, object]:
    common = {"alpha_two_sided": alpha_two_sided, "power": power, "allocation": allocation}
    if endpoint is Endpoint.normal:
        d = _require(delta, "--delta", endpoint)
        s = _require(sd, "--sd", endpoint)
        res = fixed.sample_size_normal(d, s, method=method, **common)
        inputs: dict[str, object] = {"delta": d, "sd": s, "method": method}
    elif endpoint is Endpoint.binary:
        p_c = _require(pc, "--pc", endpoint)
        p_t = _require(pt, "--pt", endpoint)
        res = fixed.sample_size_binary(p_c, p_t, **common)
        inputs = {"p_control": p_c, "p_treatment": p_t}
    else:
        h = _require(hr, "--hr", endpoint)
        ev = fixed.events_survival(h, prob_event=prob_event, **common)
        inputs = {"hazard_ratio": h, "prob_event": prob_event}
        out: dict[str, object] = {
            "endpoint": endpoint.value,
            "design": "fixed",
            **inputs,
            **common,
            "events": ev.events,
            "events_ceil": ev.events_ceil,
        }
        if ev.n_total is not None:
            out["n_total"] = ev.n_total
            out["n_total_ceil"] = ev.n_total_ceil
        return out
    return {
        "endpoint": endpoint.value,
        "design": "fixed",
        **inputs,
        **common,
        "n_control": res.n_control,
        "n_treatment": res.n_treatment,
        "n_control_ceil": res.n_control_ceil,
        "n_treatment_ceil": res.n_treatment_ceil,
        "n_total_ceil": res.n_total_ceil,
    }


@app.command()
def design(
    endpoint: Annotated[Endpoint, typer.Option(help="Primary endpoint type.")],
    delta: Annotated[
        float | None, typer.Option(help="Normal: difference in means (T - C).")
    ] = None,
    sd: Annotated[float | None, typer.Option(help="Normal: common standard deviation.")] = None,
    pc: Annotated[float | None, typer.Option(help="Binary: control response rate.")] = None,
    pt: Annotated[float | None, typer.Option(help="Binary: treatment response rate.")] = None,
    hr: Annotated[float | None, typer.Option(help="Survival: hazard ratio (T vs C).")] = None,
    prob_event: Annotated[
        float | None,
        typer.Option(help="Survival: probability of an event by the final analysis."),
    ] = None,
    alpha_two_sided: Annotated[float, typer.Option(help="Two-sided significance level.")] = 0.05,
    power: Annotated[float, typer.Option(help="Target power.")] = 0.80,
    allocation: Annotated[float, typer.Option(help="Allocation ratio n_T / n_C.")] = 1.0,
    method: Annotated[
        str, typer.Option(help='Normal only: "z" (normal approx.) or "t" (noncentral t).')
    ] = "z",
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON output.")] = False,
) -> None:
    """Fixed-design sample size (or events, for survival) for a two-arm trial."""
    if method not in ("z", "t"):
        raise typer.BadParameter('must be "z" or "t"', param_hint="--method")
    try:
        out = _fixed_design(
            endpoint,
            delta,
            sd,
            pc,
            pt,
            hr,
            prob_event,
            alpha_two_sided,
            power,
            allocation,
            cast(fixed.NormalMethod, method),
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    out["version"] = __version__
    if as_json:
        typer.echo(json.dumps(out, indent=2))
        return
    typer.echo(f"trialdesign {__version__} | fixed design | alpha is two-sided")
    width = max(len(k) for k in out)
    for key, value in out.items():
        if key == "version" or value is None:
            continue
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        typer.echo(f"  {key:<{width}}  {shown}")
