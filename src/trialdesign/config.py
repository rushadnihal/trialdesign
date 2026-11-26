"""Design files: read a TOML design and merge it with command-line flags.

A design file has up to eight tables. Every key has a command-line flag of the same name
(underscores become dashes), and a flag given on the command line overrides the file.

.. code-block:: toml

    [design]          # endpoint, effect size and error rates
    endpoint = "survival"
    hr = 0.70
    # hr0 = 1.3       # non-inferiority: hazard ratio under H0 (delta0 for normal/binary)
    alpha_one_sided = 0.025
    power = 0.80

    [sequential]      # group-sequential looks; leave out for a fixed design
    looks = [0.33, 0.67, 1.0]
    efficacy = "obf"           # or "pocock", "hsd:-4", "kd:3", "user:0.1,0.4,1"
    futility = "pocock"        # same choices, or "none"
    binding_futility = false   # true: smaller trial, alpha holds only if futility is followed

    [accrual]         # survival only: patients and calendar time (months) from events
    median_control = 12.0
    accrual_time = 24          # or end times of segments, e.g. [6, 24]
    followup = 12              # give two of accrual_time, accrual_rate, followup
    dropout = 0.05             # share lost by dropout_time (default 12, i.e. annual)

    [simulation]
    nsim = 200000
    seed = 2026
    engine = "score"           # or "patient": simulate patients (default nsim 10000)
    test = "logrank"           # survival, patient engine: or "fh:0,1" (one look only)

    [hazards]         # survival, patient engine: simulate another event-time model
    delay = 4                  # HR 1 for 4 months, then hr_after (default: hr); or
    # waning = 12              # hr_after for 12 months, then HR 1; or
    # hazard_times = [0, 4]    # piecewise hazards per arm, with
    # hazards_control = [0.0578, 0.0578]
    # hazards_treatment = [0.0578, 0.0404]; or
    # kappa = 1.5              # Weibull events with the control median and a constant HR

    [adaptation]      # sample size (survival: event) re-estimation at one interim
    interim_fraction = 0.5
    cp_min = 0.36              # promising zone: cp_min <= CP at the observed trend < cp_target
    cp_target = 0.80
    max_factor = 2.0           # cap on total patients or events, as a multiple of the planned
                               # maximum (default 2; 1.5 for survival)
    extend = "followup"        # survival: find extra events by "followup" or "accrual"
    # blinded = true           # or: blinded re-estimation of the SD (normal endpoint) from
    # pilot_n = 60             # an internal pilot of this many patients, with
    # pilot_sd = 1.2           # the observed one-sample SD of the pooled pilot data,
    # sd_adjusted = false      # optionally bias-adjusted, and
    # floor = "planned"        # the sample size never below "planned" (or "pilot")

    [bayes]           # Bayesian go/no-go interim for a binary endpoint
    pc = 0.30
    pt = 0.45
    n = 60
    interim = 30

A ``[graph]`` table describes a graphical multiple test for ``trialdesign graph`` (keys
``preset`` or ``weights`` and ``transitions``, ``names``, ``alpha_one_sided``, ``p``,
``means``, ``correlation``); see ``examples/two_doses_graph.toml``.

Sidedness is always named: use ``alpha_one_sided`` or ``alpha_two_sided``, never ``alpha``.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["SECTIONS", "ConfigError", "load_config", "merge"]

SECTIONS: dict[str, frozenset[str]] = {
    "design": frozenset(
        {
            "endpoint",
            "delta",
            "sd",
            "pc",
            "pt",
            "hr",
            "delta0",
            "hr0",
            "margin",
            "prob_event",
            "allocation",
            "alpha_one_sided",
            "alpha_two_sided",
            "power",
            "method",
        }
    ),
    "sequential": frozenset({"looks", "efficacy", "futility", "binding_futility", "analyse"}),
    "accrual": frozenset(
        {"median_control", "accrual_time", "accrual_rate", "followup", "dropout", "dropout_time"}
    ),
    "simulation": frozenset({"nsim", "seed", "engine", "test"}),
    "graph": frozenset(
        {
            "preset",
            "names",
            "weights",
            "transitions",
            "alpha_one_sided",
            "p",
            "means",
            "correlation",
        }
    ),
    "hazards": frozenset(
        {
            "delay",
            "waning",
            "hr_after",
            "hazard_times",
            "hazards_control",
            "hazards_treatment",
            "kappa",
        }
    ),
    "adaptation": frozenset(
        {
            "interim_fraction",
            "cp_min",
            "cp_target",
            "max_factor",
            "z1",
            "extend",
            "blinded",
            "pilot_n",
            "pilot_sd",
            "sd_adjusted",
            "floor",
        }
    ),
    "bayes": frozenset(
        {
            "pc",
            "pt",
            "n",
            "interim",
            "allocation",
            "prior_c",
            "prior_t",
            "success",
            "calibrate",
            "alpha_one_sided",
            "go",
            "nogo",
            "design_prior_c",
            "design_prior_t",
            "xc",
            "xt",
        }
    ),
}

Config = dict[str, dict[str, Any]]


class ConfigError(ValueError):
    """A design file that cannot be read or has unknown tables or keys."""


def _check(config: Mapping[str, Any], source: str) -> Config:
    out: Config = {}
    for section, table in config.items():
        if section not in SECTIONS:
            known = ", ".join(f"[{s}]" for s in SECTIONS)
            raise ConfigError(f"{source}: unknown table [{section}]; expected one of {known}")
        if not isinstance(table, Mapping):
            raise ConfigError(f"{source}: [{section}] must be a table")
        for key in table:
            if key == "alpha":
                raise ConfigError(
                    f"{source}: [{section}] alpha is ambiguous; "
                    "use alpha_one_sided or alpha_two_sided"
                )
            if key not in SECTIONS[section]:
                raise ConfigError(f"{source}: unknown key {key!r} in [{section}]")
        out[section] = dict(table)
    return out


def load_config(path: str | Path | None) -> Config:
    """Read and check a TOML design file; ``None`` gives an empty design.

    Raises
    ------
    ConfigError
        If the file is missing, is not valid TOML, or has unknown tables or keys.
    """
    if path is None:
        return {}
    p = Path(path)
    try:
        with p.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {p}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p} is not valid TOML: {exc}") from None
    return _check(raw, str(p))


def merge(config: Config, section: str, flags: Mapping[str, Any]) -> dict[str, Any]:
    """Values for one table: the file's values, overridden by flags that are not None."""
    out = dict(config.get(section, {}))
    for key in SECTIONS[section]:
        if flags.get(key) is not None:
            out[key] = flags[key]
    return out
