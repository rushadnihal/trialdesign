"""Generate the JSON Schemas of the ``--json`` output in ``docs/schema/``.

Run from the repository root after changing what a command prints:

    python tests/make_schemas.py

Each command's schema is inferred from the runs in :data:`CASES`: one sub-schema per case
(``anyOf``), with every key that the case prints marked required. ``docs/schema/keys-v<N>.json``
records the required keys of schema version N. The script refuses to drop a recorded key:
removing or renaming one needs ``jsonout.SCHEMA_VERSION`` bumped first, which starts a new
key file. ``tests/test_json_contract.py`` checks every case against these files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from trialdesign.cli import app
from trialdesign.jsonout import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs" / "schema"
EX = str(ROOT / "examples") + "/"

# (command, case name, arguments; "EX/" is the examples folder). Small simulations keep the
# tests fast; the shape of the output does not depend on the number of trials.
_CASES = [
    ("design", "fixed_normal", "--endpoint normal --delta 0.3 --sd 1"),
    (
        "design",
        "fixed_binary_noninferiority",
        "--endpoint binary --pc 0.85 --pt 0.85 --delta0 -0.1",
    ),
    (
        "design",
        "fixed_survival_accrual",
        "--endpoint survival --hr 0.7 --median-control 12 --accrual-time 24 --followup 12",
    ),
    ("design", "fixed_equivalence", "--endpoint normal --delta 0 --sd 1 --margin 0.5"),
    ("design", "group_sequential", "--config EX/phase3_survival.toml"),
    ("design", "group_sequential_accrual", "--config EX/phase3_survival_accrual.toml"),
    (
        "design",
        "group_sequential_analysis",
        "--endpoint normal --delta 0.3 --sd 1 --looks 3 --analyse 1.2,2.9",
    ),
    (
        "design",
        "group_sequential_equivalence",
        "--endpoint binary --pc 0.7 --pt 0.7 --margin 0.1 --looks 3",
    ),
    ("simulate", "score", "--config EX/phase3_survival.toml --nsim 2000 --seed 1"),
    (
        "simulate",
        "patient_normal",
        "--endpoint normal --delta 0.3 --sd 1 --looks 2 --engine patient --nsim 200 --seed 1",
    ),
    (
        "simulate",
        "patient_survival_nph",
        "--config EX/phase3_survival_nph.toml --nsim 200 --seed 1",
    ),
    ("bayes", "example", "--config EX/phase2_binary_bayes.toml --nsim 2000 --seed 1"),
    ("bayes", "interim_data", "--pc 0.3 --pt 0.45 --n 60 --interim 30 --xc 9 --xt 15"),
    ("ssr", "promising_zone", "--config EX/phase3_ssr.toml"),
    (
        "ssr",
        "promising_zone_interim",
        "--config EX/phase3_ssr.toml --z1 1.5 --nsim 2000 --seed 1",
    ),
    ("ssr", "survival", "--config EX/phase3_survival_ssr.toml --z1 1.2"),
    ("ssr", "blinded", "--config EX/phase3_blinded_ssr.toml --nsim 2000 --seed 1"),
    ("report", "group_sequential", "--config EX/phase3_survival.toml --nsim 2000 --seed 1"),
    ("report", "bayes", "--config EX/phase2_binary_bayes.toml --nsim 2000 --seed 1"),
    ("report", "ssr", "--config EX/phase3_ssr.toml --nsim 2000 --seed 1"),
    ("graph", "example", "--config EX/two_doses_graph.toml --nsim 2000 --seed 1"),
    ("graph", "test_only", "--preset holm:3 --p 0.01,0.02,0.2"),
]
CASES: list[tuple[str, str, list[str]]] = [
    (c, n, [EX + a[2:] if a.startswith("EX/") else a for a in args.split()])
    for c, n, args in _CASES
]

COMMANDS = ("design", "simulate", "bayes", "ssr", "report", "graph")
ENVELOPE = ("schema_version", "command", "version")


def run(command: str, args: list[str]) -> dict[str, Any]:
    """Run one case and return its parsed JSON."""
    res = CliRunner().invoke(app, [command, *args, "--json"])
    if res.exit_code != 0:
        raise RuntimeError(f"{command} {' '.join(args)} failed:\n{res.output}")
    return json.loads(res.stdout)


def _type(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


def infer(values: list[Any]) -> dict[str, Any]:
    """A schema that every value in ``values`` satisfies, as strict as the values allow."""
    types = {_type(v) for v in values}
    if types == {"null"}:
        return {}  # only ever null in the cases: any type
    if "number" in types:
        types.discard("integer")
        types.add("null")  # a non-finite number prints as null
    out: dict[str, Any] = {}
    if "object" in types:
        dicts = [v for v in values if isinstance(v, dict)]
        keys = sorted({k for d in dicts for k in d})
        out["properties"] = {k: infer([d[k] for d in dicts if k in d]) for k in keys}
        out["required"] = [k for k in keys if all(k in d for d in dicts)]
    if "array" in types:
        items = [x for v in values if isinstance(v, list) for x in v]
        if items:
            out["items"] = infer(items)
    order = ["object", "array", "string", "number", "integer", "boolean", "null"]
    names = [t for t in order if t in types]
    out = {"type": names[0] if len(names) == 1 else names, **out}
    return out


def key_paths(schema: dict[str, Any], prefix: str = "") -> set[str]:
    """Dotted paths of the required keys, ``[]`` marking array items."""
    paths: set[str] = set()
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        path = f"{prefix}.{key}" if prefix else key
        paths.add(path)
        paths |= key_paths(props[key], path)
    if "items" in schema:
        paths |= key_paths(schema["items"], f"{prefix}[]")
    return paths


def command_schema(command: str, variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    subs = []
    for name, output in variants.items():
        body = {k: v for k, v in output.items() if k not in ENVELOPE}
        sub = infer([body])
        sub["title"] = name
        subs.append(sub)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://github.com/rushadnihal/trialdesign/blob/main/docs/schema/{command}.schema.json",
        "title": f"trialdesign {command} --json",
        "description": (
            f"Output of `trialdesign {command} --json`, schema version {SCHEMA_VERSION}. "
            "Keys may be added within a version; removing or renaming one bumps "
            "schema_version. Non-finite numbers are null. Generated by tests/make_schemas.py."
        ),
        "type": "object",
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "command": {"const": command},
            "version": {"type": "string"},
        },
        "required": list(ENVELOPE),
        "anyOf": subs,
    }


def schema_keys(schema: dict[str, Any]) -> list[str]:
    return sorted(f"{sub['title']}: {p}" for sub in schema["anyOf"] for p in key_paths(sub))


def main() -> int:
    outputs: dict[str, dict[str, dict[str, Any]]] = {c: {} for c in COMMANDS}
    for command, name, args in CASES:
        print(f"{command} {name}", file=sys.stderr)
        outputs[command][name] = run(command, args)
    schemas = {c: command_schema(c, outputs[c]) for c in COMMANDS}
    keys = {c: schema_keys(schemas[c]) for c in COMMANDS}
    key_file = SCHEMA_DIR / f"keys-v{SCHEMA_VERSION}.json"
    if key_file.exists():
        recorded = json.loads(key_file.read_text())
        lost = {
            c: sorted(set(recorded.get(c, [])) - set(keys[c]))
            for c in recorded
            if set(recorded.get(c, [])) - set(keys[c])
        }
        if lost:
            print(
                f"Keys of schema version {SCHEMA_VERSION} were removed or renamed:\n"
                f"{json.dumps(lost, indent=2)}\n"
                "Bump trialdesign.jsonout.SCHEMA_VERSION (and note it in CHANGELOG.md), or "
                "restore the keys.",
                file=sys.stderr,
            )
            return 1
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for c in COMMANDS:
        text = json.dumps(schemas[c], indent=2) + "\n"
        (SCHEMA_DIR / f"{c}.schema.json").write_text(text)
    key_file.write_text(json.dumps(keys, indent=2) + "\n")
    print(f"wrote {len(COMMANDS)} schemas and {key_file.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
