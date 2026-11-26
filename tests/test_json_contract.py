"""M20: the stable ``--json`` contract.

Every case in ``tests/make_schemas.py`` must validate against its command's schema in
``docs/schema/`` (and against its own sub-schema, for a readable error). The schemas must
keep every key recorded for the current schema version in ``docs/schema/keys-v<N>.json``:
removing or renaming a key fails here until ``jsonout.SCHEMA_VERSION`` is bumped. After a
deliberate change of the output, regenerate with ``python tests/make_schemas.py``.
"""

import json
import math
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from typer.testing import CliRunner

from make_schemas import CASES, COMMANDS, SCHEMA_DIR, key_paths, run
from trialdesign import __version__, jsonout
from trialdesign.cli import app

SCHEMAS = {c: json.loads((SCHEMA_DIR / f"{c}.schema.json").read_text()) for c in COMMANDS}


@pytest.mark.parametrize(("command", "name", "args"), CASES, ids=[f"{c}-{n}" for c, n, _ in CASES])
def test_output_matches_its_schema(command, name, args):
    out = run(command, args)
    assert out["schema_version"] == jsonout.SCHEMA_VERSION
    assert out["command"] == command
    assert out["version"] == __version__
    schema = SCHEMAS[command]
    sub = next(s for s in schema["anyOf"] if s["title"] == name)
    body = {k: v for k, v in out.items() if k not in ("schema_version", "command", "version")}
    jsonschema.validate(body, sub, cls=jsonschema.Draft202012Validator)
    jsonschema.validate(out, schema, cls=jsonschema.Draft202012Validator)


def test_schemas_are_valid_json_schema():
    for schema in SCHEMAS.values():
        jsonschema.Draft202012Validator.check_schema(schema)


def test_recorded_keys_are_kept_until_the_version_is_bumped():
    path = SCHEMA_DIR / f"keys-v{jsonout.SCHEMA_VERSION}.json"
    assert path.exists(), f"bumped schema_version: run python tests/make_schemas.py ({path.name})"
    recorded = json.loads(path.read_text())
    assert set(recorded) == set(COMMANDS)
    for command, keys in recorded.items():
        have = {f"{sub['title']}: {p}" for sub in SCHEMAS[command]["anyOf"] for p in key_paths(sub)}
        lost = sorted(set(keys) - have)
        assert not lost, (
            f"{command}: keys of schema version {jsonout.SCHEMA_VERSION} removed: {lost[:10]}; "
            "bump trialdesign.jsonout.SCHEMA_VERSION"
        )


def test_every_case_has_a_sub_schema_and_back():
    for command in COMMANDS:
        titles = {s["title"] for s in SCHEMAS[command]["anyOf"]}
        assert titles == {n for c, n, _ in CASES if c == command}


def test_non_finite_numbers_become_null():
    raw = {"a": float("nan"), "b": [np.inf, -np.inf, 1.5], "c": np.float64(2.0), "d": np.int64(3)}
    text = jsonout.dumps("design", raw, "0")
    out = json.loads(text)
    assert out["a"] is None and out["b"] == [None, None, 1.5]
    assert out["c"] == 2.0 and out["d"] == 3 and isinstance(out["d"], int)
    assert "NaN" not in text and "Infinity" not in text
    assert jsonout.clean({"x": (np.array([1.0, math.nan]), np.bool_(True))}) == {
        "x": [[1.0, None], True]
    }


def test_simulated_nph_hazard_ratio_is_null_not_nan():
    # Under non-proportional hazards there is no single hazard ratio; it used to print NaN.
    case = next(a for c, n, a in CASES if n == "patient_survival_nph")
    text = CliRunner().invoke(app, ["simulate", *case, "--json"]).stdout
    assert "NaN" not in text
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard constant {c}"))


@pytest.mark.parametrize("command", ["design", "simulate", "bayes", "ssr", "report", "graph"])
def test_errors_in_json_mode_are_json_on_stderr(command, tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[nonsense]\nx = 1\n")
    res = CliRunner().invoke(app, [command, "--config", str(bad), "--json"])
    assert res.exit_code == 2
    assert res.stdout == ""
    err = json.loads(res.stderr)
    assert set(err) == {"error", "code"}
    assert err["code"] == 2 and "nonsense" in err["error"]


def test_errors_without_json_stay_plain_text():
    res = CliRunner().invoke(app, ["design", "--endpoint", "normal"])
    assert res.exit_code == 2
    assert res.stderr.startswith("Error: ")


def test_a_removed_key_fails_validation():
    case = next(a for c, n, a in CASES if n == "fixed_normal")
    out = run("design", case)
    del out["n_control"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(out, SCHEMAS["design"], cls=jsonschema.Draft202012Validator)


def test_schema_folder_holds_only_generated_files():
    # docs/schema holds only generated files: schemas and the key record.
    names = sorted(p.name for p in Path(SCHEMA_DIR).iterdir())
    assert names == sorted([*(f"{c}.schema.json" for c in COMMANDS), *_key_files()])


def _key_files():
    return [p.name for p in Path(SCHEMA_DIR).glob("keys-v*.json")]
