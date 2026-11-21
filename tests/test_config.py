from pathlib import Path

import pytest

from trialdesign import config

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize("name", ["phase3_survival.toml", "phase2_binary_bayes.toml"])
def test_example_configs_load(name: str) -> None:
    cfg = config.load_config(EXAMPLES / name)
    assert set(cfg) <= set(config.SECTIONS)
    assert cfg["simulation"] == {"nsim": 200_000, "seed": 2026}


def test_no_config_is_empty() -> None:
    assert config.load_config(None) == {}


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError, match="config file not found"):
        config.load_config(tmp_path / "nope.toml")


def test_invalid_toml(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("[design\nhr = 0.7\n")
    with pytest.raises(config.ConfigError, match="not valid TOML"):
        config.load_config(p)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[desing]\nhr = 0.7\n", r"unknown table \[desing\]"),
        ("[design]\nhazard = 0.7\n", "unknown key 'hazard' in \\[design\\]"),
        ("[design]\nalpha = 0.025\n", "alpha is ambiguous; use alpha_one_sided or alpha_two_sided"),
        ("design = 1\n", r"\[design\] must be a table"),
    ],
)
def test_unknown_or_ambiguous_keys_fail_early(tmp_path: Path, text: str, message: str) -> None:
    p = tmp_path / "design.toml"
    p.write_text(text)
    with pytest.raises(config.ConfigError, match=message):
        config.load_config(p)


def test_flags_override_file_and_none_flags_do_not() -> None:
    cfg = {"design": {"endpoint": "survival", "hr": 0.7, "power": 0.8}}
    out = config.merge(cfg, "design", {"hr": 0.8, "power": None, "nsim": 10})
    assert out == {"endpoint": "survival", "hr": 0.8, "power": 0.8}


def test_merge_of_missing_section_uses_flags_only() -> None:
    assert config.merge({}, "sequential", {"looks": "3", "hr": 0.7}) == {"looks": "3"}
