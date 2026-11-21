from typer.testing import CliRunner

from trialdesign import __version__
from trialdesign.cli import app

runner = CliRunner()


def test_version_flag_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"trialdesign {__version__}"
