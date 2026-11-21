"""Command-line interface for trialdesign."""

from typing import Annotated

import typer

from trialdesign import __version__

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
