"""Command-line entry point."""

import typer

app = typer.Typer(
    name="gpt2-reasoning-search",
    help="Train and serve a GPT-2-style reasoning model with search.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo("0.1.0")
