#! ./.venv/bin/python

import os
from typing import Optional
from enum import Enum


try:
    import typer
except ImportError as e:
    os.system("pip install typer")
    import typer


app = typer.Typer()


class Environment(str, Enum):
    dev = "dev"
    test = "test"
    staging = "staging"
    production = "production"


@app.command()
def install():
    typer.echo("\nInstalling packages 🚀")
    os.system("pip install -r src/requirements.txt")
    typer.echo(f"\nPackages installed. Have fun 😁 \n")


@app.command("test")
def test():
    os.system(f"ENV=test pytest -s")


@app.command("coverage")
def coverage():
    os.system(f"ENV=test pytest -s --cov=timepulse")
    # os.system(f"coverage report")


@app.command("seed")
def seed(env: Optional[Environment] = Environment.dev):
    typer.echo(f"\nRunning Seed | Environment: {env} 🚀 \n")
    os.system(f"ENV=dev python3 seed.py")


if __name__ == "__main__":
    app()
