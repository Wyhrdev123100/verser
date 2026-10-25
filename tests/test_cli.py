import subprocess, sys, os
from typer.testing import CliRunner
from uatp.cli import app


def test_cli_hardware():
    runner = CliRunner()
    r = runner.invoke(app, ["hardware"])
    assert r.exit_code == 0
    assert "device" in r.output


def test_cli_models():
    runner = CliRunner()
    r = runner.invoke(app, ["models"])
    assert r.exit_code == 0
    assert "toy-gpt" in r.output


def test_cli_version():
    runner = CliRunner()
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0