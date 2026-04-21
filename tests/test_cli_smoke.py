"""CLI smoke tests that do not require a real GPU."""
from __future__ import annotations

import subprocess


def test_cli_help() -> None:
    r = subprocess.run(["rock-snitch", "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for cmd in ("index", "detect", "pseudolabel", "train", "eval", "viz"):
        assert cmd in r.stdout


def test_cli_version() -> None:
    r = subprocess.run(["rock-snitch", "--version"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "0.1" in r.stdout


def test_cli_subcommand_helps() -> None:
    for cmd in ["index", "detect", "pseudolabel", "train", "eval", "viz"]:
        r = subprocess.run(["rock-snitch", cmd, "--help"], capture_output=True, text=True)
        assert r.returncode == 0, f"{cmd} --help failed: {r.stderr}"
