"""Nox configuration file.

Usage:

    # Run all sessions in LOCATIONS
    nox

    # Run install, ruff and pytest sessions on LOCATIONS
    nox -s install ruff pytest

    # Run ruff on src/domain
    nox -s ruff -- src/domain

    # Run ruff and fix code in LOCATIONS
    nox -s ruff_fix

    # Run pylint on LOCATIONS
    nox -s pylint

Author: Mikel Sagardia
Date: 2024-12-04
"""
import nox

nox.options.sessions = (
    "install",
    "ruff",
    "pylint", # This is superfluous, as ruff already lints...
    "ruff_fix",
    "pytest"
)

LOCATIONS = "src", "tests", "scripts", "noxfile.py"


@nox.session(python=False) # Run in current Python env, don't create a new venv
def install(session):
    """Install dependencies."""
    # Compile requirements and sync dependencies
    session.run("pip-compile", "requirements-dev.in")
    session.run("pip-sync", "requirements-dev.txt")
    # Install the package in editable mode
    session.run("pip", "install", "-e", ".")


@nox.session(python=False)
def ruff(session):
    """Run ruff linting without fixing."""
    # posargs = ["param1", "param2"] if `ruff -s ruff -- param1 param2` else LOCATIONS
    args = session.posargs or LOCATIONS
    session.run("ruff", "check", *args)


@nox.session(python=False)
def ruff_fix(session):
    """Run ruff linting and fix code."""
    args = session.posargs or LOCATIONS
    session.run("ruff", "check", "--fix", *args)


@nox.session(python=False)
def pylint(session):
    """Run pylint."""
    args = session.posargs or LOCATIONS
    session.run("pylint", "--output-format=text", *args)


@nox.session(python=False)
def pytest(session):
    """Run pytest."""
    session.run("pytest")
