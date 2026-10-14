"""Shared fixtures: the workflow config, loaded exactly as Snakemake loads it."""

import sys
from pathlib import Path

import pytest
import yaml

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))


@pytest.fixture(scope="session")
def config() -> dict:
    """`config/config.yaml`."""
    return yaml.safe_load((STUDY / "config" / "config.yaml").read_text())
