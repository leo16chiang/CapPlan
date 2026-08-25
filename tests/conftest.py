"""Shared fixtures.

The synthetic panel is built once per session and shared -- it takes a couple of
seconds and every test needs it.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capplan.config import load_config
from capplan.data.calendar import grid_from_config
from capplan.data.synth import SynthSpec, generate


@pytest.fixture(scope="session")
def cfg():
    return load_config(Path(__file__).resolve().parents[1] / "config" / "capplan.yaml")


@pytest.fixture(scope="session")
def grid(cfg):
    return grid_from_config(cfg)


@pytest.fixture(scope="session")
def small_spec():
    """Small enough to be fast, large enough to keep every structural property."""
    return SynthSpec(
        n_apps=8,
        start=date(2023, 11, 1),
        end=date(2025, 10, 31),
        seed=11,
        n_dr_events=2,
        n_ist_events=2,
        n_gcc_events=1,
    )


@pytest.fixture(scope="session")
def synth(grid, small_spec):
    return generate(grid, small_spec)
