"""Configuration loading.

One YAML file is the single source of truth for every modelling assumption.
Anything that would need defending in a capacity review lives there, not in a
function default.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path("config/capplan.yaml")


class ConfigError(ValueError):
    """Raised when configuration is missing or internally inconsistent."""


class _Missing:
    """Sentinel distinguishing "no default given" from an explicit None."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


@dataclass(frozen=True)
class Config:
    """Immutable view over the parsed YAML with dotted-path lookup."""

    data: Mapping[str, Any]
    source: Path

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if isinstance(default, _Missing):
                    raise ConfigError(f"missing config key {dotted!r} in {self.source}")
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        node = self.get(name)
        if not isinstance(node, Mapping):
            raise ConfigError(f"config section {name!r} is not a mapping")
        return copy.deepcopy(dict(node))

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a copy with dotted-path overrides applied (used by scenarios)."""
        data = copy.deepcopy(dict(self.data))
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node = data
            for part in parts[:-1]:
                nxt = node.setdefault(part, {})
                if not isinstance(nxt, dict):
                    raise ConfigError(f"cannot override through non-mapping {dotted!r}")
                node = nxt
            node[parts[-1]] = value
        return Config(data=data, source=self.source)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.data))


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load configuration from `path`, `$CAPPLAN_CONFIG`, or the default."""
    resolved = Path(path or os.environ.get("CAPPLAN_CONFIG") or DEFAULT_CONFIG_PATH)
    if not resolved.exists():
        raise ConfigError(f"config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}
    if not isinstance(parsed, dict):
        raise ConfigError(f"config root must be a mapping, got {type(parsed).__name__}")
    cfg = Config(data=parsed, source=resolved)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Fail loudly on configurations that would silently produce nonsense."""
    minutes = cfg.get("calendar.interval_minutes")
    if not isinstance(minutes, int) or minutes <= 0 or 1440 % minutes != 0:
        raise ConfigError(f"interval_minutes={minutes!r} must divide 1440 evenly")

    start_month = cfg.get("calendar.fiscal_year_start_month")
    if not 1 <= start_month <= 12:
        raise ConfigError(f"fiscal_year_start_month={start_month} out of range")

    quantiles = list(cfg.get("model.quantiles"))
    if sorted(quantiles) != quantiles:
        raise ConfigError("model.quantiles must be sorted ascending")
    if not all(0.0 < q < 1.0 for q in quantiles):
        raise ConfigError("model.quantiles must lie strictly inside (0, 1)")
    if 0.5 not in quantiles:
        raise ConfigError("model.quantiles must include the median (0.5)")

    n_paths = cfg.get("simulation.n_paths")
    chunk = cfg.get("simulation.path_chunk")
    if chunk <= 0 or chunk > n_paths:
        raise ConfigError(f"simulation.path_chunk={chunk} must be in (0, n_paths={n_paths}]")

    dependence = cfg.get("simulation.dependence")
    if dependence not in {"block_bootstrap", "gaussian_copula"}:
        raise ConfigError(f"unknown simulation.dependence={dependence!r}")
