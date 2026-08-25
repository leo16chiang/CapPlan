"""Serving: what the existing Dash app and the custodian interview consume.

No new front end. A scheduled batch writes parquet; Dash reads it. The only
genuinely new artefact is the custodian pack, and that exists because the
deliverable is not a number, it is a number that survives an interview.
"""

from capplan.serve.forecast_store import publish, read_published
from capplan.serve.pack_gen import generate_pack
from capplan.serve.scenario import Scenario, apply_scenario

__all__ = ["Scenario", "apply_scenario", "generate_pack", "publish", "read_published"]
