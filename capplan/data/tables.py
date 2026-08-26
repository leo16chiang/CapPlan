"""Catalogue of IZPCA / TDSz source tables: what each one measures, at what
grain, and -- the part that matters -- what it must not be used for.

IBM Z Performance and Capacity Analytics (IZPCA), previously Tivoli Decision
Support for z/OS (TDSz), stores its performance database as Db2 tables with a
suffix naming the aggregation level:

    _T   detail / timestamp        _D   daily
    _H   hourly                    _W   weekly
                                   _M   monthly

A trailing `V` is a view over the underlying table (`..._HV` is an hourly view).
Names are truncated to 18 characters, which is why `CICS_TRANSACTIO_DP` and
`MVS_ADDRDIS_ACCT_M` look clipped -- that was the Db2 for z/OS table-name limit.

Two facts drive everything below.

**The hourly tables are the finest grain available.** TDSz's own aggregation
interval is set in the `MVSPM_TIME_RES` lookup table, defaults to one hour, and
cannot be shorter than the SMF interval feeding it. So sub-hourly timing is not
in the performance database unless someone has deliberately configured it and
accepted the row-count cost.

**Most of these tables count the same CPU twice.** A CICS transaction's CPU
appears in the CICS transaction table, *and* in the CICS region's service class
in SMF 72, *and* in the region's address space in SMF 30. They are different
views of one consumption, not additive components of it. Summing MIPS across
layers is the single most expensive mistake available here, and it produces a
number that looks plausible -- roughly double -- rather than one that looks
wrong.

Hence `TableRole`. The ingest path refuses to combine two `SPINE` sources or to
treat a `DRIVER` as MIPS, because getting this wrong is not recoverable
downstream: every subsequent stage would be arithmetically correct on top of a
double-counted base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Grain(str, Enum):
    """Aggregation level. Determines what questions the table can answer."""

    INTERVAL = "interval"   # sub-hourly, if MVSPM_TIME_RES was set finer
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

    @property
    def has_intraday_timing(self) -> bool:
        """Can this table say *when within the day* something peaked?

        The coincidence factor is defined by intra-day timing. A daily table
        gives one number per application per day; even when that number is a
        daily maximum, it does not say which hour it landed in, so two daily
        maxima cannot be told apart from two simultaneous ones. Coincidence is
        not merely hard to measure from daily data -- it is absent from it.
        """
        return self in (Grain.INTERVAL, Grain.HOUR)


class TableRole(str, Enum):
    """What a table is allowed to contribute."""

    SPINE = "spine"
    """Complete, non-overlapping CPU consumption with intra-day timing.

    The forecast target. There is normally exactly one: SMF 72 workload
    activity, because WLM sees all dispatchable work and classifies it exactly
    once. Two spines would double-count."""

    DRIVER = "driver"
    """Business volume: transactions, servlet requests, job executions.

    Never summed as MIPS -- the CPU behind these is already in the spine. Two
    legitimate uses: attributing a shared service class across application
    codes, and as a leading indicator (transaction growth precedes MIPS
    growth, and a custodian can forecast transactions far more confidently
    than they can forecast MIPS)."""

    ATTRIBUTION = "attribution"
    """Address-space or account level consumption used to split the spine.

    Overlaps the spine by construction. Used for apportioning, never added."""

    VALIDATION = "validation"
    """An independent figure to check against. Never an input.

    A site's own capacity table belongs here: agreeing with it is evidence,
    and quietly using it as an input would make that agreement circular."""

    REFERENCE = "reference"
    """Calendars, mappings, submissions, LPAR totals. Not workload."""


@dataclass(frozen=True)
class TableSpec:
    """One source table."""

    name: str
    grain: Grain
    role: TableRole
    smf: str
    measures: str
    app_key: str = "app_code"
    notes: str = ""
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def can_measure_coincidence(self) -> bool:
        return self.grain.has_intraday_timing and self.role is TableRole.SPINE


# ---------------------------------------------------------------------------
# The catalogue.
#
# Grain and role are inferred from the IZPCA/TDSz naming convention and from
# which SMF record feeds each component. Confirm the columns against your own
# Db2 catalogue with `capplan probe` -- suffixes are a convention, and a site
# can rename or re-derive anything.
# ---------------------------------------------------------------------------

CATALOGUE: dict[str, TableSpec] = {}


def _add(spec: TableSpec) -> None:
    CATALOGUE[spec.name] = spec


_add(TableSpec(
    name="MVSPM_WORKLOAD2_HV",
    grain=Grain.HOUR,
    role=TableRole.SPINE,
    smf="SMF 72 subtype 3 (Workload Activity)",
    measures="CPU consumption by WLM service class / report class, hourly",
    notes=(
        "The spine, and the only table here that can carry the forecast. WLM "
        "classifies every unit of dispatchable work exactly once, so this is "
        "complete and non-overlapping -- the two properties nothing else in "
        "this list has. It is also the only source with intra-day timing, so "
        "it is the only place the coincidence factor can be measured at all."
    ),
    caveats=(
        "A view (trailing V) over MVSPM_WORKLOAD2_H. Confirm it does not "
        "already filter or pre-aggregate -- a view that quietly restricts to "
        "one sysplex is easy to miss and hard to spot downstream.",
        "Hourly means an hourly *average*. The true peak inside the peak hour "
        "is higher; see docs/peak_vs_average.md for the uplift factor and how "
        "to measure it rather than assume it.",
        "Report class granularity decides application granularity. Two "
        "applications sharing a report class cannot be separated here, and no "
        "modelling downstream can undo that.",
    ),
))

_add(TableSpec(
    name="CICS_TRANSACTIO_DP",
    grain=Grain.DAY,
    role=TableRole.DRIVER,
    smf="SMF 110 subtype 1 (CICS monitoring)",
    measures="CICS transaction counts, response and CPU time by transaction, daily",
    notes=(
        "Transaction volume per application code -- the best growth driver on "
        "this list, because a custodian can forecast transaction counts with "
        "far more confidence than MIPS, and volume growth leads MIPS growth."
    ),
    caveats=(
        "Its CPU is already inside the CICS regions' service classes in the "
        "spine. Adding it to spine MIPS double-counts.",
        "Daily grain: no intra-day timing, so it cannot contribute to the "
        "coincidence measurement.",
    ),
))

_add(TableSpec(
    name="IMS_SYSTEM_TRAN2_D",
    grain=Grain.DAY,
    role=TableRole.DRIVER,
    smf="IMS log / SMF, IMS component",
    measures="IMS transaction counts and response time by transaction, daily",
    notes="The IMS equivalent of the CICS table: volume driver and attribution key.",
    caveats=(
        "Its CPU is already in the IMS regions' service classes in the spine.",
        "Daily grain: no intra-day timing.",
    ),
))

_add(TableSpec(
    name="WAS_INT_SERVLETS_H",
    grain=Grain.HOUR,
    role=TableRole.DRIVER,
    smf="SMF 120 (WebSphere Application Server)",
    measures="Servlet request counts and response time, hourly",
    notes=(
        "Hourly, so unlike the CICS and IMS tables it *does* carry intra-day "
        "timing. That makes it useful for validating the spine's intra-day "
        "shape for WAS workloads independently -- if servlet volume peaks at "
        "10:00 and the spine says the WAS service class peaks at 14:00, one of "
        "the two attributions is wrong."
    ),
    caveats=(
        "Request counts, not MIPS. Its CPU is in the WAS service classes in "
        "the spine.",
    ),
))

_add(TableSpec(
    name="MVSPM_WORKLOAD2_HV_WEBGROUPS",
    grain=Grain.HOUR,
    role=TableRole.DRIVER,
    smf="SMF 72 / SMF 120, WAS web groups",
    measures="WebSphere web group activity, hourly",
    notes="Web-group attribution for WAS work that shares a service class.",
    caveats=("Overlaps the spine; an attribution key, not an addend.",),
))

_add(TableSpec(
    name="KPMZ_JOB_INT_D",
    grain=Grain.DAY,
    role=TableRole.DRIVER,
    smf="SMF 30 (job/step accounting), KPM component",
    measures="Batch job execution counts and elapsed/CPU time by job, daily",
    notes=(
        "Batch volume per application. Batch is the workload most likely to "
        "move wholesale -- a migrated or re-scheduled job stream is a step "
        "change, not a trend -- so this is where scenario overrides usually "
        "come from."
    ),
    caveats=(
        "Its CPU is already in the batch service classes in the spine.",
        "Daily grain: no intra-day timing.",
        "Batch frequently peaks OUTSIDE prime time. If the deliverable is an "
        "R4HA figure, restricting to prime time will miss the daily peak "
        "entirely -- see docs/peak_vs_average.md.",
    ),
))

_add(TableSpec(
    name="MVS_ADDRSPACE_D",
    grain=Grain.DAY,
    role=TableRole.ATTRIBUTION,
    smf="SMF 30 (address space accounting)",
    measures="CPU consumption by address space / job name, daily",
    notes=(
        "The bridge from a service class back to named address spaces, which "
        "is how a shared service class gets split across application codes."
    ),
    caveats=(
        "Overlaps the spine completely -- the same CPU, cut by address space "
        "instead of service class. An apportioning key, never an addend.",
        "Daily grain: gives per-application *shares*, not per-application "
        "timing.",
    ),
))

_add(TableSpec(
    name="MVS_ADDRDIS_ACCT_M",
    grain=Grain.MONTH,
    role=TableRole.ATTRIBUTION,
    smf="SMF 30 with accounting fields",
    measures="Address space CPU distribution by account code, monthly",
    notes=(
        "Account-code attribution, which at most sites is the closest thing to "
        "an authoritative application ownership record."
    ),
    caveats=(
        "Monthly grain. Twelve points a year is a mapping, not a time series: "
        "usable to decide which application owns what, useless for forecasting.",
        "Overlaps the spine.",
    ),
))

_add(TableSpec(
    name="CAP_GRP_MIPS_D",
    grain=Grain.DAY,
    role=TableRole.VALIDATION,
    smf="site-defined (custom table)",
    measures="MIPS by capacity group / application, daily",
    notes=(
        "Whatever the site already reports as application MIPS. Deliberately "
        "VALIDATION rather than an input: if CapPlan's independently derived "
        "application MIPS agree with this, that agreement is evidence, and "
        "feeding it in as an input would make the agreement circular. It is "
        "also the number custodians have already seen, so any disagreement is "
        "a conversation to have before the pack goes out, not after."
    ),
    caveats=(
        "Custom, so its derivation is a site decision that must be read before "
        "it is trusted -- in particular whether its MIPS come from the same "
        "capture ratio and MIPS-per-MSU table CapPlan is configured with.",
        "Daily grain.",
    ),
))


# ---------------------------------------------------------------------------


class DoubleCountError(ValueError):
    """Raised when a configuration would add the same CPU twice."""


def spec(name: str) -> TableSpec | None:
    """Look up a table, tolerating site prefixes and view suffixes."""
    key = name.strip().upper()
    if key in CATALOGUE:
        return CATALOGUE[key]
    bare = key.rsplit(".", 1)[-1]           # strip SCHEMA.
    if bare in CATALOGUE:
        return CATALOGUE[bare]
    for candidate, value in CATALOGUE.items():
        if bare.startswith(candidate) or candidate.startswith(bare.rstrip("V")):
            return value
    return None


def validate_roles(table_names: dict[str, str]) -> list[str]:
    """Check a {lake_table: source_table} mapping for double counting.

    Returns warnings; raises `DoubleCountError` on the one configuration that
    is unrecoverable -- two spines, or a driver used as the MIPS source.
    """
    warnings: list[str] = []
    spines: list[str] = []

    for lake_table, source_name in table_names.items():
        found = spec(source_name)
        if found is None:
            warnings.append(
                f"{source_name}: not in the catalogue. Confirm its grain and whether "
                "its CPU overlaps another source before trusting the result."
            )
            continue
        # Spine-ness is a property of the table, not of which slot it was put
        # in: a second spine listed anywhere in the configuration adds the same
        # CPU twice regardless of the key it sits under.
        if found.role is TableRole.SPINE:
            spines.append(found.name)

        if lake_table == "intervals":
            if found.role in (TableRole.DRIVER, TableRole.ATTRIBUTION):
                raise DoubleCountError(
                    f"{found.name} is a {found.role.value}, not a spine, so its CPU is "
                    f"already counted inside SMF 72 workload activity. Using it as the "
                    f"'intervals' source double-counts. {found.caveats[0] if found.caveats else ''}"
                )
            elif found.role is TableRole.VALIDATION:
                raise DoubleCountError(
                    f"{found.name} is a validation source. Feeding it in as the forecast "
                    "input makes any later agreement with it circular -- the whole point "
                    "is to derive the number independently and then compare."
                )
            if not found.grain.has_intraday_timing:
                warnings.append(
                    f"{found.name} is {found.grain.value}-grain, so it carries no "
                    "intra-day timing. The coincidence factor cannot be measured from "
                    "it, which removes the go/no-go diagnostic and the simulation "
                    "backtest."
                )

    if len(set(spines)) > 1:
        raise DoubleCountError(
            f"more than one spine configured ({', '.join(spines)}). WLM classifies each "
            "unit of work exactly once, so two spines add the same CPU twice."
        )
    if not spines:
        warnings.append(
            "no spine configured. Without a complete, non-overlapping, intra-day CPU "
            "source (normally SMF 72 workload activity) there is nothing to forecast "
            "that is not either incomplete or double-counted."
        )
    return warnings


def describe_catalogue() -> str:
    """Table for `capplan tables`."""
    lines = [
        f"{'table':<28} {'grain':<9} {'role':<12} measures",
        f"{'-' * 28} {'-' * 9} {'-' * 12} {'-' * 40}",
    ]
    for name in sorted(CATALOGUE):
        t = CATALOGUE[name]
        lines.append(f"{name:<28} {t.grain.value:<9} {t.role.value:<12} {t.measures}")
    lines.append("")
    lines.append(
        "Only a SPINE source may feed 'intervals'. DRIVER and ATTRIBUTION tables "
        "overlap it -- their CPU is already counted -- so they attribute and explain "
        "rather than add. VALIDATION is compared against, never fed in."
    )
    return "\n".join(lines)
