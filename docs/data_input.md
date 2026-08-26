# Getting your data in

CapPlan reads four tables. Your job is to make your Db2 warehouse produce them;
everything after that -- scoping, normalisation, anomaly labelling -- is shared
regardless of where the data came from.

```
intervals     one row per (app, LPAR, prime-time interval)   <- SMF 72 subtype 3
lpar_totals   one row per (LPAR, business day)               <- SMF 70 subtype 1
events        DR / IST / GCC SDF windows                     <- change calendar
submissions   what custodians forecast last cycle            <- capacity planning
```

`intervals` is the only one that is strictly required. The others degrade in
documented ways, and `capplan probe` tells you which capability each one costs
you.

## The 30-second version

```bash
export CAPPLAN_DB2_HOST=your_host
export CAPPLAN_DB2_PORT=50000
export CAPPLAN_DB2_DATABASE=your_db
export CAPPLAN_DB2_USER=your_user
export CAPPLAN_DB2_PASSWORD='...'          # or CAPPLAN_DB2_PASSWORD_CMD
export CAPPLAN_DB2_DRIVER='{IBM DB2 ODBC DRIVER}'

pip install -e '.[db2]'
capplan sources                             # driver present? env vars set?

# edit config/sources.yaml -- the table and column names are guesses
capplan probe --source db2 --from 2025-06-01 --to 2025-06-07

capplan ingest --source db2 --from 2022-11-01 --to 2025-10-31
capplan diagnostics
```

## What changed relative to your snippet

Your connection code is right; it is the extraction shape around it that needs
to differ.

```python
cursor.execute("SELECT * FROM schema.table FETCH FIRST 100 ROWS ONLY")
rows = cursor.fetchall()
```

Three things, in order of how much they will bite.

**`fetchall()` on the real query.** Scoped to prime time, three years of ~35
applications is roughly 950,000 rows. The raw SMF 72 table before that filter is
10-20x larger, because it holds every interval of every night and weekend.
`fetchall()` materialises all of it as a list of pyodbc `Row` objects, each with
per-row Python overhead. `capplan/data/sources/sql.py` extracts a month at a
time, `fetchmany(50_000)` within each month, and converts each batch to a
DataFrame before fetching the next.

**`SELECT *`.** Two costs. The network carries columns nobody uses, and -- worse
-- when someone renames a column the run does not fail, it produces something
subtly different. The query templates name every column and alias it to the lake
contract, and extraction raises `SchemaDriftError` if an expected alias is
missing.

**Filtering in Python.** Prime time is about 21% of the wall-clock week (9 hours
x 5 days out of 168). The predicate belongs in the `WHERE` clause so the network
carries a fifth of the rows. `capplan ingest` re-applies the filter itself, so an
over-broad query is slow but never wrong.

One thing your snippet gets right that is easy to lose: `finally: conn.close()`.
Long extracts also need to survive idle-session timeouts on the gateway, so
`SqlSource` reconnects between date chunks by default rather than holding one
session open for four hours.

## Credentials

`config/sources.yaml` is in version control. Nothing secret goes in it -- a
password in git history outlives every attempt to remove it. Credentials come
from the environment:

| Variable | Notes |
|---|---|
| `CAPPLAN_DB2_DSN` | A complete ODBC string. Overrides everything below. |
| `CAPPLAN_DB2_HOST` | required |
| `CAPPLAN_DB2_PORT` | default `50000` |
| `CAPPLAN_DB2_DATABASE` | required |
| `CAPPLAN_DB2_USER` | required |
| `CAPPLAN_DB2_PASSWORD` | or `CAPPLAN_DB2_PASSWORD_CMD` |
| `CAPPLAN_DB2_PASSWORD_CMD` | a command whose stdout is the password |
| `CAPPLAN_DB2_DRIVER` | default `{IBM DB2 ODBC DRIVER}` |

`CAPPLAN_DB2_PASSWORD_CMD` exists so a secret store can supply the password
without it ever sitting in a shell variable:

```bash
export CAPPLAN_DB2_PASSWORD_CMD='vault read -field=password secret/db2/capplan'
```

`capplan sources` prints which variables are set (never their values).

## The one thing that is not a detail

**`intervals` and `lpar_totals` must come from different SMF records.**

```
intervals    SMF 72 subtype 3   per service class / report class -> application
lpar_totals  SMF 70 subtype 1   per LPAR partition
```

It is tempting to build `lpar_totals` with a `GROUP BY business_date, lpar` over
the same workload table you used for `intervals`. Do not.

The evaluation strategy is: simulate the LPAR peak from application-level data,
then check it against what the LPAR actually did. If the "actual" is itself
computed by summing application rows, the backtest compares the simulation
against its own assumption. It will pass no matter how wrong the coincidence
model is, and that failure stays invisible until a hardware configuration has
been signed.

A useful side effect of keeping them separate: SMF 70-1 includes work that SMF
72-3 does not attribute to any of your top-N applications. `capplan diagnostics`
reports the gap as `n_unattributed_days`, and it is a real scoping finding
rather than noise.

## Mapping SMF to `app_id`

The hardest column, and it is a decision rather than a lookup. "Application" has
to come from something SMF records -- usually service class, report class, or a
site table keyed on one of them. The template joins a mapping table:

```sql
COALESCE(M.APPLICATION_ID, W.REPORT_CLASS) AS app_id
```

Two failure modes worth knowing before you see them:

- **Two applications share a service class.** CapPlan cannot tell them apart,
  the custodian of each will disagree with the number, and no modelling fixes
  it. Either split the service class or forecast them as one unit and say so.
- **The mapping changed mid-history.** A re-platformed application looks like a
  step change, and the growth estimator will happily extrapolate it for two
  fiscal years. Check `capplan diagnostics` output for applications whose fitted
  growth hits the ±clip.

If `capplan probe` reports one distinct `app_id`, the join is not joining.

## Editing the queries

`config/sources.yaml` holds one query per table. Each takes exactly two
parameter markers, in this order:

```
?   window start (inclusive)
?   window end   (exclusive)
```

CapPlan passes `datetime.date` objects and lets the driver bind them. Do not
format dates into the SQL yourself -- it is both an injection risk and a
portability problem between Db2 for z/OS and Db2 LUW.

The contract is only that the query returns the aliased names CapPlan expects:

| Table | Required | Useful if you have it |
|---|---|---|
| `intervals` | `ts`, `app_id`, `lpar`, and one of `msu` / `mips` | `environment`, `cpu_model` |
| `lpar_totals` | `business_date`, `lpar`, `peak_mips` | `mean_mips`, `peak_interval_idx` |
| `events` | `event_type`, `start_ts`, `end_ts` | `lpar`, `app_id` (`'*'` = all), `note` |
| `submissions` | `app_id`, `fiscal_year`, `submitted_peak_mips` | `submitted_on`, `basis` |

Missing optional columns are filled with explicit "unknown" markers
(`peak_interval_idx` becomes `-1`, never a real interval index) rather than
plausible-looking defaults.

If you cannot change the SQL -- a view you do not own, a DBA who will not add
aliases -- use `column_map` instead:

```yaml
column_map:
  intervals:
    smf_interval_start: ts
    report_class: app_id
```

Db2 folds unquoted identifiers to upper case, and CapPlan lower-cases every
returned column name, so write `column_map` keys in lower case.

### MSU, MIPS, or CPU seconds

Return whichever your warehouse holds. `msu` is converted using the
`mips_per_msu` and `capture_ratio` values in `config/capplan.yaml`, applied as
given and recorded on every row for audit. If you hold CPU seconds, convert in
the query using your site's own factor -- CapPlan applies what it is given and
does not second-guess it.

## `capplan probe` before `capplan ingest`

`probe` samples a few hundred rows over a short window and reports what came
back. Run it after every edit to the queries. It answers the week-1 questions
directly:

```
OK   intervals: 15-minute intervals, matching config
WARN intervals: 12 distinct app_id in the sample against a top_n_apps of 35.
FAIL lpar_totals: nothing came back. Without realised SMF 70-1 LPAR peaks
     there is no simulation backtest, and without that there is no evidence
     the coincidence model is right.
WARN submissions: none found. No per-app bias score, and no benchmark to beat.
```

The interval-length check is the one that matters most. Everything downstream
assumes 15 minutes: 36 prime-time intervals a day, and the ~950k rows that
justify a neural Stage 1 at all. If your extract is 30-minute or hourly, `probe`
says so and `calendar.interval_minutes` needs changing before anything else --
along with the argument for the neural backend.

## If someone else runs the extract

Common at sites where analysts do not get direct warehouse access. Point the
`files` source at a landing directory:

```bash
capplan ingest --source files --from 2022-11-01 --to 2025-10-31
```

```
data_drop/
  intervals_2023.parquet
  intervals_2024.parquet
  lpar_totals.csv
  events.csv
  submissions.csv
```

Column names must already match the contract (or be mapped in
`config/sources.yaml` under `files.column_map`). Parquet, CSV, gzipped CSV and
TSV all work; globs per table are fine, and a table with no matching file is
simply absent.

## What each missing table costs you

| Absent | Consequence |
|---|---|
| `intervals` | Nothing runs. |
| `lpar_totals` | No `capplan backtest`. The simulation is unvalidated -- you can produce a number, but no evidence the coincidence model is right. This is the most important dependency in the project; resolve it in week one, not week six. |
| `events` | DR/IST/GCC SDF are caught only by the robust-z spike detector, which flags candidates for a human rather than labelling them. Unlabelled events inflate every marginal and get resampled as ordinary variation. |
| `submissions` | No per-app bias score and no benchmark. The model has nothing to be better than. |

## Extraction shape, if you would rather write your own

The pattern that matters, stripped of CapPlan:

```python
import pandas as pd, pyodbc
from datetime import date, timedelta

def extract(conn_str, query, start, end, chunk_days=30, batch=50_000):
    cursor_date = start
    while cursor_date < end:
        hi = min(cursor_date + timedelta(days=chunk_days), end)
        conn = pyodbc.connect(conn_str, autocommit=True)   # reconnect per chunk
        try:
            cur = conn.cursor()
            cur.execute(query, (cursor_date, hi))          # bound, not formatted
            columns = [d[0].lower() for d in cur.description]
            while True:
                rows = cur.fetchmany(batch)                # never fetchall()
                if not rows:
                    break
                yield pd.DataFrame([tuple(r) for r in rows], columns=columns)
        finally:
            conn.close()
        cursor_date = hi
```

Two details that cost an afternoon each if missed:

- `tuple(r)` — pyodbc `Row` objects make the DataFrame constructor take a slow
  path that treats each row as a mapping.
- Strip your string columns. Fixed-width `CHAR` columns pad, and `'PAYMENTS  '`
  is not `'PAYMENTS'`. Unstripped, one application silently becomes two, each
  with half the load. `SqlSource` strips `app_id`, `lpar`, `environment` and
  `event_type` on the way in.
