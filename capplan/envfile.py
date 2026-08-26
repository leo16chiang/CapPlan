"""`.env` loading.

Exporting variables by hand is fine for one shell and bad for everything else:
it does not survive a new terminal, it does not survive a scheduled batch, and
the usual fix -- putting the exports in `.bashrc` -- puts a production password
in a file that gets backed up, synced and shoulder-surfed.

So: `.env` in the project root, git-ignored, loaded automatically.

Load order, later files winning:

    .env            the normal case, git-ignored
    .env.local      personal overrides on a shared checkout, also git-ignored

A variable already present in the real environment is never overwritten. That
ordering matters: it means a scheduled job or a container can inject
credentials the usual way, and `.env` acts as a developer convenience rather
than something that silently overrides production configuration.

No dependency. `python-dotenv` is a fine library and this is forty lines --
the parsing rules that matter (quotes, comments, `export` prefixes, blank
lines) all fit here, and one fewer wheel to get through the internal mirror is
worth something on this project specifically.
"""

from __future__ import annotations

import os
from pathlib import Path

from capplan.logging_utils import get_logger

LOG = get_logger(__name__)

DEFAULT_FILES = (".env", ".env.local")


def parse(text: str) -> dict[str, str]:
    """Parse dotenv text. Returns declaration order, last wins."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        out[key] = _value(value)
    return out


def _value(value: str) -> str:
    """Unquote and strip inline comments.

    A quoted value ends at its closing quote, not at the end of the line --
    otherwise `KEY="a b"  # note` keeps the quotes and the comment, and the
    password that reaches Db2 has a `"` on the front. An unquoted value is
    truncated at the first ` #`, which is what every other dotenv
    implementation does.
    """
    if value[:1] in ("'", '"'):
        quote = value[0]
        closing = value.find(quote, 1)
        if closing != -1:
            return value[1:closing]
        return value[1:]          # unbalanced quote: take the rest verbatim
    return value.split(" #", 1)[0].rstrip()


def load(
    files: tuple[str, ...] = DEFAULT_FILES,
    root: str | os.PathLike[str] = ".",
    override: bool = False,
) -> list[Path]:
    """Load dotenv files into `os.environ`. Returns the files actually read.

    Existing environment variables win unless `override=True`, so an operator
    can always beat a checked-out file.
    """
    loaded: list[Path] = []
    base = Path(root)
    for name in files:
        path = base / name
        if not path.exists():
            continue
        values = parse(path.read_text(encoding="utf-8"))
        applied = 0
        for key, value in values.items():
            if override or key not in os.environ:
                os.environ[key] = value
                applied += 1
        loaded.append(path)
        LOG.debug("loaded %d/%d variables from %s", applied, len(values), path)
    return loaded


def describe(files: tuple[str, ...] = DEFAULT_FILES, root: str | os.PathLike[str] = ".") -> str:
    """Which dotenv files exist, for `capplan sources`. Never prints values."""
    base = Path(root)
    present = [str(base / n) for n in files if (base / n).exists()]
    if not present:
        return (
            f"no dotenv file found (looked for {', '.join(files)}). "
            "Copy .env.example to .env and fill it in."
        )
    return "loaded: " + ", ".join(present)
