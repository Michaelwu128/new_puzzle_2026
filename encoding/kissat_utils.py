"""Locate the Kissat executable from CLI input, environment, or PATH."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

KISSAT_DEFAULT = os.environ.get("KISSAT", "kissat")


def resolve_kissat(value: str | Path) -> Path:
    """Return an executable Kissat path or raise FileNotFoundError."""
    raw = os.path.expanduser(str(value))
    direct = Path(raw)
    if direct.is_file() and os.access(direct, os.X_OK):
        return direct.resolve()

    discovered = shutil.which(raw)
    if discovered:
        return Path(discovered).resolve()

    raise FileNotFoundError(
        f"Kissat executable not found: {value}. "
        "Install kissat in PATH, set KISSAT, or pass --kissat /path/to/kissat."
    )
