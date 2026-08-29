"""Reading and saving the sequence tables the rest of the package works on.

Everything downstream (`judge`, the optimizer loop) expects a DataFrame with
an ``id`` column and a ``sequence`` column, so the loaders here rename the
Polish CSV headers once and nothing else has to think about it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "DATA_DIR",
    "ID_COL",
    "PROMOTERS_CSV",
    "ROOT",
    "SEQ_COL",
    "read_dataframe",
    "read_promoters",
    "save_dataframe",
    "sequence_map",
]


def _find_root() -> Path:
    """Project root: the nearest ancestor holding ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


ROOT = _find_root()
DATA_DIR = ROOT / "HackThePromotor"
PROMOTERS_CSV = DATA_DIR / "Promotory.csv"

# Column names used across the package.
ID_COL = "id"
SEQ_COL = "sequence"

# The CSV ships with Polish headers; these are the ones we care about.
RENAME = {"nazwa": ID_COL, "sekwencja": SEQ_COL}


def read_dataframe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a DataFrame, picking the reader from the file suffix.

    ``.csv`` defaults to ``sep=";"`` (what the hackathon data uses); pass
    ``sep=","`` for a comma file. ``.tsv``, ``.parquet``, ``.json`` and
    ``.jsonl`` are also understood.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        kwargs.setdefault("sep", ";")
        return pd.read_csv(path, **kwargs)
    if suffix in {".tsv", ".tab"}:
        kwargs.setdefault("sep", "\t")
        return pd.read_csv(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".jsonl":
        kwargs.setdefault("lines", True)
        return pd.read_json(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    raise ValueError(f"don't know how to read {path.name!r}")


def save_dataframe(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write a DataFrame, picking the writer from the file suffix.

    Creates missing parent directories and returns the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        kwargs.setdefault("sep", ";")
        kwargs.setdefault("index", False)
        df.to_csv(path, **kwargs)
    elif suffix in {".tsv", ".tab"}:
        kwargs.setdefault("sep", "\t")
        kwargs.setdefault("index", False)
        df.to_csv(path, **kwargs)
    elif suffix == ".parquet":
        kwargs.setdefault("index", False)
        df.to_parquet(path, **kwargs)
    elif suffix == ".jsonl":
        kwargs.setdefault("orient", "records")
        kwargs.setdefault("lines", True)
        df.to_json(path, **kwargs)
    elif suffix == ".json":
        kwargs.setdefault("orient", "records")
        df.to_json(path, **kwargs)
    else:
        raise ValueError(f"don't know how to write {path.name!r}")
    return path


def read_promoters(path: str | Path = PROMOTERS_CSV) -> pd.DataFrame:
    """The 100 natural promoters, with ``nazwa``/``sekwencja`` renamed to
    ``id``/``sequence``. Other columns (gatunek, dlugosc, N, ...) are kept."""
    df = read_dataframe(path)
    return df.rename(columns=RENAME)


def sequence_map(
    df: pd.DataFrame,
    id_col: str = ID_COL,
    seq_col: str = SEQ_COL,
) -> dict[str, str]:
    """``{id: sequence}`` -- what the judge needs to turn ids into API calls."""
    return dict(zip(df[id_col], df[seq_col]))
