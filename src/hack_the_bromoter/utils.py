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
    "convert_promoters",
    "read_dataframe",
    "save_dataframe",
    "sequence_map",
    "init_history"
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

HISTORY_DIR = ROOT / "history"
RAW_DIR = HISTORY_DIR / "raw"
FASTA_DIR = HISTORY_DIR / "fasta"


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

import csv
import os


def convert_promoters(input_path: str, output_path: str = "sequences.csv") -> int:
    """Convert the source promoter CSV into `id;gen;sekwencja` format.

    `gen` is a placeholder set to 0 for every row.
    Returns the number of rows written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, encoding="utf-8") as src, \
         open(output_path, "w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src, delimiter=";")
        writer = csv.writer(dst, delimiter=";")
        writer.writerow(["id", "sequence", "gen", "top10", "pozycja_top100", "points"])

        count = 0
        for row in reader:
            count += 1
            writer.writerow([count, row["sekwencja"], 0, 0, 0, 0])

    return count

def save_raw(df: pd.DataFrame, generation: int, directory: str | Path = RAW_DIR) -> Path:
    """Save one generation's dataframe snapshot to ``history/raw/gen_<N>.csv``.
 
    Kept separate from the rolling ``sequences.csv`` backlog so every
    generation stays inspectable on its own, even after later generations
    are appended to the backlog.
    """
    path = Path(directory) / f"gen_{generation:05d}.csv"
    return save_dataframe(df, path)
 
 
def save_fasta(fasta: str, generation: int, directory: str | Path = FASTA_DIR) -> Path:
    """Save the FASTA text submitted to /wgraj to ``history/fasta/gen_<N>.fasta``."""
    path = Path(directory) / f"gen_{generation:05d}.fasta"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fasta, encoding="utf-8")
    return path


def init_history():

    pass

def sequence_map(
    df: pd.DataFrame,
    id_col: str = ID_COL,
    seq_col: str = SEQ_COL,
) -> dict[str, str]:
    """``{id: sequence}`` -- what the judge needs to turn ids into API calls."""
    return dict(zip(df[id_col], df[seq_col]))
