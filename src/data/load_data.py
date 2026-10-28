"""Discover and read the two source CSVs without changing their contents."""

import hashlib
from pathlib import Path

import pandas as pd


def discover_datasets(raw_dir: Path) -> dict[str, Path]:
    """Require one UNSW-NB15 filename for each supplied split."""
    files = sorted(raw_dir.glob("*.csv"))
    result = {}
    for split, token in (("train", "training"), ("test", "testing")):
        matches = [p for p in files if "unsw" in p.stem.lower() and "nb15" in p.stem.lower().replace("-", "").replace("_", "") and token in p.stem.lower()]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {split} CSV; found {[p.name for p in matches]}")
        result[split] = matches[0]
    return result


def sha256(path: Path) -> str:
    """Hash bytes in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_datasets(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read each split separately using pandas' unchanged default inference."""
    return pd.read_csv(paths["train"], low_memory=False), pd.read_csv(paths["test"], low_memory=False)
