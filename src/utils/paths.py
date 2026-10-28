"""Portable discovery of the repository from a notebook or Python script."""

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Search upward for this project's layout; fail rather than guess."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if all((candidate / part).is_dir() for part in ("data/raw", "src", "notebooks")) and (candidate / "README.md").is_file():
            return candidate
    raise FileNotFoundError("Run from the project root or a directory within it.")
