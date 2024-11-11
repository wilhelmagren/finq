
from pathlib import Path
from typing import Optional


def default_finq_path() -> Path:
    """Get the absolute path to the `finq` working directory.
    
    Returns
    -------
    Path
        The absolute path to the `finq` working directory.
    """
    return Path.home() / ".finq"


def setup_finq_paths() -> Optional[FileExistsError]:
    """Create the local `finq` working directory for storing data etc."""

    path = default_finq_path()
    path.mkdir(parents=True, exist_ok=False)

def _finq_working_directory_is_setup() -> bool:
    """Check whether the necessary `finq` working directory structure is already setup."""
    path = default_finq_path()
    return path.is_dir()
