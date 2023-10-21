"""
MIT License

Copyright (c) 2023 Wilhelm Ågren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File created: 2023-10-21
Last updated: 2023-10-21
"""

import logging
from pathlib import Path
from typing import (
    List,
    Optional,
    Union,
)

log = logging.getLogger(__name__)


def default_finq_cache_path() -> Path:
    """
    Get the default absolute path to the ``finq`` http response cache.

    Returns
    -------
    Path
        The absolute path to the cache.

    """

    return Path.home() / ".finq" / "http_cache"


def default_finq_save_path() -> Path:
    """
    Get the default absolute path to the ``finq`` data directory. Default behaviour
    is a path to the home directory of the user.

    Returns
    -------
    Path
        The absolute path to the ``finq`` data directory.

    """

    return Path.home() / ".finq" / "data"


def all_tickers_saved(path: Union[Path, str], tickers: List[str]) -> bool:
    """
    Check whether or not all tickers have been saved locally.

    Parameters
    ----------
    path : Path | str
        The local path to the potentially saved data for a ``Dataset``.
    tickers : list
        The list of ticker symbols to try and find saved files for.

    Returns
    -------
    bool
        ``True`` if all ticker files exist locally, else ``False``.

    """

    if isinstance(path, str):
        path = Path(path)

    info_path = path / "info"
    data_path = path / "data"

    if info_path.is_dir():
        for ticker in tickers:
            if not Path(info_path / f"{ticker}.json").exists():
                return False

    if data_path.is_dir():
        for ticker in tickers:
            if not Path(data_path / f"{ticker}.csv").exists():
                return False

    if (not info_path.is_dir()) or (not data_path.is_dir()):
        return False

    return True


def setup_finq_save_path(path: Union[Path, str]) -> Optional[NotADirectoryError]:
    """
    Create the local paths required so save any fetched ticker info and data.

    Parameters
    ----------
    path : Path | str
        The local path of a ``Dataset`` where the required paths should be created.

    Raises
    ------
    NotADirectoryError
        If the local save path already exists but is not a directory.

    """

    if isinstance(path, str):
        path = Path(path)

    log.info(f"setting up {path} for data saving...")
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(
                "Your specified path to save fetched data to is not a directory, "
                "maybe you provided a path to a file that you want to create?"
            )

        log.warning(f"path {path} already exists, will overwrite existing data...")

    log.info(f"creating {path}...")
    path.mkdir(parents=True, exist_ok=True)
    log.info("OK!")

    info_path = path / "info"
    data_path = path / "data"

    log.info(f"creating path {info_path}...")
    info_path.mkdir(parents=False, exist_ok=True)
    log.info("OK!")

    log.info(f"creating path {data_path}...")
    data_path.mkdir(parents=False, exist_ok=True)
    log.info("OK!")
