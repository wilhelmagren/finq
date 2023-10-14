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

File created: 2023-10-11
Last updated: 2023-10-14
"""

import logging
from finq.datasets.dataset import Dataset
from finq.datautil import _fetch_names_and_symbols

from pathlib import Path
from typing import (
    List,
    Dict,
    Union,
    Optional,
)

log = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """ """

    def __init__(
        self,
        names: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        *,
        nasdaq_index: Optional[str] = None,
        save_path: Union[str, Path] = ".data/CUSTOM/",
        **kwargs: Dict,
    ):
        """ """

        if all(map(lambda x: x is None, (names, symbols, nasdaq_index))):
            raise ValueError("all can't be None")

        if isinstance(nasdaq_index, str):
            log.info(
                f"trying to create `{self.__class__.__name__}` from `{nasdaq_index}`..."
            )
            names, symbols = _fetch_names_and_symbols(nasdaq_index)
            save_path = f".data/{nasdaq_index}/"

        super(CustomDataset, self).__init__(
            names,
            symbols,
            save_path=save_path,
            **kwargs,
        )
