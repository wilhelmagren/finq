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
Last updated: 2023-10-16
"""

from finq.datasets.dataset import Dataset

from pathlib import Path
from typing import (
    List,
    Dict,
    Union,
    Optional,
)


class CustomDataset(Dataset):
    """ """

    def __init__(
        self,
        names: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        *,
        index_name: Optional[str] = None,
        market: Optional[str] = None,
        save_path: Union[str, Path] = ".data/CUSTOM/",
        **kwargs: Dict,
    ):
        """ """

        if all(map(lambda x: x is None, (names, symbols, index_name))):
            raise ValueError(
                "All values can't be None. You need to either specify "
                "`index_name` or `names` and `symbols`. If you specify an "
                "unknown index name, we will try and find it on NASDAQ, but might fail."
            )

        if index_name:
            if not market:
                raise ValueError(
                    "When defining a `CustomDataset` you need to specify what market "
                    "that you intend to fetch the ticker symbols from, e.g., `OMX`."
                )
            save_path = f".data/{index_name}/"

        super(CustomDataset, self).__init__(
            names,
            symbols,
            index_name=index_name,
            market=market,
            save_path=save_path,
            **kwargs,
        )
