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
Last updated: 2023-10-21
"""

from finq.exceptions import InvalidCombinationOfArgumentsError
from finq.datasets.dataset import Dataset

from typing import (
    Any,
    List,
    Dict,
    Optional,
)


class CustomDataset(Dataset):
    """
    An implementation of the base ``Dataset`` class which allows the user to define a
    custom dataset based on either: a list of names and ticker symbols, or the name of
    an index and the market it is available on.

    Parameters
    ----------
    names : list | None
        The names of the financial assets to create a dataset with.
    symbols : list | None
        The ticker symbols corresponding to the names of the financial assets.
    market : str | None
        The name of the market to fetch the historical price data from.
    index_name : str | None
        The name of the financial index to get ticker symbols and names from.
    dataset_name : str
        The name of the ``Dataset`` class instance.

    """

    def __init__(
        self,
        names: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        *,
        market: Optional[str] = None,
        index_name: Optional[str] = None,
        dataset_name: Optional[str] = "custom",
        **kwargs: Dict[str, Any],
    ) -> Optional[InvalidCombinationOfArgumentsError]:
        """ """

        if all(map(lambda x: x is None, (names, symbols, index_name))):
            raise InvalidCombinationOfArgumentsError(
                "All values can't be None. You need to either specify `index_name` or "
                "`names` and `symbols`. If you specify an unknown index name, we will "
                "try and find it on NASDAQ, but might fail."
            )

        if index_name:
            if not market:
                raise InvalidCombinationOfArgumentsError(
                    "When defining a `CustomDataset` you need to specify what market "
                    "that you intend to fetch the ticker symbols from, e.g., `OMX`."
                )
            dataset_name = index_name

        super(CustomDataset, self).__init__(
            names,
            symbols,
            market=market,
            index_name=index_name,
            dataset_name=dataset_name,
            **kwargs,
        )
