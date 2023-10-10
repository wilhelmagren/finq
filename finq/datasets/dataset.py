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

File created: 2023-10-10
Last updated: 2023-10-10
"""

from __future__ import annotations

import logging
import json
import yfinance as yf
import pandas as pd
import numpy as np

from .rate_limiter import CachedRateLimiter
from tqdm import tqdm
from pyrate_limiter import (
    Duration,
    RequestRate,
    Limiter,
)

from collections import OrderedDict
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Union,
    NoReturn,
)

log = logging.getLogger(__name__)


class Dataset(object):
    """ """

    def __init__(
        self,
        names: List[str],
        symbols: List[str],
        *,
        save: bool = False,
        save_path: Union[str, Path] = ".data/",
        missing_values: str = "interpolate",
        **kwargs: Dict[str, Any],
    ) -> Dataset:
        """ """

        if not (len(names) == len(symbols)):
            raise ValueError(
                f"Number of names does not match the list of symbols, "
                f"{len(names)} != {len(symbols)}. {names=}\n{symbols=}"
            )

        self._names = names
        self._symbols = symbols

        if save:
            save_path = Path(save_path)
            log.info(f"will save fetched data to path `{save_path}`")
            self._validate_save_path(save_path)

        self._save = save
        self._save_path = save_path

        log.info(f"will handle missing values according to strategy `{missing_values}`")
        self._missing_values = missing_values

    @staticmethod
    def _validate_save_path(path: Path) -> Union[Exception, NoReturn]:
        """ """

        log.info(f"validating `{path}` path...")
        if path.exists():
            if not path.is_dir():
                raise ValueError(
                    f"Your specified path to save the fetched data to is not a directory, "
                    f"maybe you provided a path to a file you want to create?"
                )
            log.warn(
                f"path `{path}` already exists, will overwrite any newly fetched data..."
            )
        else:
            log.info(f"trying to create your specified save path...")
            path.mkdir(parents=True, exist_ok=False)
            log.info(f"OK!")

            data_path = path / "data"
            info_path = path / "info"

            log.info(f"creating path `{data_path}`...")
            data_path.mkdir(parents=False, exist_ok=False)
            log.info("OK!")

            log.info(f"creating path `{info_path}`...")
            info_path.mkdir(parents=False, exist_ok=False)
            log.info("OK!")

    @staticmethod
    def _interpolate_values(
        v1: Union[int, float],
        v2: Union[int, float],
    ) -> np.float32:
        """ """
        return np.mean((v1, v2))

    def fetch_data(
        self,
        period: str,
        *,
        n_requests: int = 2,
        interval: int = 1,
        separator: str = ";",
    ) -> Dataset:
        """ """

        # We combine a requests_cache with rate-limiting to avoid triggering Yahoo's rate-limiter
        # that can otherwise corrupt data. We specify maximum number of requests per X seconds.
        session = CachedRateLimiter(
            limiter=Limiter(
                RequestRate(
                    n_requests,
                    Duration.SECOND * interval,
                )
            ),
        )

        info = {}
        data = {}
        dates = OrderedDict()

        for symbol in (bar := tqdm(self._symbols)):
            bar.set_description(f"Fetching `{symbol}`\t data from Yahoo! Finance")
            ticker = yf.Ticker(symbol, session=session)
            info[symbol] = ticker.info
            data[symbol] = pd.DataFrame(ticker.history(period=period)["Close"])
            for date in data[symbol].index:
                dates[date] = None

        if self._save:
            # self._save_fetched_data()
            log.info(f"saving fetched data to `{self._save_path}`")
            for symbol in self._symbols:
                data[symbol].to_csv(
                    self._save_path / "data" / f"{symbol}.csv", sep=separator
                )
                with open(self._save_path / "info" / f"{symbol}.json", "w") as f:
                    json.dump(info[symbol], f)

            log.info("OK!")

        self._info = info
        self._data = data
        self._dates = list(dates.keys())

        return self

    def fix_missing_data(self) -> Dataset:
        """ """

        ticker_dates = {}

        log.info(f"using missing values strategy: `{self._missing_values}`")
        for symbol in self._symbols:
            dates = self._data[symbol].index
            ticker_dates[symbol] = set(dates)

        missed_data = []
        for symbol in (bar := tqdm(self._symbols)):
            bar.set_description(f"Fixing `{symbol}`\t missing values")

            diff_dates = set(self._dates) - ticker_dates[symbol]
            df = self._data[symbol]

            if diff_dates:
                missed_data.append(symbol)

            for date in diff_dates:
                d_idx = self._dates.index(date)

                if d_idx == len(self._dates) - 1:
                    # we are missing the last date, so we can't interpolate
                    # with next day, use the two earlier days for reference
                    v1 = df.iloc[d_idx - 2]["Close"]
                    v2 = df.iloc[d_idx - 1]["Close"]
                    interpolated = v2 + (v2 - v1)

                elif d_idx == 0:
                    # we are missing the first date, so we can't interpolate
                    # with the earlier day, use the two next days to get a
                    # somewhat accuracate starting price
                    v1 = df.iloc[d_idx + 1]["Close"]
                    v2 = df.iloc[d_idx + 2]["Close"]
                    interpolated = v1 - (v2 - v1)

                else:
                    v1 = df.iloc[d_idx - 1]["Close"]
                    v2 = df.iloc[d_idx + 1]["Close"]
                    interpolated = self._interpolate_values(v1, v2)

                df = pd.concat(
                    (
                        df.iloc[:d_idx],
                        pd.DataFrame({"Close": interpolated}, index=[date]),
                        df.iloc[d_idx:],
                    )
                )

                df = df.sort_index()

            self._data[symbol] = df

        if missed_data:
            log.info(
                f'the following symbols had missing data: `{",".join(missed_data)}`'
            )
        log.info("OK!")
        return self

    def verify_data(self) -> Union[Exception, Dataset]:
        """ """

        log.info("verifying that stored data has no missing values...")
        for symbol in (bar := tqdm(self._symbols)):
            bar.set_description(f"Verifying `{symbol}`\t no missing values")
            dates = set(self._data[symbol].index)
            diff = dates - set(self._dates)
            if diff:
                raise ValueError(
                    f"There is a difference in dates for symbol `{symbol}`, have you "
                    f"tried fixing missing values prior to verifying? To do that, run "
                    f"dataset.fix_missing_data() with your initialized dataset class."
                )
        log.info("OK!")
        return self

    def get_tickers(self) -> List[str]:
        """ """
        return self._symbols

    def get_data(self) -> Dict[str, pd.Series]:
        """ """
        return self._data

    def as_numpy(self) -> np.ndarray:
        """ """
        return np.array([d["Close"] for d in self._data.values()]).astype(np.float32)
