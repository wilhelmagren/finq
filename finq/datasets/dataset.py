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
Last updated: 2023-10-14
"""

from __future__ import annotations

import logging
import json
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

from finq.datautil import CachedRateLimiter
from tqdm import tqdm
from pyrate_limiter import (
    Duration,
    RequestRate,
    Limiter,
)

from collections import OrderedDict
from pathlib import Path
from typing import (
    Literal,
    Optional,
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
                "Number of names does not match the list of symbols, "
                f"{len(names)} != {len(symbols)}. {names=}\n{symbols=}"
            )

        self._names = names
        self._symbols = symbols

        if save:
            save_path = Path(save_path)
            log.debug(f"will save fetched data to path `{save_path}`")

        self._save = save
        self._save_path = save_path

        # TODO: implement more missing values strategies(?)
        log.debug(f"will handle missing values according to strategy `{missing_values}`")
        self._missing_values = missing_values

    @staticmethod
    def _validate_save_path(path: Path) -> Union[Exception, NoReturn]:
        """ """

        log.debug(f"validating `{path}` path...")
        if path.exists():
            if not path.is_dir():
                raise ValueError(
                    "Your specified path to save the fetched data to is not a "
                    "directory, maybe you provided a path to a file you want to create?"
                )
            log.warn(
                f"path `{path}` already exists, will overwrite newly fetched data..."
            )
        else:
            log.debug("trying to create your specified save path...")
            path.mkdir(parents=True, exist_ok=False)
            log.debug("OK!")

            data_path = path / "data"
            info_path = path / "info"

            log.debug(f"creating path `{data_path}`...")
            data_path.mkdir(parents=False, exist_ok=False)
            log.debug("OK!")

            log.debug(f"creating path `{info_path}`...")
            info_path.mkdir(parents=False, exist_ok=False)
            log.debug("OK!")

    def load_local_data(
        self,
        *,
        separator: Literal[",", ".", ";", ":"] = ";",
    ) -> Union[FileNotFoundError, NoReturn]:
        """ """

        path = Path(self._save_path)

        if not path.is_dir():
            raise FileNotFoundError(
                "",
            )

        info_path = path / "info"
        data_path = path / "data"

        if not info_path.is_dir():
            raise FileNotFoundError(
                f"The path `{info_path}` is not an existing directory."
                f"Maybe you have not yet fetched any data?"
            )

        if not data_path.is_dir():
            raise FileNotFoundError(
                f"The path `{data_path}` is not an existing directory."
                f"Maybe you have not yet fetched any data?"
            )

        info = {}
        data = {}
        dates = OrderedDict()

        for symbol in (bar := tqdm(self._symbols)):
            bar.set_description(f"Loading ticker `{symbol}` from local path `{path}`")

            try:
                with open(info_path / f"{symbol}.json", "r") as f:
                    symbol_info = json.load(f)
                    info[symbol] = symbol_info
            except:
                raise FileNotFoundError(
                    f"Could not load `{symbol}` from local path `{info_path}`. "
                    "Perhaps you want have not yet fetched the data?"
                )

            try:
                symbol_data = pd.read_csv(
                    data_path / f"{symbol}.csv",
                    sep=separator,
                )

                data[symbol] = symbol_data
            except:
                raise FileNotFoundError(
                    f"Could not load `{symbol}` from local path `{data_path}`. "
                    "Perhaps you want have not yet fetched the data?"
                )

            for date in data[symbol].index:
                dates[date] = None

        self._info = info
        self._data = data
        self._dates = list(dates.keys())

    def fetch_data(
        self,
        period: str,
        *,
        n_requests: int = 2,
        interval: int = 1,
        separator: Literal[",", ".", ";", ":"] = ";",
    ) -> Dataset:
        """ """

        path = Path(self._save_path)
        info_path = path / "info"
        data_path = path / "data"

        if info_path.is_dir() and data_path.is_dir():
            log.info(
                f"found local files for `{self.__class__.__name__}`, attempting load..."
            )
            self.load_local_data()
            log.info("OK!")
            return self

        # We combine a requests_cache with rate-limiting to avoid triggering
        # Yahoo's rate-limiter that can otherwise corrupt data. We specify
        # a maximum number of requests N per X seconds.
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
            bar.set_description(f"Fetching ticker `{symbol}` data from Yahoo! Finance")
            ticker = yf.Ticker(symbol, session=session)
            info[symbol] = ticker.info
            data[symbol] = pd.DataFrame(
                ticker.history(period=period)[
                    [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                    ]
                ]
            )
            for date in data[symbol].index:
                dates[date] = None

        if self._save:
            self._validate_save_path(self._save_path)
            log.info(f"saving fetched data to `{self._save_path}`...")
            for symbol in self._symbols:
                data[symbol].to_csv(
                    self._save_path / "data" / f"{symbol}.csv",
                    sep=separator,
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

        log.debug(f"using missing values strategy: `{self._missing_values}`")
        for symbol in self._symbols:
            dates = self._data[symbol].index
            ticker_dates[symbol] = set(dates)

        missed_data = []
        for symbol in (bar := tqdm(self._symbols)):
            bar.set_description(f"Fixing ticker `{symbol}` potential missing values")

            diff_dates = set(self._dates) - ticker_dates[symbol]
            df = self._data[symbol]

            if diff_dates:
                missed_data.append(symbol)

                _nan_array = np.full((len(diff_dates), len(df.columns)), np.nan)
                _df_to_append = pd.DataFrame(
                    _nan_array,
                    columns=df.columns,
                    index=list(diff_dates),
                )

                df = (
                    pd.concat(
                        [
                            df,
                            _df_to_append,
                        ]
                    )
                    .sort_index()
                    .interpolate()
                )

                self._data[symbol] = df

        if missed_data:
            log.info(
                f"the following symbols had missing data: `{','.join(missed_data)}`"
            )
        log.info("OK!")
        return self

    def verify_data(self) -> Union[Exception, Dataset]:
        """ """

        log.info("verifying that stored data has no missing values...")
        for symbol in (bar := tqdm(self._symbols)):
            bar.set_description(f"Verifying ticker `{symbol}` data")
            dates = set(self._data[symbol].index)
            diff = dates - set(self._dates)
            if diff:
                raise ValueError(
                    f"There is a difference in dates for symbol `{symbol}`, have you "
                    "tried fixing missing values prior to verifying? To do that, run "
                    "dataset.fix_missing_data() with your initialized dataset class."
                )
        log.info("OK!")
        return self

    def run(self, period: str) -> Union[Exception, NoReturn]:
        """ """
        self.fetch_data(period).fix_missing_data().verify_data()

    def visualize(
        self,
        *,
        title: str = "Historical stock data",
        xlabel: Optional[str] = None,
        ylabel: str = "Closing price [$]",
        ticks_rotation: int = 80,
        legend_loc: str = "best",
        log_scale: bool = False,
        save_path: Optional[str] = None,
        series: Literal[
            "Open",
            "High",
            "Low",
            "Close",
        ] = "Close",
        block: bool = False,
        pause: int = 0,
    ):
        """ """

        for symbol, data in self._data.items():
            plt.plot(
                np.log(data[series]) if log_scale else data[series],
                label=symbol,
            )

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel + " (log scale)" if log_scale else "")
        plt.xticks(rotation=ticks_rotation)
        plt.legend(loc=legend_loc)

        if save_path:
            log.debug(f"saving plot to path `{save_path}`")
            plt.savefig(save_path)
            log.debug("OK!")

        plt.show(block=block)
        plt.pause(pause)
        plt.close()

    def get_tickers(self) -> List[str]:
        """ """
        return self._symbols

    def get_data(self) -> Dict[str, pd.Series]:
        """ """
        return self._data

    def as_numpy(
        self,
        series: Literal[
            "Open",
            "High",
            "Low",
            "Close",
        ] = "Close",
    ) -> np.ndarray:
        """ """
        return np.array([d[series] for d in self._data.values()]).astype(np.float32)
