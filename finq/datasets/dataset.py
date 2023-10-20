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
Last updated: 2023-10-21
"""

from __future__ import annotations

import logging
import json
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

from finq.asset import Asset
from finq.datautil import (
    CachedRateLimiter,
    _fetch_names_and_symbols,
)
from tqdm import tqdm
from pyrate_limiter import (
    Duration,
    RequestRate,
    Limiter,
)

from pathlib import Path
from typing import (
    Optional,
    Callable,
    Dict,
    List,
    Union,
    Tuple,
)

log = logging.getLogger(__name__)


class Dataset(object):
    """ """

    def __init__(
        self,
        names: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        *,
        market: str = "OMX",
        index_name: Optional[str] = None,
        proxy: Optional[str] = None,
        n_requests: int = 5,
        t_interval: int = 1,
        save: bool = False,
        save_path: Union[str, Path] = ".data/dataset/",
        separator: str = ";",
    ) -> Dataset:
        """ """

        log.info(
            "creating cached rate-limited session with "
            f"{n_requests} requests per {t_interval} seconds"
        )

        # We combine a cache with rate-limiting to avoid triggering
        # Yahoo! Finance's rate-limiter that can otherwise corrupt data.
        # We specify a maximum number of requests N per X seconds.
        session = CachedRateLimiter(
            limiter=Limiter(
                RequestRate(
                    n_requests,
                    Duration.SECOND * t_interval,
                ),
            ),
        )

        if proxy:
            session.proxies.update(
                {
                    "https": proxy,
                }
            )

        self._proxy = proxy
        self._session = session
        self._n_requests = n_requests
        self._t_interval = t_interval

        if (not names or not symbols) and isinstance(index_name, str):
            names, symbols = _fetch_names_and_symbols(
                index_name,
                market=market,
                session=session,
            )

        if not names or not symbols:
            raise ValueError(
                "You did not pass in a list of names and symbols, and if you "
                "passed in an index name to fetch, the request failed since "
                f"`{names=}` and `{symbols=}`. Did you pass in a valid index name?"
            )

        if not (len(names) == len(symbols)):
            raise ValueError(
                "Number of names does not match the number of ticker symbols, "
                f"{len(names)} != {len(symbols)}.\n{names=}\n{symbols=}"
            )

        self._names = names
        self._symbols = symbols
        self._market = market
        self._index_name = index_name

        self._save = save
        self._save_path = Path(save_path)
        self._separator = separator

    def __getitem__(self, key: str) -> Optional[pd.DataFrame]:
        """ """
        return self._data.get(key, None)

    @staticmethod
    def _save_info(info: dict, path: Union[Path, str]):
        """
        Save the ticker information dictionary to a local file as a ``json`` object.

        Parameters
        ----------
        info : dict
            The ticker information dictionary to save as a ``json`` file.
        path : Path | str
            The local file name to save the dictionary to.

        """
        with open(path, "w") as f:
            json.dump(info, f)

    @staticmethod
    def _save_data(data: pd.DataFrame, path: Union[Path, str], separator: str):
        """
        Save the historical price data for a ticker to a local csv file.

        Parameters
        ----------
        data : pd.DataFrame
            The ``pd.DataFrame`` to save as a csv file.
        path : Path | str
            The local file name to save the csv to.
        separator : str
            The csv separator to use when saving the data. Defaults to ``;``.

        """
        data.to_csv(path, sep=separator)

    @staticmethod
    def _load_info(path: Union[Path, str]) -> dict:
        """ """
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _load_data(path: Union[Path, str], separator: str) -> pd.DataFrame:
        """ """
        return pd.read_csv(path, sep=separator)

    @staticmethod
    def _extract_dates_from_data(data: pd.DataFrame) -> Tuple[List, Dict]:
        """ """
        dates = {}
        all_dates = []

        for ticker, df in data.items():
            dates[ticker] = df["Date"].tolist()
            all_dates.extend(dates[ticker])

        unique_dates = (
            pd.DataFrame({"Date": list(set(all_dates))})
            .sort_values(
                by="Date",
                ascending=True,
            )["Date"]
            .tolist()
        )

        return unique_dates, dates

    def _prepare_save_path(self) -> Union[Exception, None]:
        """ """

        path = Path(self._save_path)

        log.debug(f"preparing {path} path...")
        if path.exists():
            if not path.is_dir():
                raise ValueError(
                    "Your specified path to save fetched data to is not a directory, "
                    "maybe you provided a path to a file you want to create?"
                )

            log.warning(f"path {path} already exists, will overwrite existing data...")

        log.debug(f"creating {path}...")
        path.mkdir(parents=True, exist_ok=True)
        log.debug("OK!")

        info_path = path / "info"
        data_path = path / "data"
        log.debug(f"creating path {info_path}...")
        info_path.mkdir(parents=False, exist_ok=True)
        log.debug("OK!")

        log.debug(f"creating path {data_path}...")
        data_path.mkdir(parents=False, exist_ok=True)
        log.debug("OK!")

    def _all_local_files_saved(self) -> bool:
        """ """
        path = Path(self._save_path)
        info_path = path / "info"
        data_path = path / "data"

        if info_path.is_dir():
            for ticker in self._symbols:
                if not Path(info_path / f"{ticker}.json").exists():
                    return False
        else:
            return False

        if data_path.is_dir():
            for ticker in self._symbols:
                if not Path(data_path / f"{ticker}.csv").exists():
                    return False
        else:
            return False

        return True

    def _save_info_and_data(self):
        """ """

        log.info(f"saving fetched tickers to {self._save_path}...")
        for ticker in self._symbols:
            self._save_info(
                self._info[ticker], self._save_path / "info" / f"{ticker}.json"
            )
            self._save_data(
                self._data[ticker],
                self._save_path / "data" / f"{ticker}.csv",
                separator=self._separator,
            )

        log.info("OK!")

    def _fetch_tickers(
        self,
        period: str,
        cols: List[str],
    ):
        """ """

        info = {}
        data = {}

        for ticker in (bar := tqdm(self._symbols)):
            bar.set_description(f"Fetching ticker {ticker} data from Yahoo! Finance")

            fetched = yf.Ticker(ticker, session=self._session)
            info[ticker] = fetched.info

            data[ticker] = fetched.history(
                period=period,
                proxy=self._proxy,
            ).reset_index()[cols]

        all_dates, dates = self._extract_dates_from_data(data)

        self._info = info
        self._data = data
        self._dates = dates
        self._all_dates = all_dates

    def load_local_files(self) -> bool:
        """ """

        path = Path(self._save_path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"The local save path `{path}` does not exist. Perhaps you haven't"
                "tried fetching any data? To do that, run `dataset.fetch_data(...)`."
            )

        info_path = path / "info"
        data_path = path / "data"

        if not info_path.is_dir():
            raise FileNotFoundError(
                f"The local save path `{info_path}` does not exist. Perhaps you haven't"
                "tried fetching any data? To do that, run `dataset.fetch_data(...)`."
            )

        if not data_path.is_dir():
            raise FileNotFoundError(
                f"The local save path `{data_path}` does not exist. Perhaps you haven't"
                "tried fetching any data? To do that, run `dataset.fetch_data(...)`."
            )

        info = {}
        data = {}

        for ticker in (bar := tqdm(self._symbols)):
            bar.set_description(f"Loading ticker {ticker} from local path {path}")
            info[ticker] = self._load_info(info_path / f"{ticker}.json")
            data[ticker] = self._load_data(
                data_path / f"{ticker}.csv",
                separator=self._separator,
            )

        all_dates, dates = self._extract_dates_from_data(data)

        self._info = info
        self._data = data
        self._dates = dates
        self._all_dates = all_dates

    def fetch_data(
        self,
        period: str,
        *,
        cols: List[str] = ["Date", "Open", "High", "Low", "Close"],
    ) -> Dataset:
        """ """

        if self._all_local_files_saved():
            log.info(
                f"found existing local files for {self.__class__.__name__}, "
                "attempting local load..."
            )

            try:
                self.load_local_files()
                log.info("OK!")
                return self

            except FileNotFoundError:
                log.warning("failed to load local files, will attempt new fetch...")

        self._fetch_tickers(period, cols)

        if self._save:
            self._prepare_save_path()
            self._save_info_and_data()

        return self

    def fix_missing_data(
        self,
        *,
        cols: List[str] = ["Open", "High", "Low", "Close"],
        resave: bool = True,
    ) -> Dataset:
        """ """

        log.info("attempting to fix any missing data...")

        n_missing_data = 0
        for ticker in (bar := tqdm(self._symbols)):
            bar.set_description(f"Fixing ticker {ticker} potential missing values")

            df = self._data[ticker]
            diff = set(self._all_dates) - set(self._dates[ticker])

            if diff:
                n_missing_data += 1
                df_missed = pd.DataFrame({"Date": list(diff)})

                df_fixed = pd.concat((df, df_missed), axis=0)
                df_fixed = df_fixed.sort_values("Date", ascending=True).reset_index()
                df_fixed[cols] = df_fixed[cols].interpolate()

                if df_fixed[df_fixed.isnull().any(axis=1)].index.values.size:
                    log.error(
                        f"failed to interpolate missing prices for ticker {ticker}!"
                    )

                self._data[ticker] = df_fixed
                self._dates[ticker] = self._all_dates

        if n_missing_data and resave:
            log.info(f"fixed {n_missing_data} tickers with missing data")
            if self._save:
                log.info(f"saving fixed data to {self._save_path}...")
                self._save_info_and_data()

        log.info("OK!")
        return self

    def verify_data(self) -> Union[ValueError, Dataset]:
        """
        Tries to verify that the stored data does not contain any missing values.
        This is performed by comparing the dates in each ticker ``pd.DataFrame``
        with the known set of all fetched dates.

        Returns
        -------
        Dataset
            The initialized instance of ``self``.

        Raises
        ------
        ValueError
            If there exists missing values in any stored ``pd.DataFrame``.

        """

        log.info("verifying that stored data has no missing values...")
        for ticker in (bar := tqdm(self._symbols)):
            bar.set_description(f"Verifying ticker {ticker} data")

            diff = set(self._all_dates) - set(self._dates[ticker])
            if diff:
                raise ValueError(
                    f"There is a difference in dates for symbol {ticker}, have you "
                    "tried fixing missing values prior to verifying? To do that, run "
                    "dataset.fix_missing_data() with your initialized Dataset class."
                )

        log.info("OK!")
        return self

    def run(self, period: str = "1y") -> Union[FileNotFoundError, ValueError, Dataset]:
        """
        Call the three core methods for the ``Dataset`` class which fetches data,
        tries to fix missing values, and lastly verifies that there is no missing data.

        Parameters
        ----------
        period : str
            The time period to try and fetch data from. Valid values are (``1d``,
            ``5d``, ``1mo``, ``3mo``, ``6mo``, ``1y``, ``2y``, ``5y``, ``10y``,
            ``ytd``, ``max``). Defaults to ``1y``.

        Returns
        -------
        Dataset
            The intialized instance of ``self``.

        Raises
        ------
        FileNotFoundError
            If the function ``load_local_data()`` fails to find the local data
            filepaths for the initialized dataset. This can only occur if you
            call ``fetch_data(period)`` with locally saved data and the paths
            are removed during the function call.

        ValueError
            If the data is not valid when calling ``verify_data()``, i.e., it contains
            missing values.

        """
        return self.fetch_data(period).fix_missing_data().verify_data()

    def visualize(
        self,
        *,
        title: str = "Historical stock data",
        xlabel: str = "Dates",
        ylabel: str = "Closing price [$]",
        ticks_rotation: int = 80,
        legend_loc: str = "best",
        log_scale: bool = False,
        save_path: Optional[str] = None,
        price_type: str = "Close",
        show: bool = True,
        block: bool = True,
        data_transform: Optional[Callable] = None,
    ):
        """
        Plot the historical ticker price data over time.

        Parameters
        ----------
        title : str
            The header title to set on the generated plot.
        xlabel : str
            The label to use for the x-axis.
        ylabel : str
            The label to use for the y-axis.
        ticks_rotation : int
            The amount of degrees to rotate the x-axis ticks with. Defaults to ``80``.
        legend_loc : str
            The location of the legend. Some possible values are (``best``, ``center``,
            ``upper left``, ``upper right``, ``lower left``, ``lower right``).
            Defaults to ``best``.
        log_scale : bool
            ``True`` if the historical data should be log scaled, otherwise ``False``.
        save_path : str | None
            The local file to save the generated plot to. Does not save the plot if
            the argument is ``None``.
        price_type : str
            The price type of the historical data to plot. Has to be one
            of (``Open``, ``High``, ``Low``, ``Close``). Defaults to ``Close``.
        show : bool
            ``True`` if the generated plot should be shown on the screen, otherwise
            ``False``. Defaults to ``True``.
        block : bool
            Whether to wait for all figures to be closed before returning. When ``False``
            the figure windows will be displayed and returned immediately. Defaults to
            ``True``.
        data_transform : Callable | None
            A function which transforms the data to be used for the plot. If parameter
            ``log_scale=True`` then it takes the value ``np.log``. Defaults to ``None``.

        """

        if data_transform is None:

            def data_transform(d):
                return d

        if log_scale:
            data_transform = np.log

        for ticker, data in self._data.items():
            plt.plot(
                data_transform(data[price_type]),
                label=ticker,
            )

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=ticks_rotation)
        plt.legend(loc=legend_loc)

        if save_path:
            log.debug(f"saving plot to path {save_path}")
            plt.savefig(save_path)
            log.debug("OK!")

        if show:
            plt.show(block=block)
            plt.close()

    def get_tickers(self) -> List[str]:
        """
        Return the saved list of ticker symbols.

        Returns
        -------
        list
            A list of ``str`` ticker symbols.

        """
        return self._symbols

    def get_data(self) -> Dict[str, pd.DataFrame]:
        """
        Return the saved dictionary which maps ticker symbols to their
        corresponding historical data with the following columns:
        (``Date``, ``Open``, ``High``, ``Low``, ``Close``).

        Returns
        -------
        dict
            A dictionary with key: ``str`` and value: ``pd.DataFrame``.

        """
        return self._data

    def as_assets(self, *, price_type: str = "Close") -> List[Asset]:
        """
        Create a list of Assets for each ticker and specified price type.

        Parameters
        ----------
        price_type : str
            The price type data to create an ``Asset`` object with. Has to be one
            of (``Open``, ``High``, ``Low``, ``Close``). Defaults to ``Close``.

        Returns
        -------
        list
            A list of newly created ``Asset`` objects.

        """
        return [
            Asset(
                self._data[ticker],
                self._names[i],
                price_type=price_type,
                market=self._market,
                index_name=self._index_name,
            )
            for i, ticker in enumerate(self._symbols)
        ]

    def as_df(self, *, price_type: str = "Close") -> pd.DataFrame:
        """
        Create an aggregated ``pd.DataFrame`` for the specified price type.
        It will have the shape (n_samples, n_tickers).

        Parameters
        ----------
        price_type : str
            The price type data to create the ``pd.DataFrame`` object with. Has to
            be one of (``Open``, ``High``, ``Low``, ``Close``). Defaults to ``Close``.

        Returns
        -------
        pd.DataFrame
            A new ``pd.DataFrame`` with ticker names as columns.

        """
        return pd.DataFrame(
            {t: d[price_type] for t, d in zip(self._symbols, self._data.values())}
        )

    def as_numpy(
        self,
        price_type: str = "Close",
        *,
        dtype: np.typing.DTypeLike = np.float32,
    ) -> np.ndarray:
        """
        Extract the specified price type from stored data as np.ndarray.
        It will have the shape (n_tickers, n_samples).

        Parameters
        ----------
        price_type : str
            The price type data to create the ``np.ndarray`` with. Has to be one
            of (``Open``, ``High``, ``Low``, ``Close``). Defaults to ``Close``.
        dtype : np.typing.DTypeLike
            The data type to create the new ``np.ndarray`` as.
            Defaults to ``np.float32``.

        Returns
        -------
        np.ndarray
            A new ``np.ndarray`` from the specified price type and dtype.

        """
        return np.array(
            [d[price_type].to_numpy().astype(dtype) for d in self._data.values()]
        )
