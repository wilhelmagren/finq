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
Last updated: 2023-10-22
"""

from __future__ import annotations

import logging
import json
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

from finq.exceptions import (
    DirectoryNotFoundError,
    InvalidCombinationOfArgumentsError,
)
from finq.asset import Asset
from finq.datautil import (
    CachedRateLimiter,
    all_tickers_saved,
    default_finq_cache_path,
    default_finq_save_path,
    setup_finq_save_path,
    fetch_names_and_symbols,
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
    """
    A collection of ticker symbols and their historical price data. Fetches information
    and prices from Yahoo! Finance and optionally saves them to a local path for later
    use. Supports fixing missing values by interpolating ``NaN`` and verifying the
    integrity of the fetched data.

    Parameters
    ----------
    names : list | None
        The names of the financial assets to create a dataset with.
    symbols : list | None
        The ticker symbols corresponding to the names of the financial assets.
    market : str
        The name of the market to fetch the historical price data from.
        Defaults to ``OMX``.
    index_name : str | None
        The name of the financial index to get ticker symbols and names from.
    proxy : str | None
        The name of the proxy url to use for REST requests.
    cache_name: Path | str
        The name of the path to the file which stores the cache.
        Defaults to ``/home/.finq/http_cache``.
    n_requests : int
        The max number of requests to perform per ``t_interval``. Defaults to ``5``.
    t_interval : int
        The time interval (in seconds) to use with the ``CachedRateLimiter``.
        Defaults to ``1``.
    save : bool
        Wether or not to save the fetched data to a local file path.
    save_path : Path | str
        The local file path to potentially save any fetched data to.
        Defaults to ``.data/dataset/``.
    dataset_name : str
        The name of the ``Dataset`` class instance.
    separator : str
        The csv separator to use when loading and saving any ``pd.DataFrame``.
        Defaults to ``;``.

    """

    def __init__(
        self,
        names: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        *,
        market: str = "OMX",
        index_name: Optional[str] = None,
        proxy: Optional[str] = None,
        cache_name: Union[Path, str] = default_finq_cache_path(),
        n_requests: int = 5,
        t_interval: int = 1,
        save: bool = False,
        save_path: Union[Path, str] = default_finq_save_path(),
        dataset_name: str = "dataset",
        separator: str = ";",
        filter_symbols: Callable = lambda s: s,
    ) -> Optional[InvalidCombinationOfArgumentsError]:
        """ """

        log.info(
            "creating cached rate-limited session with "
            f"{n_requests} requests per {t_interval} seconds"
        )

        # We combine a cache with rate-limiting to avoid triggering
        # Yahoo! Finance's rate-limiter that can otherwise corrupt data.
        # We specify a maximum number of requests N per X seconds.
        session = CachedRateLimiter(
            cache_name=cache_name,
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
            if market == "OMX":

                def filter_symbols(s):
                    return s.replace(" ", "-") + ".ST"

            names, symbols = fetch_names_and_symbols(
                index_name,
                market=market,
                session=session,
                filter_symbols=filter_symbols,
            )

        if not names or not symbols:
            raise InvalidCombinationOfArgumentsError(
                "You did not pass in a list of names and symbols, and if you "
                "passed in an index name to fetch, the request failed since "
                f"`{names=}` and `{symbols=}`. Did you pass in a valid index name?"
            )

        if not (len(names) == len(symbols)):
            raise InvalidCombinationOfArgumentsError(
                "Number of names does not match the number of ticker symbols, "
                f"{len(names)} != {len(symbols)}.\n{names=}\n{symbols=}"
            )

        self._names = names
        self._symbols = symbols
        self._market = market
        self._index_name = index_name

        self._save = save
        self._save_path = Path(save_path) / dataset_name
        self._dataset_name = dataset_name
        self._separator = separator

    def __getitem__(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get the ``pd.DataFrame`` from the locally stored dictionary which maps ticker
        symbols to their corresponding historical price data.

        Parameters
        ----------
        key : str
            The dictionary key to get data for.

        Returns
        -------
        pd.DataFrame
            The data that is associated with the provided ticker key.

        """
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
        """
        Parameters
        ----------
        path : Path | str
            The local file path to read the json object from.

        Returns
        -------
        dict
            A dictionary containing the information for the ticker.

        """
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _load_data(path: Union[Path, str], separator: str) -> pd.DataFrame:
        """
        Create a new ``pd.DataFrame`` from data that is stored locally as a ``csv``.

        Parameters
        ----------
        path : Path | str
            The local file path to read the csv from.
        separator : str
            The separator to use for parsing the csv.

        Returns
        -------
        pd.DataFrame
            The data that was stored in the csv.

        """
        return pd.read_csv(path, sep=separator)

    @staticmethod
    def _extract_dates_from_data(data: pd.DataFrame) -> Tuple[List, Dict]:
        """
        Extract the ``Date`` column from a ``pd.DataFrame`` and produce a sorted list of
        unique dates for the ticker.

        Parameters
        ----------
        data : pd.DataFrame
            The data to extract ``Date`` column from.

        Returns
        -------
        tuple
            A list of the unique dates (sorted in ascending order) and a dictionary
            containing all ticker dates as key: ``str`` and value: ``list``.

        """
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

    def _save_info_and_data(self):
        """
        Saves the info and data objects to a local file path.

        """

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
        """
        Use the `yfinance` library to fetch historical ticker data for the specified time
        period. The performance of the REST requests is highly dependent on three things:
        the config of your `CachedRateLimiter`, the amount of tickers you want to fetch,
        and the multi-threading support of your CPU.

        Parameters
        ----------
        period : str
            The time period to try and fetch data from.
        cols : list
            The columns of the fetched ticker data to collect.

        """

        info = {}
        data = {}

        for ticker in (bar := tqdm(self._symbols)):
            bar.set_description(f"Fetching ticker {ticker} data from Yahoo! Finance")

            fetched = yf.Ticker(ticker, session=self._session)
            info[ticker] = fetched.get_info(proxy=self._proxy)

            data[ticker] = fetched.history(
                period=period,
                proxy=self._proxy,
            ).reset_index()[cols]

        all_dates, dates = self._extract_dates_from_data(data)

        self._info = info
        self._data = data
        self._dates = dates
        self._all_dates = all_dates

    def load_local_files(self) -> Optional[DirectoryNotFoundError]:
        """
        Load the locally saved info and data files. The info is read from file as a
        ``json`` and the data is read from ``csv`` as a ``pd.DataFrame``.

        Raises
        ------
        DirectoryNotFoundError
            When either of the paths to the saved ``info`` and ``data`` is not a directory.

        """

        path = Path(self._save_path)
        if not path.is_dir():
            raise DirectoryNotFoundError(
                f"The local save path {path} does not exist. Perhaps you haven't "
                "tried fetching any data? To do that, run `dataset.fetch_data(...)`."
            )

        info_path = path / "info"
        data_path = path / "data"

        if not info_path.is_dir():
            raise DirectoryNotFoundError(
                f"The local save path {info_path} does not exist. Perhaps you haven't "
                "tried fetching any data? To do that, run `dataset.fetch_data(...)`."
            )

        if not data_path.is_dir():
            raise DirectoryNotFoundError(
                f"The local save path {data_path} does not exist. Perhaps you haven't "
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
        """
        Fetch the historical ticker data for the specified time period. If there exists
        locally saved files for all tickers, will try and load them instead of fetching
        from Yahoo! Finance. Saves the fetched files if ``save=True`` was specified in
        the class constructor.

        Parameters
        ----------
        period : str
            The time period to try and fetch data from. Valid values are (``1d``,
            ``5d``, ``1mo``, ``3mo``, ``6mo``, ``1y``, ``2y``, ``5y``, ``10y``,
            ``ytd``, ``max``).
        cols : list
            The columns of the fetched ticker data to collect. Defaults to
            (``Date``, ``Open``, ``High``, ``Low``, ``Close``).

        Returns
        -------
        Dataset
            The initialized instance of ``self``.

        """

        if all_tickers_saved(self._save_path, self._symbols):
            log.info(
                f"found existing local files for {self.__class__.__name__}, "
                "attempting local load..."
            )

            try:
                self.load_local_files()
                log.info("OK!")
                return self

            except DirectoryNotFoundError:
                log.warning("failed to load local files, will attempt new fetch...")

        self._fetch_tickers(period, cols)

        if self._save:
            setup_finq_save_path(self._save_path)
            self._save_info_and_data()

        return self

    def fix_missing_data(
        self,
        *,
        cols: List[str] = ["Open", "High", "Low", "Close"],
        resave: bool = True,
    ) -> Dataset:
        """
        Compares each tickers dates in their corresponding ``pd.DataFrame`` and compares
        to the known set of dates collected. If there are any missing values, will add
        the missing dates to the dataframe and then use ``df.interpolate()`` to fix them.
        Default interpolation strategy is ``linear``.

        Parameters
        ----------
        cols : list
            The columns of the ``pd.DataFrame`` to consider when looking for missing data
            to interpolate. Defaults to (``Open``, ``High``, ``Low``, ``Close``).
        resave : bool
            Whether or not to resave the data to local path after fixing missing values.
            Defaults to ``True`` but will onlyesave if there existed missing data.

        Returns
        -------
        Dataset
            The initialized instance of ``self``.

        """

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

    def run(self, period: str = "1y") -> Dataset:
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
            log.info(f"saving plot to path {save_path}")
            plt.savefig(save_path)
            log.info("OK!")

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

    def as_assets(self, price_type: str = "Close") -> Dict[str, Asset]:
        """
        Create a list of Assets for each ticker and specified price type.

        Parameters
        ----------
        price_type : str
            The price type data to create an ``Asset`` object with. Has to be one
            of (``Open``, ``High``, ``Low``, ``Close``). Defaults to ``Close``.

        Returns
        -------
        dict
            A dictionary of newly created ``Asset`` objects with ticker symbols as keys.

        """
        return {
            ticker: Asset(
                self._data[ticker][price_type],
                self._names[i],
                market=self._market,
                index_name=self._index_name,
                price_type=price_type,
            )
            for i, ticker in enumerate(self._symbols)
        }

    def as_df(self, price_type: str = "Close") -> pd.DataFrame:
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
