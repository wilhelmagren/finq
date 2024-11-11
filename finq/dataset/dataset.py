from __future__ import annotations

from finq.datautil import fetch_tickers
from typing import Iterable


class Dataset(object):
    """"""

    @classmethod
    def with_tickers(cls, tickers: Iterable[str]) -> Dataset:
        """"""
        data = fetch_tickers(tickers)
        cls()