import yfinance as yf
from requests import Session
from requests_cache import CacheMixin
from requests_ratelimiter import LimiterMixin
from typing import (
    Iterable,
    Optional,
)


class CachedRateLimiter(CacheMixin, LimiterMixin, Session):
    """A cached rate-limiter session for accessing the Yahoo! Finance api more efficiently."""
    pass

def fetch_tickers(
    tickers: Iterable[str],
    period: str,
    *,
    session: Optional[CachedRateLimiter] = None,
):
    """Fetch historical data from the Yahoo! Finance api."""
    yf_tickers = yf.Tickers(tickers, session=session)
    data = yf_tickers.history()
    return data