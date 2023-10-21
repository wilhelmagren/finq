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

File created: 2023-10-12
Last updated: 2023-10-21
"""

import logging
import os
import random
import string
import requests
import pandas as pd

from requests.exceptions import HTTPError
from datetime import (
    datetime,
    timedelta,
)
from typing import (
    Any,
    Union,
    List,
    Dict,
    Tuple,
    Optional,
    Callable,
)

log = logging.getLogger(__name__)

BASE_URL = "https://indexes.nasdaqomx.com/Index/ExportWeightings/"

IMPLEMENTED_INDEX = (
    "NDX",
    "OMXS30",
    "OMXSBESGNI",
    "OMXSPI",
)


def fetch_names_and_symbols(
    index: str,
    *,
    session: Optional[requests.Session] = None,
    query_params: Dict[Any, Any] = {},
    headers: Dict[Any, Any] = {},
    market: str = "OMX",
    filter_symbols: Callable = lambda s: s,
) -> Union[HTTPError, Tuple[List[str], List[str]]]:
    """ """

    if index not in IMPLEMENTED_INDEX:
        log.warning(
            f"`{index}` is not a natively implemented index, "
            "but will attempt to fetch from NASDAQ..."
        )

    url = BASE_URL + index

    today = datetime.today()
    if today.hour < 12:
        today = today - timedelta(1)

    weekday_diff = max(today.isoweekday() - 5, 0)
    last_weekday = datetime.date(today - timedelta(weekday_diff)).strftime("%Y-%m-%d")

    params = {
        "tradeDate": f"{last_weekday}T00:00:00.000",
        "timeOfDay": "SOD",
    }

    query_params = {**query_params, **params}

    log.info(f"performing GET request to: `{url}`")
    log.debug(f"with query parameters: `{query_params}`")
    log.debug(f"with headers: `{headers}`")

    if session is None:
        session = requests

    response = session.get(
        url,
        params=query_params,
        headers=headers,
    )

    if response.status_code != 200:
        raise HTTPError(f"Could not get the index components from nasdaq, {response}")

    log.info(f"{response.status_code} OK")

    rand_string = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
    tmp_xlsx_path = f"{rand_string}-{index}.xlsx"

    with open(tmp_xlsx_path, "wb") as f:
        f.write(response.content)

    log.debug(f"attempting to read excel at `{tmp_xlsx_path}`...")
    df = pd.read_excel(
        tmp_xlsx_path,
        names=("Company Name", "Security Symbol"),
        skiprows=4,
        engine="openpyxl",
    ).dropna()

    os.remove(tmp_xlsx_path)
    log.debug("OK!")

    names = df["Company Name"].tolist()
    symbols = list(
        map(
            filter_symbols,
            df["Security Symbol"],
        )
    )

    return (names, symbols)
