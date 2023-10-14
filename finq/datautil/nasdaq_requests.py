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
Last updated: 2023-10-12
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
    Union,
    List,
    Dict,
    Tuple,
    Optional,
    Callable,
)

log = logging.getLogger(__name__)

BASE_URL = "https://indexes.nasdaqomx.com/Index/ExportWeightings/"

SUPPORTED_INDEX = (
    "OMXS30",
    "OMXSPI",
    "NDX",
)


def _fetch_names_and_symbols(
    index: str,
    *,
    query_params: Dict = {},
    headers: Dict = {},
    filter_symbols: Optional[Callable] = None,
) -> Union[Exception, Tuple[List[str], List[str]]]:
    """ """

    if index not in SUPPORTED_INDEX:
        raise ValueError(
            f"`{index}` is not a currently supported index, did you mean ",
            f"one of the following? `{','.join(SUPPORTED_INDEX)}`",
        )

    url = BASE_URL + index

    weekday_diff = 7 - datetime.today().isoweekday()
    last_weekday = datetime.date(datetime.today() - timedelta(weekday_diff)).strftime(
        "%Y-%m-%d"
    )

    params = {
        "tradeDate": f"{last_weekday}T00:00:00.000",
        "timeOfDay": "SOD",
    }

    query_params = {**query_params, **params}

    log.info(f"performing GET request to: `{url}`")
    log.info(f"with query parameters: `{query_params}`")
    log.info(f"with headers: `{headers}`")

    response = requests.get(
        url,
        params=query_params,
        headers=headers,
    )

    if response.status_code != 200:
        raise HTTPError(
            f"Could not get the index components from nasdaq, {response}",
        )

    log.info(f"{response.status_code} OK")

    rand_string = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
    tmp_xlsx_path = f"{rand_string}-{index}.xlsx"

    with open(tmp_xlsx_path, "wb") as f:
        f.write(response.content)

    log.info(f"attempting to read excel at `{tmp_xlsx_path}`...")
    df = pd.read_excel(
        tmp_xlsx_path,
        names=("Company Name", "Security Symbol"),
        skiprows=4,
        engine="openpyxl",
    ).dropna()

    os.remove(tmp_xlsx_path)
    log.info("OK!")

    if filter_symbols is None:
        filter_symbols = lambda s: s.replace(" ", "-") + ".ST"

    names = df["Company Name"].tolist()
    symbols = list(
        map(
            filter_symbols,
            df["Security Symbol"],
        )
    )

    return (names, symbols)
