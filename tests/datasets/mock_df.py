import numpy as np
import pandas as pd
from datetime import (
    datetime,
    timedelta,
)
from typing import List


def _random_df(cols: List[str]) -> pd.DataFrame:
    """Randomize some some for 30 days with given columns."""

    date_today = datetime.now()
    days = pd.date_range(date_today, date_today + timedelta(30), freq="D")

    data = np.random.normal(500, 10, size=(len(days), len(cols)))
    df = pd.DataFrame(data, columns=cols, index=days)

    df.index.name = "Date"
    df.index = pd.to_datetime(df.index)

    return df
