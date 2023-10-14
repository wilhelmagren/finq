<div align="center">
<br/>
<div align="left">
<br/>
<p align="center">
</p>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/wilhelmagren/finq/graph/badge.svg?token=9QTX90NYYG)](https://codecov.io/gh/wilhelmagren/finq)
[![CI](https://github.com/wilhelmagren/finq/actions/workflows/ci.yml/badge.svg)](https://github.com/wilhelmagren/finq/actions/workflows/ci.yml)
[![Tests](https://github.com/wilhelmagren/finq/actions/workflows/tests.yml/badge.svg)](https://github.com/wilhelmagren/finq/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

</div>

## 🔎 Overview
The goal of *finq* is to provide an all-in-one Python module for quantiative analysis on
historical and real-time financial data.

## 📦 Installation
Either clone this repository and perform a local install with [poetry](https://github.com/python-poetry/poetry/tree/master) accordingly
```
git clone https://github.com/wilhelmagren/finq.git
cd finq
poetry install
```
or install the most recent release from the Python Package Index (PyPI).
```
pip install pyfinq
```

## 🚀 Example usage
The standard way to define a custom dataset is to provide a list of security names and
their related ticker symbols,

```python
from finq.datasets import CustomDataset

names = ["Alfa Laval", "Boliden", "SEB A", "Sv. Handelsbanken A"]
symbols = ["AFLA.ST", "BOL.ST", "SEB-A.ST", "SHB-A.ST"]

dataset = CustomDataset(names, symbols, save=True)
dataset.fetch_data("3y") \
    .fix_missing_data() \
    .verify_data()

dataset.visualize(log_scale=True)
...
```

or if you don't know the syntax of the ticker symbols that you want, you can pass in a
valid NASDAQ index name and try and fetch its related ticker components. We can also
call the function `.run(period)` which performs the three steps the above cell does
(fetching, fixing, and verifying).
```python
from finq.datasets import CustomDataset

dataset = CustomDataset(nasdaq_index='NDX', save=False)
dataset.run("1y")

closing_prices = dataset.as_numpy("Close")
...
```

## 📋 License
All code is to be held under a general MIT license, please see [LICENSE](https://github.com/wilhelmagren/finq/blob/main/LICENSE) for specific information.

