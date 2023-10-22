<div align="center">
<br/>
<div align="left">
<br/>
<p align="center">
<a href="https://github.com/wilhelmagren/finq">
<img align="center" width=75% src="https://github.com/wilhelmagren/finq/blob/6636192f668e45f51d2092b4b0e861e1a61a9af7/docs/images/finq-banner.png"></img>
</a>
</p>
</div>

[![PyPI - Version](https://img.shields.io/pypi/v/pyfinq)](https://pypi.org/project/pyfinq)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/wilhelmagren/finq/graph/badge.svg?token=9QTX90NYYG)](https://codecov.io/gh/wilhelmagren/finq)
[![CI](https://github.com/wilhelmagren/finq/actions/workflows/ci.yml/badge.svg)](https://github.com/wilhelmagren/finq/actions/workflows/ci.yml)
[![CD](https://github.com/wilhelmagren/finq/actions/workflows/cd.yml/badge.svg)](https://github.com/wilhelmagren/finq/actions/workflows/cd.yml)
[![Tests](https://github.com/wilhelmagren/finq/actions/workflows/tests.yml/badge.svg)](https://github.com/wilhelmagren/finq/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

</div>

## 🔎 Overview
The goal of *finq* is to provide an all-in-one Python library for **quantitative portfolio analysis and optimization** on historical and real-time financial data.

**NOTE:** Features are currently being determined and developed continuously. The repo is undergoing heavy modifications and could introduce **breaking changes** up until first major release. Current version is **v0.3.0**.

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
symbols = ["ALFA.ST", "BOL.ST", "SEB-A.ST", "SHB-A.ST"]

dataset = CustomDataset(names, symbols, market="OMX", save=True)
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

dataset = CustomDataset(index_name='NDX', market="NASDAQ", save=False)
dataset.run("1y")

closing_prices = dataset.as_numpy("Close")
...
```

## 📋 License
All code is to be held under a general MIT license, please see [LICENSE](https://github.com/wilhelmagren/finq/blob/main/LICENSE) for specific information.
