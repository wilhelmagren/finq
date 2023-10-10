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
Last updated: 2023-10-10
"""

from finq.datasets import Dataset

from pathlib import Path
from typing import (
    Dict,
    Union,
)

names = [
    "ABB Ltd",
    "Alfa Laval",
    "Autoliv SDB",
    "ASSA ABLOY B",
    "Atlas Copco A",
    "Atlas Copco B",
    "AstraZeneca",
    "Boliden",
    "Electrolux B",
    "Ericsson B",
    "Essity B",
    "Evolution",
    "Getinge B",
    "Hexagon B",
    "Hennes & Mauritz B",
    "Investor B",
    "Kinnevik B",
    "Nordea Bank Abp",
    "NIBE Industrier B",
    "Sandvik",
    "Samhällsbyggnadbo.i Norden AB",
    "SCA B",
    "SEB A",
    "Sv. Handelsbanken A",
    "Sinch",
    "SKF B",
    "Swedbank A",
    "Tele2 B",
    "Telia Company",
    "Volvo B",
]

symbols = [
    "ABB.ST",
    "ALFA.ST",
    "ALIV-SDB.ST",
    "ASSA-B.ST",
    "ATCO-A.ST",
    "ATCO-B.ST",
    "AZN.ST",
    "BOL.ST",
    "ELUX-B.ST",
    "ERIC-B.ST",
    "ESSITY-B.ST",
    "EVO.ST",
    "GETI-B.ST",
    "HEXA-B.ST",
    "HM-B.ST",
    "INVE-B.ST",
    "KINV-B.ST",
    "NDA-SE.ST",
    "NIBE-B.ST",
    "SAND.ST",
    "SBB-B.ST",
    "SCA-B.ST",
    "SEB-A.ST",
    "SHB-A.ST",
    "SINCH.ST",
    "SKF-B.ST",
    "SWED-A.ST",
    "TEL2-B.ST",
    "TELIA.ST",
    "VOLV-B.ST",
]


class OMXS30(Dataset):
    """ """

    def __init__(
        self,
        *,
        save_path: Union[str, Path] = ".data/OMXS30/",
        **kwargs: Dict,
    ):
        """ """

        super(OMXS30, self).__init__(
            names,
            symbols,
            save_path=save_path,
            **kwargs,
        )
