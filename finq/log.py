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

File created: 2023-10-15
Last updated: 2023-10-18
"""

import logging
from enum import Enum


class Color(Enum):
    BLACK = "\x1b[0;30m"
    RED = "\x1b[0;31m"
    GREEN = "\x1b[0;32m"
    YELLOW = "\x1b[0;33m"
    BLUE = "\x1b[0;34m"
    PURPLE = "\x1b[0;35m"
    CYAN = "\x1b[0;36m"
    WHITE = "\x1b[0;37m"

    BOLD_RED = "\x1b[1;31m"
    UNDERLINE_RED = "\x1b[4;31m"
    INTENSE_RED = "\x1b[0;91m"
    INTENSE_BOLD_RED = "\x1b[1;91m"

    RESET = "\x1b[0m"


class ColoredFormatter(logging.Formatter):
    """ """

    def __init__(self, format: str):
        """ """
        super(ColoredFormatter, self).__init__()

        self._formats = {
            logging.DEBUG: Color.GREEN.value + format + Color.RESET.value,
            logging.INFO: Color.WHITE.value + format + Color.WHITE.value,
            logging.WARNING: Color.YELLOW.value + format + Color.RESET.value,
            logging.ERROR: Color.RED.value + format + Color.RESET.value,
            logging.CRITICAL: Color.BOLD_RED.value + format + Color.RESET.value,
        }

    def format(self, record: logging.LogRecord) -> str:
        """ """

        log_format = self._formats.get(record.levelno, None)
        if log_format is None:
            raise ValueError(
                f"Unexpected log level found in provided logRecord, `{record.levelno}`.",
            )

        formatter = logging.Formatter(log_format)
        return formatter.format(record)
