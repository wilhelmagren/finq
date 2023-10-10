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

File created: 2023-10-09
Last updated: 2023-10-09
"""

from __future__ import annotations

import logging
import os
import numpy as np

from finq.portfolio.solution import Solution
from finq.errors import NoFeasibleSolutionError
from dwave.system import LeapHybridCQMSampler
from typing import (
    Callable,
    Optional,
    Union,
)

log = logging.getLogger(__name__)

class Portfolio(object):
    """ """

    def __init__(self, *args, **kwargs):
        """ """
        pass

    @staticmethod
    def _divide(u: np.ndarray, v: np.ndarray) -> Union[Exception, np.ndarray]:
        """ """
        if u == 0:
            if v == 0:
                return np.ones(u.shape).astype(u.dtype)
            raise ValueError(
                f'Can not divide by zero, {u}/{v}.'
            )
        return u / v

    def optimize(self, *,
        token: Optional[str] = None,
        filter_: Optional[Callable] = None,
        label: str = 'CQM Portfolio',
        time_limit: int = 10,
    ) -> Solution:
        """ """

        if token is None:
            log.warn('No API token passed to the optimize function, checking for `DWAVE_TOKEN` in environment variables...')
            token = os.getenv('DWAVE_TOKEN', None)
        
        if token is None:
            raise ValueError(
                f'You need to provide an API token to access `D-Wave Leap QPUs`, token=`{token}`'
            )
        
        if filter_ is None:
            log.info('No filtering method provided, will use: `lambda s: s.is_feasible`')
            filter_ = lambda s: s.is_feasible
        
        sampler = LeapHybridCQMSampler(token=token)
        log.info('Sampling `{label}` using CQM on D-Wave Leap...')
        sample_set = sampler.sample_cqm(self._model, label=label, time_limit=time_limit)

        n_samples = len(sample_set.record)
        feasible_samples = sample_set.filter(filter_)

        if not feasible_samples:
            raise NoFeasibleSolutionError(
                f'Found no feasible solutions from `{n_samples}` number of samples'
            )
        
        optimal = feasible_samples.first.sample
        optimal_solution = Solution(optimal)

        print(f'{len(feasible_samples)} feasible solutions sampled.')
        print(f'Lowest energy:\t\t\t{sample_set.first.energy:.3f}')
        print(f'Lowest energy (feasible):\t{feasible_samples.first.energy:.3f}')

        return optimal_solution

    def sample_random_solutions(self,
        n_samples: int,
        **kwargs: dict,
    ) -> list[Solution]:
        """ """
        pass
