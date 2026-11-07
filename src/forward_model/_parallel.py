"""Run a block function over curve chunks on a thread pool.

Threads rather than processes: the work is almost entirely NumPy ufuncs, which release the
GIL, so threads scale nearly linearly here without paying to pickle arrays across process
boundaries. Blocks are written back by index, so the result is bit-identical to the serial
version regardless of worker count or completion order.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

# Past this, memory bandwidth rather than cores is the limit, and more workers only add
# contention.
MAX_WORKERS = 8
# Below this much work, the pool costs more than it saves.
_SERIAL_THRESHOLD = 512


def default_workers() -> int:
    return max(1, min(MAX_WORKERS, os.cpu_count() or 1))


def map_curve_blocks(
        params: NDArray[np.float64],
        block: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        n_columns: int,
        curve_chunk: int,
        workers: Optional[int] = None,
) -> NDArray[np.float64]:
    out = np.empty((params.shape[0], n_columns), dtype=np.float64)
    bounds = [(start, min(start + curve_chunk, params.shape[0]))
              for start in range(0, params.shape[0], curve_chunk)]

    if workers is None:
        workers = default_workers()
    workers = min(workers, len(bounds))

    def run(span):
        start, stop = span
        out[start:stop] = block(params[start:stop])

    if workers <= 1 or params.shape[0] < _SERIAL_THRESHOLD:
        for span in bounds:
            run(span)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, bounds))
    return out
