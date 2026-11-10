"""Reproduce the two-method convolution check shown in the portfolio.

Dependencies: NumPy and SciPy. Synthetic PMFs only.
Run: python flow-benchmark.py
"""
from __future__ import annotations

import json
import platform
from statistics import median
from time import perf_counter_ns

import numpy as np
import scipy
from scipy.signal import fftconvolve


SIZES = (1024, 4096, 8192)
REPEATS = 9
SEED = 202511
ATOL = 1e-12
RTOL = 1e-10


def measure_ms(function, a, b):
    started = perf_counter_ns()
    result = function(a, b, mode="full")
    return (perf_counter_ns() - started) / 1_000_000, result


def compare(size):
    rng = np.random.default_rng(SEED + size)
    a, b = rng.random((2, size))
    a /= a.sum()
    b /= b.sum()
    direct_ms, fft_ms = [], []
    max_error = 0.0
    max_mass_error = 0.0
    functions = ((np.convolve, direct_ms), (fftconvolve, fft_ms))
    for method, _ in functions:
        method(a, b, mode="full")
    for run in range(REPEATS):
        results = {}
        for method, times in functions[::1 if run % 2 == 0 else -1]:
            elapsed, output = measure_ms(method, a, b)
            times.append(elapsed)
            results[method] = output
        direct, fft = results[np.convolve], results[fftconvolve]
        np.testing.assert_allclose(direct, fft, rtol=RTOL, atol=ATOL)
        max_error = max(max_error, float(np.max(np.abs(direct - fft))))
        max_mass_error = max(max_mass_error, abs(float(fft.sum()) - 1.0))
    return {
        "bins_each": size,
        "output_bins": 2 * size - 1,
        "direct_median_ms": round(median(direct_ms), 6),
        "fft_median_ms": round(median(fft_ms), 6),
        "max_absolute_error": max_error,
        "max_probability_mass_error": max_mass_error,
    }


if __name__ == "__main__":
    print(json.dumps({
        "input": "synthetic normalized random PMFs",
        "seed": SEED,
        "repeats": REPEATS,
        "absolute_tolerance": ATOL,
        "relative_tolerance": RTOL,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "runs": [compare(size) for size in SIZES],
    }, indent=2))
