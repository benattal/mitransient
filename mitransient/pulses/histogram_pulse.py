"""Histogram-based temporal pulse shape."""

from __future__ import annotations
from typing import Tuple, Union

import drjit as dr
import mitsuba as mi
import numpy as np

from .pulse_shape import PulseShape


class HistogramPulse(PulseShape):
    """
    Histogram-based pulse shape from a 1D array.

    The histogram defines pulse values at discrete time points.
    Values between bins are linearly interpolated.

    The pulse is automatically normalized so that it integrates to 1.

    The pulse is defined over the range [start_opl, start_opl + num_bins * bin_width_opl].
    Outside this range, the pulse value is 0.

    Args:
        values: Either a NumPy array of pulse values, or a file path to load from
        start_opl: Start time of the histogram (in OPL units).
                   For a pulse centered at t=0, use negative value.
        bin_width_opl: Width of each histogram bin (in OPL units)
    """

    def __init__(self, values: Union[np.ndarray, str],
                 start_opl: float, bin_width_opl: float):
        # Load values from file if string path provided
        if isinstance(values, str):
            values = np.load(values)

        # Ensure values is a 1D float array
        values = np.asarray(values, dtype=np.float32).flatten()

        self.num_bins = len(values)
        self.start_opl = start_opl
        self.bin_width_opl = bin_width_opl
        self.end_opl = start_opl + self.num_bins * bin_width_opl

        # Normalize the pulse so that the integral equals 1
        # Integral ≈ sum(values) * bin_width_opl (treating each bin as a rectangle)
        total_integral = np.sum(values) * bin_width_opl
        if total_integral > 0:
            values = values / total_integral

        # Store normalized values as DrJit array
        self.values = mi.Float(values)

        # Build CDF for importance sampling (based on bin values)
        pmf = values * bin_width_opl  # Each bin's probability mass
        cdf = np.cumsum(pmf)
        cdf = cdf / cdf[-1] if cdf[-1] > 0 else np.linspace(0, 1, len(cdf))
        self.cdf = mi.Float(cdf)

    def eval(self, t: mi.Float) -> mi.Float:
        """
        Evaluate histogram pulse at time t using linear interpolation.

        Args:
            t: Time offset (in OPL units)

        Returns:
            Interpolated pulse value at t (0 if outside histogram range)
        """
        # Convert time to bin index (floating point)
        bin_idx_f = (t - self.start_opl) / self.bin_width_opl

        # Get integer bin index (floor)
        bin_idx = mi.Int32(dr.floor(bin_idx_f))

        # Interpolation weight (fractional part)
        alpha = bin_idx_f - dr.floor(bin_idx_f)

        # Check bounds for both bins we need for interpolation
        valid = (bin_idx >= 0) & (bin_idx < self.num_bins - 1)
        valid_edge = (bin_idx == self.num_bins - 1)  # Last bin, no interpolation

        # Gather values (with bounds checking via valid mask)
        v0 = dr.gather(mi.Float, self.values, dr.clip(bin_idx, 0, self.num_bins - 1), valid | valid_edge)
        v1 = dr.gather(mi.Float, self.values, dr.clip(bin_idx + 1, 0, self.num_bins - 1), valid)

        # Linear interpolation
        interpolated = dr.lerp(v0, v1, alpha)

        # Return interpolated value if valid, edge value if at last bin, else 0
        result = dr.select(valid, interpolated,
                          dr.select(valid_edge, v0, mi.Float(0.0)))

        return result

    def sample(self, xi: mi.Float) -> Tuple[mi.Float, mi.Float]:
        """
        Importance sample a time offset from the histogram distribution.

        Uses inverse CDF sampling with jittering within the selected bin.

        Args:
            xi: Uniform random number in [0, 1]

        Returns:
            Tuple of (sampled_time, weight) where weight=1.0 for normalized histogram
        """
        # Binary search in CDF to find bin
        bin_idx = dr.clip(
            dr.binary_search(0, self.num_bins - 1,
                           lambda idx: dr.gather(mi.Float, self.cdf, idx) < xi),
            0, self.num_bins - 1
        )

        # Get CDF values at bin boundaries for interpolation
        cdf_low = dr.select(
            bin_idx > 0,
            dr.gather(mi.Float, self.cdf, dr.clip(bin_idx - 1, 0, self.num_bins - 1)),
            mi.Float(0.0)
        )
        cdf_high = dr.gather(mi.Float, self.cdf, bin_idx)

        # Linear interpolation within bin
        bin_width_cdf = dr.maximum(cdf_high - cdf_low, 1e-10)
        alpha = (xi - cdf_low) / bin_width_cdf

        # Compute sampled time
        sampled_time = self.start_opl + (mi.Float(bin_idx) + alpha) * self.bin_width_opl

        # Weight is 1.0 for normalized pulse (integral = 1)
        weight = mi.Float(1.0)

        return sampled_time, weight
