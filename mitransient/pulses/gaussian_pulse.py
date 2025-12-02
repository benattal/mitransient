"""Gaussian temporal pulse shape."""

from __future__ import annotations
from typing import Tuple

import drjit as dr
import mitsuba as mi

from .pulse_shape import PulseShape


class GaussianPulse(PulseShape):
    """
    Gaussian pulse shape centered at t=0.

    The pulse is defined as:
        p(t) = exp(-0.5 * (t / width)^2) / (width * sqrt(2 * pi))

    This is a normalized Gaussian with standard deviation = width.
    The pulse integrates to 1 over (-inf, +inf).

    Args:
        width_opl: Standard deviation of the Gaussian in optical path length units
    """

    def __init__(self, width_opl: float):
        self.width_opl = width_opl
        self.normalization = 1.0 / (width_opl * dr.sqrt(2.0 * dr.pi))

    def eval(self, t: mi.Float) -> mi.Float:
        """
        Evaluate Gaussian pulse at time t.

        Args:
            t: Time offset from pulse center (in OPL units)

        Returns:
            Normalized Gaussian value at t
        """
        t_normalized = t / self.width_opl
        return self.normalization * dr.exp(-0.5 * t_normalized * t_normalized)

    def sample(self, xi: mi.Float) -> Tuple[mi.Float, mi.Float]:
        """
        Sample a time offset from the Gaussian distribution using inverse CDF.

        Uses the inverse error function to sample from the Gaussian.

        Args:
            xi: Uniform random number in [0, 1]

        Returns:
            Tuple of (sampled_time, weight) where weight=1.0 for normalized Gaussian
        """
        # Inverse CDF of standard normal: sqrt(2) * erfinv(2*xi - 1)
        # Scale by width_opl for our distribution
        xi_clamped = dr.clip(xi, 1e-6, 1.0 - 1e-6)
        z = dr.sqrt(mi.Float(2.0)) * dr.erfinv(2.0 * xi_clamped - 1.0)
        sampled_time = z * self.width_opl

        # Weight is 1.0 for normalized pulse (integral = 1)
        weight = mi.Float(1.0)

        return sampled_time, weight
