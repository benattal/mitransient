"""Base class for temporal pulse shapes."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

import drjit as dr
import mitsuba as mi


class PulseShape(ABC):
    """
    Abstract base class for temporal pulse shapes.

    All pulses are centered at t=0, meaning:
    - eval(0) returns the maximum value
    - eval(t) decreases as |t| increases

    Pulses should be normalized so that the integral equals 1.
    """

    @abstractmethod
    def eval(self, t: mi.Float) -> mi.Float:
        """
        Evaluate the pulse at time t.

        Args:
            t: Time offset from pulse center (can be negative)

        Returns:
            Pulse amplitude at time t
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, xi: mi.Float) -> Tuple[mi.Float, mi.Float]:
        """
        Importance sample a time offset from the pulse distribution.

        Args:
            xi: Uniform random number in [0, 1]

        Returns:
            Tuple of (sampled_time, weight) where:
            - sampled_time: Time offset sampled from pulse distribution
            - weight: Weight to multiply the sample by. For normalized pulses,
                      this is 1.0 (the integral of the pulse).
        """
        raise NotImplementedError
