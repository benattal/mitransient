"""Temporal pulse shapes for time-domain transient rendering."""

import mitsuba as mi

# Only load if a non-scalar variant is set
if mi.variant() is not None and not mi.variant().startswith('scalar'):
    from .pulse_shape import PulseShape
    from .gaussian_pulse import GaussianPulse
    from .histogram_pulse import HistogramPulse

    __all__ = ['PulseShape', 'GaussianPulse', 'HistogramPulse']
else:
    __all__ = []
