"""Opt-in CPU thread cap for shared multi-user nodes.

Dr.Jit sizes its nanothread pool to every visible core (192 on the lab
H100 nodes) and Torch follows ``OMP_NUM_THREADS``. On a shared node a CUDA
render therefore still spawns hundreds of idle-but-scheduled workers. When the
environment variable ``NLOS_MAX_THREADS`` holds a positive integer, importing
``mitransient`` caps Dr.Jit's pool and Torch's intra-op pool to that value.
Unset or invalid values leave both libraries at their defaults.
"""
from __future__ import annotations

import os

ENV_VAR = "NLOS_MAX_THREADS"


def requested_thread_limit(environ=None) -> int | None:
    """Return the positive integer cap from ``NLOS_MAX_THREADS`` or ``None``."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def apply_thread_limits(environ=None) -> int | None:
    """Cap Dr.Jit and (already imported) Torch worker pools; return the cap."""
    limit = requested_thread_limit(environ)
    if limit is None:
        return None
    try:
        import drjit as dr

        dr.set_thread_count(limit)
    except Exception:  # pragma: no cover - defensive on exotic builds
        pass
    import sys

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.get_num_threads() > limit:
                torch.set_num_threads(limit)
        except Exception:  # pragma: no cover
            pass
    return limit
