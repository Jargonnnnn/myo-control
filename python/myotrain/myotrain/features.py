"""Time-domain EMG features — the contract shared with the Rust loop.

These mirror ``crates/myo-rt/src/features.rs`` *exactly* (same ZC/SSC
threshold convention), so a model trained here decodes correctly there. We
deliberately reimplement them rather than use LibEMG, whose threshold
conventions may differ and would silently break the cross-language contract.

Feature order per channel is ``[rms, mav, wl, zc, ssc]``, flattened
channel-major — identical to ``FeatureSet::to_vec`` on the Rust side.
"""

from __future__ import annotations

import numpy as np

FEATURES: list[str] = ["rms", "mav", "wl", "zc", "ssc"]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def mav(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x))) if x.size else 0.0


def wl(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def zc(x: np.ndarray, threshold: float) -> float:
    """Zero crossings: opposite-sign neighbours whose gap clears ``threshold``."""
    count = 0
    for i in range(x.size - 1):
        if x[i] * x[i + 1] < 0.0 and abs(x[i + 1] - x[i]) >= threshold:
            count += 1
    return float(count)


def ssc(x: np.ndarray, threshold: float) -> float:
    """Slope sign changes: interior turning points exceeding ``threshold``."""
    count = 0
    for i in range(1, x.size - 1):
        if (x[i] - x[i - 1]) * (x[i] - x[i + 1]) >= threshold:
            count += 1
    return float(count)


def window_features(window: np.ndarray, threshold: float) -> list[float]:
    """Feature vector for one window of shape ``[n_samples, n_channels]``.

    Channel-major ``[rms, mav, wl, zc, ssc]`` per channel.
    """
    channels = window.shape[1]
    out: list[float] = []
    for c in range(channels):
        ch = window[:, c]
        out.extend([rms(ch), mav(ch), wl(ch), zc(ch, threshold), ssc(ch, threshold)])
    return out
