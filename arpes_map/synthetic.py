"""Reproducible single-band ARPES-like benchmark with known dispersion."""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit


def simulate(seed=42, ne=192, nk=100, peak_counts=12.0, missing=False, gamma_ev=0.009):
    """Simulate a Lorentzian band; gamma_ev is its intrinsic HWHM in eV."""
    if ne < 8 or nk < 8:
        raise ValueError("ne and nk must be >= 8")
    if not np.isfinite(peak_counts) or peak_counts <= 0:
        raise ValueError("peak_counts must be finite and positive")
    if not np.isfinite(gamma_ev) or gamma_ev <= 0:
        raise ValueError("gamma_ev must be finite and positive")
    rng = np.random.default_rng(seed)
    energy = np.linspace(-0.5, 0.06, ne)
    momentum = np.linspace(-0.8, 0.8, nk)
    truth = -0.29 + 0.23 * momentum**2 + 0.012 * np.sin(5 * momentum)
    gamma = gamma_ev
    peak = gamma**2 / ((energy[:, None] - truth[None, :]) ** 2 + gamma**2)
    modulation = 0.65 + 0.3 * np.cos(4 * momentum)
    clean = peak * modulation[None, :] * expit(-energy[:, None] / 0.006)
    clean += 0.04 + 0.015 * (energy[-1] - energy[:, None]) / np.ptp(energy)
    clean = gaussian_filter(clean, (0.65, 0.25), mode="nearest")
    clean /= clean.max()
    intensity = rng.poisson(clean * peak_counts).astype(float) / peak_counts
    # A detector dropout: these columns contain no local energy evidence.
    if missing:
        intensity[:, nk // 2 - 2 : nk // 2 + 2] = 0
    return dict(intensity=intensity, clean=clean, energy=energy, momentum=momentum, truth=truth)
