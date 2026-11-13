"""Reproducible ARPES-like benchmarks with independent approximate dispersions."""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit


def approximate_dispersion(momentum, bands=1):
    """Analytic initial model, evaluated without access to truth or intensities.

    Its offsets and effective curvatures differ from the data-generating model;
    it also omits the sinusoidal structure present in truth.
    """
    if bands not in (1, 2, 3):
        raise ValueError("bands must be 1, 2 or 3")
    k = np.asarray(momentum, dtype=float)
    middle = -0.275 + 0.20 * k**2
    if bands == 1:
        return middle
    upper = -0.080 - 0.025 * k**2
    if bands == 2:
        return np.stack((middle, upper))
    lower = -0.385 + 0.085 * k**2
    return np.stack((lower, middle, upper))


def simulate(seed=42, ne=192, nk=100, peak_counts=12.0, missing=False, gamma_ev=0.009, bands=1):
    """Simulate 1–3 noncrossing Lorentzian bands; gamma_ev is their HWHM.

    Single-band arrays retain their original 1D shape. Multiband truth and
    initial_guess have shape (bands, nk); intensities always have shape (ne, nk).
    """
    if bands not in (1, 2, 3):
        raise ValueError("bands must be 1, 2 or 3")
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
    if bands > 1:
        upper = -0.095 - 0.018 * momentum**2 + 0.008 * np.cos(4 * momentum)
        if bands == 2:
            truth = np.stack((truth, upper))
        else:
            lower = -0.405 + 0.11 * momentum**2 + 0.009 * np.sin(3 * momentum)
            truth = np.stack((lower, truth, upper))
    gamma = gamma_ev
    clean = np.zeros((ne, nk))
    for band, curve in enumerate(np.atleast_2d(truth)):
        peak = gamma**2 / ((energy[:, None] - curve[None, :]) ** 2 + gamma**2)
        modulation = (0.65 + 0.3 * np.cos(4 * momentum + 0.7 * band)) * (1 - 0.12 * band)
        clean += peak * modulation[None, :] * expit(-energy[:, None] / 0.006)
    clean += 0.04 + 0.015 * (energy[-1] - energy[:, None]) / np.ptp(energy)
    clean = gaussian_filter(clean, (0.65, 0.25), mode="nearest")
    clean /= clean.max()
    intensity = rng.poisson(clean * peak_counts).astype(float) / peak_counts
    # A detector dropout: these columns contain no local energy evidence.
    if missing:
        intensity[:, nk // 2 - 2 : nk // 2 + 2] = 0
    return dict(intensity=intensity, clean=clean, energy=energy, momentum=momentum, truth=truth,
                initial_guess=approximate_dispersion(momentum, bands))
