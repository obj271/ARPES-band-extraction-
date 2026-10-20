"""Curvature and minimum-gradient intensity maps for visual comparison."""

import numpy as np
from scipy.ndimage import gaussian_filter

from .inference import validate_data


def comparison_intensities(intensity, energy, momentum, *, smoothing=(0.7, 0.0)):
    """Return nonnegative, globally peak-normalized maps on the input grid.

    Both methods use the same globally scaled, Gaussian-smoothed input, before
    any MAP background subtraction. Coordinates are divided by their median
    spacing to balance energy and momentum in pixel-equivalent units, retaining
    irregular spacing. Curvature uses Zhang et al.'s 2D formula with Cx=Cy=1;
    minimum gradient uses He et al.'s eight neighbor directional differences.
    A floor of 1e-3 times the largest gradient modulus regularizes division.
    Original columns without energy contrast are masked to zero.
    """
    intensity, energy, momentum = validate_data(intensity, energy, momentum)
    smoothing = np.asarray(smoothing, dtype=float)
    if smoothing.shape != (2,) or not np.all(np.isfinite(smoothing)) or np.any(smoothing < 0):
        raise ValueError("smoothing must contain two finite nonnegative pixel sigmas")
    peak = intensity.max()
    raw_scale = intensity.max(axis=0)
    observed = np.ptp(intensity, axis=0) > 1e-12 * np.maximum(raw_scale, np.finfo(float).tiny)
    if not np.any(observed):
        return {name: np.zeros_like(intensity)
                for name in ("curvature_intensity", "gradient_intensity")}
    f = gaussian_filter(intensity / peak, smoothing, mode="nearest")
    y = (energy - energy[0]) / np.median(np.diff(energy))
    x = (momentum - momentum[0]) / np.median(np.diff(momentum))
    fy, fx = np.gradient(f, y, x)
    fyy, fyx = np.gradient(fy, y, x)
    fxy, fxx = np.gradient(fx, y, x)
    mixed = (fyx + fxy) / 2
    curvature = -((1 + fx**2) * fyy - 2 * fx * fy * mixed + (1 + fy**2) * fxx)
    curvature /= (1 + fx**2 + fy**2) ** 1.5
    curvature = np.maximum(curvature, 0)

    # Boundary pixels use only the neighbors present in the measured grid.
    # RMS rather than sum keeps the boundary scale consistent with the interior;
    # the global factor sqrt(8) disappears during final peak normalization.
    squared_gradient = np.zeros_like(f)
    neighbor_count = np.zeros_like(f)
    ne, nk = f.shape
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)):
        rows = slice(max(0, -dy), min(ne, ne - dy))
        cols = slice(max(0, -dx), min(nk, nk - dx))
        next_rows = slice(max(0, dy), min(ne, ne + dy))
        next_cols = slice(max(0, dx), min(nk, nk + dx))
        distance = np.hypot((y[next_rows] - y[rows])[:, None],
                            (x[next_cols] - x[cols])[None, :])
        difference = (f[next_rows, next_cols] - f[rows, cols]) / distance
        squared_gradient[rows, cols] += difference**2
        neighbor_count[rows, cols] += 1
    modulus = np.sqrt(squared_gradient / neighbor_count)
    floor = max(1e-3 * modulus.max(), np.finfo(float).eps)
    gradient = f / np.maximum(modulus, floor)

    maps = dict(curvature_intensity=curvature, gradient_intensity=gradient)
    for values in maps.values():
        values[:, ~observed] = 0
        maximum = values.max()
        if maximum > 0:
            values /= maximum
    return maps
