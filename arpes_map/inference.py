"""Inference on a chain of discrete band energies; no neural-network training."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp


@dataclass
class ExtractionResult:
    band_energy: np.ndarray
    band_indices: np.ndarray
    likelihood: np.ndarray
    marginals: np.ndarray
    posterior_mean: np.ndarray
    lower_energy: np.ndarray
    upper_energy: np.ndarray
    observed: np.ndarray
    negative_log_posterior: float


def validate_data(intensity, energy, momentum):
    """Require linear, nonnegative intensities and strictly increasing axes."""
    intensity = np.asarray(intensity, dtype=float)
    energy = np.asarray(energy, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    if energy.ndim != 1 or momentum.ndim != 1:
        raise ValueError("energy and momentum must be one-dimensional")
    if energy.size < 2 or momentum.size < 2:
        raise ValueError("both axes must contain at least two points")
    if intensity.shape != (energy.size, momentum.size):
        raise ValueError("intensity shape must be (energy.size, momentum.size)")
    for name, axis in (("energy", energy), ("momentum", momentum)):
        if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
            raise ValueError(f"{name} must be finite and strictly increasing")
    if not np.all(np.isfinite(intensity)) or np.any(intensity < 0):
        raise ValueError("intensity must be finite and nonnegative")
    return intensity, energy, momentum


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _chain_inference(unary, energy, sigmas):
    """Exact joint MAP and node marginals for Gaussian pairwise potentials.

    A uniform initial state is implicit. Pairwise factors are unnormalized
    Gaussian MRF potentials, not row-normalized transition probabilities.
    """
    ne, nk = unary.shape
    squared_difference = (energy[:, None] - energy[None, :]) ** 2
    back = np.zeros((nk, ne), dtype=np.intp)
    score = unary[:, 0].copy()
    forward = np.empty_like(unary)
    forward[:, 0] = -unary[:, 0]
    forward[:, 0] -= logsumexp(forward[:, 0])
    for j in range(1, nk):
        penalty = 0.5 * squared_difference / sigmas[j - 1] ** 2
        candidates = score[:, None] + penalty
        back[j] = np.argmin(candidates, axis=0)
        score = unary[:, j] + candidates[back[j], np.arange(ne)]
        forward[:, j] = -unary[:, j] + logsumexp(
            forward[:, j - 1, None] - penalty, axis=0
        )
        forward[:, j] -= logsumexp(forward[:, j])
    indices = np.empty(nk, dtype=np.intp)
    indices[-1] = np.argmin(score)
    optimum = float(score[indices[-1]])
    for j in range(nk - 1, 0, -1):
        indices[j - 1] = back[j, indices[j]]

    backward = np.zeros(ne)
    marginals = np.empty_like(unary)
    for j in range(nk - 1, -1, -1):
        log_marginal = forward[:, j] + backward
        marginals[:, j] = np.exp(log_marginal - logsumexp(log_marginal))
        if j > 0:
            penalty = 0.5 * squared_difference / sigmas[j - 1] ** 2
            backward = logsumexp(
                -penalty - unary[None, :, j] + backward[None, :], axis=1
            )
            backward -= logsumexp(backward)
    return indices, optimum, marginals


def extract_band(
    intensity,
    energy,
    momentum,
    *,
    eta_ev=0.012,
    smoothing=(0.7, 0.0),
    background_percentile=10.0,
    likelihood_floor=1e-3,
    reference=None,
    reference_sigma_ev=0.04,
    anchors=(),
    anchor_sigma_ev=0.008,
    candidate_mask=None,
):
    """Extract one dominant band from I[energy, momentum].

    Minimize -sum(log L(E_j,j)) + sum((E_j-E_{j-1})^2/(2 sigma_j^2))
    plus optional Gaussian reference/anchor penalties. eta_ev is the neighbor
    energy standard deviation at the median momentum spacing; irregular steps
    scale sigma_j by sqrt(delta_k_j / median(delta_k)). Smoothing is in pixels.
    Anchors are (integer momentum-column index, energy in eV) pairs.
    candidate_mask optionally restricts allowed energy states at every column.
    """
    intensity, energy, momentum = validate_data(intensity, energy, momentum)
    eta_ev = _positive(eta_ev, "eta_ev")
    reference_sigma_ev = _positive(reference_sigma_ev, "reference_sigma_ev")
    anchor_sigma_ev = _positive(anchor_sigma_ev, "anchor_sigma_ev")
    if not np.isfinite(likelihood_floor) or not 0 < likelihood_floor < 1:
        raise ValueError("likelihood_floor must lie strictly between zero and one")
    if not np.isfinite(background_percentile) or not 0 <= background_percentile < 100:
        raise ValueError("background_percentile must lie in [0, 100)")
    smoothing = np.asarray(smoothing, dtype=float)
    if smoothing.shape != (2,) or not np.all(np.isfinite(smoothing)) or np.any(smoothing < 0):
        raise ValueError("smoothing must contain two finite nonnegative pixel sigmas")
    # Preserve truly featureless columns as missing evidence, even if optional
    # smoothing across momentum would otherwise borrow neighboring peaks.
    raw_scale = intensity.max(axis=0)
    observed = np.ptp(intensity, axis=0) > 1e-12 * np.maximum(raw_scale, np.finfo(float).tiny)
    if not np.any(observed):
        raise ValueError("no energy contrast: no band can be inferred from these data")
    # Scaling before filtering avoids overflow and makes intensity-unit changes
    # immaterial. Background removal is an explicit likelihood approximation.
    processed = gaussian_filter(intensity / intensity.max(), smoothing, mode="nearest")
    background = np.percentile(processed, background_percentile, axis=0)
    contrast = np.maximum(processed - background[None, :], 0)
    peak = contrast.max(axis=0)
    contrast /= np.where(peak > 0, peak, 1)[None, :]
    likelihood = likelihood_floor + (1 - likelihood_floor) * contrast
    likelihood[:, ~observed] = 1.0
    likelihood /= likelihood.sum(axis=0, keepdims=True)
    unary = -np.log(likelihood)
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if reference.shape != momentum.shape or not np.all(np.isfinite(reference)):
            raise ValueError("reference must be finite with one energy per momentum point")
        unary += 0.5 * ((energy[:, None] - reference[None, :]) / reference_sigma_ev) ** 2
    for column, value in anchors:
        if isinstance(column, (bool, np.bool_)) or not isinstance(column, (int, np.integer)):
            raise ValueError("anchor column must be an integer")
        if not 0 <= column < momentum.size or not np.isfinite(value):
            raise ValueError("anchor column is out of range or energy is not finite")
        if not energy[0] <= value <= energy[-1]:
            raise ValueError("anchor energy is outside the measured energy range")
        unary[:, column] += 0.5 * ((energy - value) / anchor_sigma_ev) ** 2
    if candidate_mask is not None:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != intensity.shape or not np.all(candidate_mask.any(axis=0)):
            raise ValueError("candidate_mask must match intensity and allow at least one state per column")
        unary = np.where(candidate_mask, unary, np.inf)
    dk = np.diff(momentum)
    sigmas = eta_ev * np.sqrt(dk / np.median(dk))
    indices, optimum, marginals = _chain_inference(unary, energy, sigmas)
    cdf = np.cumsum(marginals, axis=0)
    lower = energy[np.argmax(cdf >= 0.16, axis=0)]
    upper = energy[np.argmax(cdf >= 0.84, axis=0)]
    return ExtractionResult(
        band_energy=energy[indices],
        band_indices=indices,
        likelihood=likelihood,
        marginals=marginals,
        posterior_mean=np.sum(marginals * energy[:, None], axis=0),
        lower_energy=lower,
        upper_energy=upper,
        observed=observed,
        negative_log_posterior=optimum,
    )


def band_search_masks(energy, initial_guess):
    """Partition energy states at the midpoints of ordered approximate bands.

    initial_guess has shape (n_bands, n_momentum). The bands must stay strictly
    ordered from low to high energy; crossing-band identity is not inferred.
    These masks depend only on the supplied guesses, never on simulation truth.
    """
    energy = np.asarray(energy, dtype=float)
    guess = np.asarray(initial_guess, dtype=float)
    if (energy.ndim != 1 or energy.size < 2 or not np.all(np.isfinite(energy))
            or np.any(np.diff(energy) <= 0)):
        raise ValueError("energy must be finite and strictly increasing")
    if guess.ndim != 2 or min(guess.shape) < 1 or not np.all(np.isfinite(guess)):
        raise ValueError("initial_guess must be finite with shape (n_bands, n_momentum)")
    if np.any(np.diff(guess, axis=0) <= 0):
        raise ValueError("initial_guess bands must stay strictly ordered from low to high energy; crossings unsupported")
    if np.any((guess < energy[0]) | (guess > energy[-1])):
        raise ValueError("initial_guess must lie inside the selected energy range")
    boundaries = 0.5 * (guess[:-1] + guess[1:])
    lower = np.concatenate((np.full((1, guess.shape[1]), -np.inf), boundaries))
    upper = np.concatenate((boundaries, np.full((1, guess.shape[1]), np.inf)))
    masks = ((energy[None, :, None] >= lower[:, None, :])
             & (energy[None, :, None] < upper[:, None, :]))
    if not np.all(masks.any(axis=1)):
        raise ValueError("energy grid is too coarse to give every band a candidate at every momentum")
    return masks


def extract_bands(intensity, energy, momentum, *, initial_guess, anchors=(), **kwargs):
    """Extract ordered bands using biased references and disjoint search regions.

    Returns a tuple of ExtractionResult, ordered from low to high energy. Each
    band is an exact chain solution conditional on its reference and midpoint
    region, not an unconstrained joint multiband fit. anchors are zero-based
    (band_index, momentum_column, energy_ev) triples. kwargs go to extract_band.
    """
    intensity, energy, momentum = validate_data(intensity, energy, momentum)
    guess = np.asarray(initial_guess, dtype=float)
    masks = band_search_masks(energy, guess)
    if guess.shape[1] != momentum.size:
        raise ValueError("initial_guess must contain one energy per momentum point for each band")
    band_anchors = [[] for _ in guess]
    for band, column, value in anchors:
        if (isinstance(band, (bool, np.bool_)) or not isinstance(band, (int, np.integer))
                or not 0 <= band < len(guess)):
            raise ValueError("anchor band index is out of range or is not an integer")
        band_anchors[band].append((column, value))
    return tuple(
        extract_band(intensity, energy, momentum, reference=reference,
                     candidate_mask=mask, anchors=selected_anchors, **kwargs)
        for reference, mask, selected_anchors in zip(guess, masks, band_anchors)
    )
