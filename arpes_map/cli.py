"""Command-line demo and NPZ extraction, with CSV/NPZ/PNG/JSON outputs."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from .enhancement import comparison_intensities
from .inference import band_search_masks, extract_band, extract_bands, validate_data
from .synthetic import simulate


def _stack(results, field):
    return np.stack([getattr(result, field) for result in results])


def _compact(values):
    """Retain the original single-band NPZ array shapes."""
    return values[0] if len(values) == 1 else values


def _rmse(values, truth, observed=None):
    difference = values - truth
    if observed is not None:
        difference = difference[..., observed]
    return float(1000 * np.sqrt(np.mean(difference**2)))


def _plot(output, data, results, comparisons, comparison_bands, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k, e = data["momentum"], data["energy"]
    bands = _stack(results, "band_energy")
    guesses = np.atleast_2d(data["initial_guess"]) if "initial_guess" in data else None
    truth = np.atleast_2d(data["truth"]) if "truth" in data else None
    colors = plt.get_cmap("tab10")(np.arange(len(results)))

    def curves(ax, values, show_guess=False):
        for b, curve in enumerate(values):
            ax.plot(k, curve, color=colors[b], lw=1.5, label=f"Band {b + 1}")
            if show_guess and guesses is not None:
                ax.plot(k, guesses[b], color=colors[b], ls=":", lw=1.6,
                        label="Initial guess" if b == 0 else None)
            if truth is not None:
                ax.plot(k, truth[b], "w--", lw=1.1, label="Truth" if b == 0 else None)
        ax.legend(loc="lower left", fontsize=8)

    # One intensity panel plus one posterior panel for each band.
    if len(results) == 3:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True,
                                 sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(6 * (len(results) + 1), 5.5),
                                 constrained_layout=True, sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    image = axes[0].pcolormesh(k, e, data["intensity"], shading="auto", cmap="magma")
    fig.colorbar(image, ax=axes[0], label="Measured intensity (a.u.)")
    curves(axes[0], bands, show_guess=True)
    axes[0].set_title("MAP and approximate initial dispersions")
    vmax = max(result.marginals.max() for result in results)
    for b, (ax, result) in enumerate(zip(axes[1:], results)):
        image = ax.pcolormesh(k, e, result.marginals, shading="auto", cmap="viridis", vmin=0, vmax=vmax)
        fig.colorbar(image, ax=ax, label="Probability per energy state")
        ax.plot(k, bands[b], color="red", lw=1.2, label="Joint MAP")
        if guesses is not None:
            ax.plot(k, guesses[b], color="cyan", ls=":", lw=1.4, label="Initial guess")
        if truth is not None:
            ax.plot(k, truth[b], "w--", lw=1, label="Truth")
        ax.fill_between(k, result.lower_energy, result.upper_energy, color="white", alpha=0.25,
                        label="Pointwise central 68%")
        if np.any(~result.observed):
            ax.plot(k[~result.observed], bands[b, ~result.observed], "x", color="orange", label="No local contrast")
        ax.set_title(f"Band {b + 1}: conditional marginal posterior")
        ax.legend(loc="lower left", fontsize=8)
    for ax in axes:
        ax.set(xlabel=r"Momentum ($\mathrm{\AA}^{-1}$)", ylabel="Energy - EF (eV)")
    fig.savefig(output / "map_result.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True, sharex=True, sharey=True)
    methods = (
        ("map", "Gaussian-prior MAP", data["intensity"], bands),
        ("raw_argmax", "Raw intensity: column maximum", data["intensity"], comparison_bands["raw_band_energy"]),
        ("curvature", "2D curvature: column maximum", comparisons["curvature_intensity"],
         comparison_bands["curvature_band_energy"]),
        ("gradient", "Minimum gradient: column maximum", comparisons["gradient_intensity"],
         comparison_bands["gradient_band_energy"]),
    )
    for ax, (name, title, values, curves_energy) in zip(axes.ravel(), methods):
        image = ax.pcolormesh(k, e, values, shading="auto", cmap="magma")
        fig.colorbar(image, ax=ax, label="Intensity (a.u.)")
        curves(ax, curves_energy, show_guess=name == "map")
        if metrics:
            title += f"\nObserved RMSE: {metrics[name + '_observed_rmse_mev']:.2f} meV"
        ax.set(xlabel=r"Momentum ($\mathrm{\AA}^{-1}$)", ylabel="Energy - EF (eV)", title=title)
    caption = "Same per-band search regions for all methods; no continuity prior in column maxima"
    if metrics and "initial_guess_rmse_mev" in metrics:
        caption += f"\nInitial-guess RMSE: {metrics['initial_guess_rmse_mev']:.2f} meV (all columns)"
    fig.suptitle(caption, fontsize=12)
    fig.savefig(output / "band_comparison.png", dpi=180)
    plt.close(fig)


def _save(output, data, results, config):
    output.mkdir(parents=True, exist_ok=True)
    energy, momentum = data["energy"], data["momentum"]
    observed = results[0].observed
    if "initial_guess" in data:
        masks = band_search_masks(energy, np.atleast_2d(data["initial_guess"]))
    else:
        masks = np.ones((1, *data["intensity"].shape), dtype=bool)
    comparisons = comparison_intensities(data["intensity"], energy, momentum, smoothing=config["smoothing"])
    comparison_bands = {}
    for name, values in (("raw", data["intensity"]), ("curvature", comparisons["curvature_intensity"]),
                         ("gradient", comparisons["gradient_intensity"])):
        indices = np.argmax(np.where(masks, values[None, :, :], -np.inf), axis=1)
        comparison_bands[name + "_band_energy"] = np.where(observed[None, :], energy[indices], np.nan)
    arrays = dict(data)
    arrays.update(comparisons)
    arrays.update({key: _compact(values) for key, values in comparison_bands.items()})
    for field in ("band_energy", "band_indices", "likelihood", "marginals", "posterior_mean",
                  "lower_energy", "upper_energy"):
        arrays[field] = _compact(_stack(results, field))
    arrays.update(observed=observed, search_masks=_compact(masks))
    np.savez_compressed(output / "result.npz", **arrays)

    rows = []
    header = "momentum_inv_angstrom,map_energy_ev,posterior_mean_ev,lower16_ev,upper84_ev,observed"
    for b, result in enumerate(results):
        row = np.column_stack((momentum, result.band_energy, result.posterior_mean,
                               result.lower_energy, result.upper_energy, observed))
        if len(results) > 1:
            row = np.column_stack((np.full(len(momentum), b + 1), row))
        rows.append(row)
    if len(results) > 1:
        header = "band," + header
    np.savetxt(output / "band.csv", np.concatenate(rows), delimiter=",", comments="", header=header)

    summary = dict(
        method="Gaussian nearest-neighbor MRF; exact discrete MAP within each band's search region",
        configuration=config,
        band_count=len(results),
        shape=list(data["intensity"].shape),
        featureless_columns=int(np.sum(~observed)),
        negative_log_posterior_up_to_constants=_compact(np.array([r.negative_log_posterior for r in results])).tolist(),
        interval_note="Pointwise model-conditional marginal intervals; not calibrated experimental errors.",
        initial_guess_role="Gaussian reference prior and midpoint search partition; not an iterative optimizer seed",
        multiband_note="Ordered noncrossing bands; independent exact chains conditional on reference regions; no automatic band count",
        truth_role="Evaluation and plotting only; never passed to inference or used to define search regions",
        intensity_comparison=dict(
            curvature="Positive part of negative 2D curvature; Cx=Cy=1",
            gradient="Smoothed intensity / RMS of eight neighbor directional differences",
            coordinates="Input axes divided by their median spacing (pixel-equivalent units)",
            smoothing_pixels=list(config["smoothing"]),
            gradient_floor_relative=1e-3,
            normalization="Each enhanced map globally peak-normalized to [0, 1]; featureless columns zeroed",
            band_extraction="Column maximum within the SAME midpoint search regions used for MAP; no continuity prior",
        ),
    )
    metrics = {}
    if "truth" in data:
        truth = np.atleast_2d(data["truth"])
        methods = dict(map=_stack(results, "band_energy"), raw_argmax=comparison_bands["raw_band_energy"],
                       curvature=comparison_bands["curvature_band_energy"],
                       gradient=comparison_bands["gradient_band_energy"])
        if "initial_guess" in data:
            methods = dict(initial_guess=np.atleast_2d(data["initial_guess"]), **methods)
        per_band = [dict(band=b + 1) for b in range(len(results))]
        metric_rows = []
        for name, values in methods.items():
            metrics[name + "_observed_rmse_mev"] = _rmse(values, truth, observed)
            # Peak methods do not fill missing columns: evaluate their observed points only.
            all_rmse = None if np.any(~np.isfinite(values)) else _rmse(values, truth)
            metrics[name + "_rmse_mev"] = all_rmse
            for b in range(len(results)):
                obs_rmse = _rmse(values[b], truth[b], observed)
                band_rmse = None if np.any(~np.isfinite(values[b])) else _rmse(values[b], truth[b])
                per_band[b][name + "_rmse_mev"] = band_rmse
                per_band[b][name + "_observed_rmse_mev"] = obs_rmse
                metric_rows.append(dict(band=b + 1, method=name, all_columns_rmse_mev=band_rmse,
                                        observed_columns_rmse_mev=obs_rmse))
        summary.update(simulation_metrics=metrics, per_band_metrics=per_band)
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("band", "method", "all_columns_rmse_mev",
                                                       "observed_columns_rmse_mev"))
            writer.writeheader()
            writer.writerows(metric_rows)
    elif (output / "metrics.csv").exists():
        (output / "metrics.csv").unlink()
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    _plot(output, data, results, comparisons, comparison_bands, metrics)
    return summary


def _anchor(text):
    try:
        parts = text.split(":")
        if len(parts) == 2:
            return int(parts[0]), float(parts[1])
        if len(parts) == 3:
            return int(parts[0]), int(parts[1]), float(parts[2])
        raise ValueError("wrong number of anchor components")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("anchor must be COLUMN:ENERGY_EV or BAND:COLUMN:ENERGY_EV (zero-based)") from exc


def _parser():
    parser = argparse.ArgumentParser(description="ARPES Gaussian-prior + MAP band extraction")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("demo", "Generate and extract noisy synthetic bands with biased initial guesses"),
                            ("extract", "Extract from an NPZ containing intensity, energy, momentum")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--output", type=Path, default=Path("outputs"))
        sub.add_argument("--eta", type=float, default=0.012, help="Neighbor Gaussian sigma in eV")
        sub.add_argument("--smooth-energy", type=float, default=0.7, help="Energy smoothing sigma in pixels")
        sub.add_argument("--smooth-momentum", type=float, default=0.0, help="Momentum smoothing sigma in pixels")
        sub.add_argument("--background-percentile", type=float, default=10)
        sub.add_argument("--likelihood-floor", type=float, default=1e-3)
        sub.add_argument("--anchor", type=_anchor, action="append", default=[],
                         help="Repeatable COLUMN:ENERGY_EV (single band) or BAND:COLUMN:ENERGY_EV; zero-based")
        sub.add_argument("--anchor-sigma", type=float, default=0.008, help="Soft anchor sigma in eV")
        sub.add_argument("--reference", "--initial-guess", dest="reference", type=Path,
                         help="NPZ with momentum and initial_guess or band_energy, shaped (nk,) or (bands, nk)")
        sub.add_argument("--reference-sigma", type=float, default=0.04)
        if name == "demo":
            sub.add_argument("--bands", type=int, choices=(1, 2, 3), default=3,
                             help="Number of noncrossing simulated bands (default: 3)")
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--peak-counts", type=float, default=12)
            sub.add_argument("--gamma-ev", type=float, default=0.009,
                             help="Intrinsic Lorentzian half width at half maximum in eV")
            sub.add_argument("--energy-points", type=int, default=192)
            sub.add_argument("--momentum-points", type=int, default=100)
            dropout = sub.add_mutually_exclusive_group()
            dropout.add_argument("--dropout", action="store_true",
                                 help="Simulate four missing detector columns near k=0 (default: no dropout)")
            dropout.add_argument("--no-dropout", dest="dropout", action="store_false",
                                 help="Generate complete data (the default)")
            sub.set_defaults(dropout=False)
        else:
            sub.add_argument("--input", type=Path, required=True)
            sub.add_argument("--energy-window", type=float, nargs=2, metavar=("MIN_EV", "MAX_EV"))
    return parser


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        required = ("intensity", "energy", "momentum")
        if any(key not in archive for key in required):
            raise ValueError("input NPZ must contain intensity, energy, momentum")
        intensity, energy, momentum = validate_data(*(archive[key] for key in required))
        data = dict(intensity=intensity, energy=energy, momentum=momentum)
        for key in ("truth", "initial_guess", "clean"):
            if key in archive:
                data[key] = archive[key]
        if "initial_guess" not in data and "reference_energy" in archive:
            data["initial_guess"] = archive["reference_energy"]
    return data


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            data = simulate(seed=args.seed, ne=args.energy_points, nk=args.momentum_points,
                            peak_counts=args.peak_counts, missing=args.dropout, gamma_ev=args.gamma_ev,
                            bands=args.bands)
        else:
            data = _load(args.input)
            if args.energy_window:
                low, high = args.energy_window
                if not np.isfinite([low, high]).all() or low >= high:
                    raise ValueError("energy-window requires finite MIN_EV < MAX_EV")
                select = (data["energy"] >= low) & (data["energy"] <= high)
                if np.sum(select) < 2:
                    raise ValueError("energy-window must include at least two energy samples")
                data["energy"] = data["energy"][select]
                data["intensity"] = data["intensity"][select]
                if "clean" in data:
                    data["clean"] = data["clean"][select]
        reference = data.get("initial_guess")
        if args.reference:
            with np.load(args.reference, allow_pickle=False) as archive:
                key = "initial_guess" if "initial_guess" in archive else "band_energy"
                if "momentum" not in archive or key not in archive:
                    raise ValueError("reference NPZ must contain momentum and initial_guess or band_energy")
                if (archive["momentum"].shape != data["momentum"].shape or
                        not np.allclose(archive["momentum"], data["momentum"], rtol=1e-7, atol=1e-10)):
                    raise ValueError("reference momentum axis must match input; interpolate it first")
                reference = archive[key]
        config = dict(eta_ev=args.eta, smoothing=(args.smooth_energy, args.smooth_momentum),
                      background_percentile=args.background_percentile, likelihood_floor=args.likelihood_floor,
                      anchor_sigma_ev=args.anchor_sigma, reference_sigma_ev=args.reference_sigma)
        if reference is None:
            if any(len(anchor) != 2 for anchor in args.anchor):
                raise ValueError("without initial guesses, use single-band anchors COLUMN:ENERGY_EV")
            results = (extract_band(data["intensity"], data["energy"], data["momentum"],
                                    anchors=args.anchor, **config),)
        else:
            reference = np.atleast_2d(reference)
            anchors = []
            for anchor in args.anchor:
                if len(anchor) == 2:
                    if len(reference) > 1:
                        raise ValueError("multiband anchors require BAND:COLUMN:ENERGY_EV (zero-based)")
                    anchor = (0, *anchor)
                anchors.append(anchor)
            results = extract_bands(data["intensity"], data["energy"], data["momentum"],
                                    initial_guess=reference, anchors=anchors, **config)
            data["initial_guess"] = _compact(reference)
            data["reference_energy"] = _compact(reference)
        if "truth" in data:
            truth = np.asarray(data["truth"])
            if np.atleast_2d(truth).shape != (len(results), len(data["momentum"])) or not np.all(np.isfinite(truth)):
                raise ValueError("truth must be finite and match the extracted band and momentum counts")
        if reference is not None:
            guess_source = str(args.reference) if args.reference else (
                "independent approximate analytic dispersion" if args.command == "demo" else "input NPZ initial_guess")
        else:
            guess_source = None
        config.update(command=args.command, reference=guess_source, anchors=args.anchor, bands=len(results))
        if args.command == "demo":
            config.update(seed=args.seed, peak_counts=args.peak_counts, missing=args.dropout,
                          gamma_ev=args.gamma_ev)
        else:
            config.update(input=str(args.input), energy_window=args.energy_window)
        summary = _save(args.output, data, results, config)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Saved extraction to {args.output.resolve()}")
    if "simulation_metrics" in summary:
        print(json.dumps(summary["simulation_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
