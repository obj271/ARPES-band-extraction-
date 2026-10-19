"""Command-line demo and NPZ extraction, with CSV/NPZ/PNG/JSON outputs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .inference import extract_band, validate_data
from .synthetic import simulate


def _plot(output, data, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k, e = data["momentum"], data["energy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax in axes:
        ax.set(xlabel=r"Momentum ($\mathrm{\AA}^{-1}$)", ylabel="Energy - EF (eV)")
    image = axes[0].pcolormesh(k, e, data["intensity"], shading="auto", cmap="magma")
    fig.colorbar(image, ax=axes[0], label="Linear intensity (a.u.)")
    axes[0].plot(k, result.band_energy, color="cyan", lw=1.6, label="Joint MAP")
    if "truth" in data:
        axes[0].plot(k, data["truth"], "w--", lw=1, label="Simulation truth")
    axes[0].legend(loc="lower left")
    axes[0].set_title("Measured intensity and extracted band")
    image = axes[1].pcolormesh(k, e, result.marginals, shading="auto", cmap="viridis")
    fig.colorbar(image, ax=axes[1], label="Marginal probability per energy bin")
    axes[1].plot(k, result.band_energy, color="red", lw=1, label="Joint MAP")
    axes[1].fill_between(k, result.lower_energy, result.upper_energy, color="white", alpha=0.25,
                         label="Pointwise central 68% interval")
    if np.any(~result.observed):
        axes[1].plot(k[~result.observed], result.band_energy[~result.observed], "x",
                     color="orange", label="No local energy contrast")
    axes[1].set_title("Posterior conditional on the chosen model")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.savefig(output / "extraction.png", dpi=170)
    plt.close(fig)


def _save(output, data, result, config):
    output.mkdir(parents=True, exist_ok=True)
    arrays = dict(data)
    arrays.update(
        band_energy=result.band_energy,
        band_indices=result.band_indices,
        likelihood=result.likelihood,
        marginals=result.marginals,
        posterior_mean=result.posterior_mean,
        lower_energy=result.lower_energy,
        upper_energy=result.upper_energy,
        observed=result.observed,
    )
    np.savez_compressed(output / "result.npz", **arrays)
    np.savetxt(
        output / "band.csv",
        np.column_stack((data["momentum"], result.band_energy, result.posterior_mean,
                         result.lower_energy, result.upper_energy, result.observed)),
        delimiter=",", comments="",
        header="momentum_inv_angstrom,map_energy_ev,posterior_mean_ev,lower16_ev,upper84_ev,observed",
    )
    summary = dict(
        method="Gaussian nearest-neighbor MRF; exact discrete joint MAP by dynamic programming",
        configuration=config,
        shape=list(data["intensity"].shape),
        featureless_columns=int(np.sum(~result.observed)),
        negative_log_posterior_up_to_constants=result.negative_log_posterior,
        interval_note="Pointwise model-conditional marginal intervals; not calibrated experimental errors.",
    )
    if "truth" in data:
        truth = data["truth"]
        map_error = result.band_energy - truth
        baseline = data["energy"][np.argmax(data["intensity"], axis=0)]
        summary["simulation_metrics"] = dict(
            map_rmse_mev=float(1000 * np.sqrt(np.mean(map_error**2))),
            map_observed_rmse_mev=float(1000 * np.sqrt(np.mean(map_error[result.observed] ** 2))),
            raw_argmax_observed_rmse_mev=float(1000 * np.sqrt(np.mean(
                (baseline[result.observed] - truth[result.observed]) ** 2))),
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    _plot(output, data, result)
    return summary


def _anchor(text):
    try:
        column, energy = text.split(":")
        return int(column), float(energy)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("anchor must be COLUMN:ENERGY_EV, e.g. 20:-0.25") from exc


def _parser():
    parser = argparse.ArgumentParser(description="ARPES Gaussian-prior + MAP band extraction")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("demo", "Generate and extract a noisy synthetic band"),
                            ("extract", "Extract from an NPZ containing intensity, energy, momentum")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--output", type=Path, default=Path("outputs") / name)
        sub.add_argument("--eta", type=float, default=0.012, help="Neighbor Gaussian sigma in eV")
        sub.add_argument("--smooth-energy", type=float, default=0.7, help="Energy smoothing sigma in pixels")
        sub.add_argument("--smooth-momentum", type=float, default=0.0, help="Momentum smoothing sigma in pixels")
        sub.add_argument("--background-percentile", type=float, default=10)
        sub.add_argument("--likelihood-floor", type=float, default=1e-3)
        sub.add_argument("--anchor", type=_anchor, action="append", default=[], help="Repeatable COLUMN:ENERGY_EV")
        sub.add_argument("--anchor-sigma", type=float, default=0.008, help="Soft anchor sigma in eV")
        sub.add_argument("--reference", type=Path, help="NPZ containing band_energy on the same momentum axis")
        sub.add_argument("--reference-sigma", type=float, default=0.04)
        if name == "demo":
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--peak-counts", type=float, default=12)
            sub.add_argument("--energy-points", type=int, default=192)
            sub.add_argument("--momentum-points", type=int, default=100)
            sub.add_argument("--no-dropout", action="store_true")
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
    return dict(intensity=intensity, energy=energy, momentum=momentum)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            data = simulate(seed=args.seed, ne=args.energy_points, nk=args.momentum_points,
                            peak_counts=args.peak_counts, missing=not args.no_dropout)
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
        reference = None
        if args.reference:
            with np.load(args.reference, allow_pickle=False) as archive:
                if "momentum" not in archive or "band_energy" not in archive:
                    raise ValueError("reference NPZ must contain momentum and band_energy")
                if (archive["momentum"].shape != data["momentum"].shape or
                        not np.allclose(archive["momentum"], data["momentum"], rtol=1e-7, atol=1e-10)):
                    raise ValueError("reference momentum axis must match input; interpolate it first")
                reference = archive["band_energy"]
        config = dict(eta_ev=args.eta, smoothing=(args.smooth_energy, args.smooth_momentum),
                      background_percentile=args.background_percentile, likelihood_floor=args.likelihood_floor,
                      anchors=args.anchor, anchor_sigma_ev=args.anchor_sigma, reference_sigma_ev=args.reference_sigma)
        result = extract_band(data["intensity"], data["energy"], data["momentum"], reference=reference, **config)
        if reference is not None:
            data["reference_energy"] = reference
        config.update(command=args.command, reference=str(args.reference) if args.reference else None)
        if args.command == "demo":
            config.update(seed=args.seed, peak_counts=args.peak_counts, missing=not args.no_dropout)
        else:
            config.update(input=str(args.input), energy_window=args.energy_window)
        summary = _save(args.output, data, result, config)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Saved extraction to {args.output.resolve()}")
    if "simulation_metrics" in summary:
        print(json.dumps(summary["simulation_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
