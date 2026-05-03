import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_snn import BASE_OUTPUT_DIR


SPIKE_MATCH_TOLERANCE_S = 5e-3


def match_spikes(
    python_spikes,
    cadence_spikes,
    tolerance_s=SPIKE_MATCH_TOLERANCE_S,
):
    python_spikes = np.asarray(python_spikes, dtype=float)
    cadence_spikes = np.asarray(cadence_spikes, dtype=float)

    matched_python = []
    matched_cadence = []
    errors = []

    used = np.zeros(len(cadence_spikes), dtype=bool)

    for p in python_spikes:
        if len(cadence_spikes) == 0:
            break

        diffs = np.abs(cadence_spikes - p)

        diffs[used] = np.inf

        idx = np.argmin(diffs)

        if np.isfinite(diffs[idx]) and diffs[idx] <= tolerance_s:
            used[idx] = True
            matched_python.append(p)
            matched_cadence.append(cadence_spikes[idx])
            errors.append(p - cadence_spikes[idx])

    return (
        np.array(matched_python),
        np.array(matched_cadence),
        np.array(errors),
    )


def compare_neuron(
    python_spike_csv: str,
    cadence_spike_csv: str,
    neuron_index: int,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    python_spikes = pd.read_csv(python_spike_csv)["spike_time_s"].dropna().to_numpy(dtype=float)

    cadence_spikes = pd.read_csv(cadence_spike_csv)["spike_time_s"].dropna().to_numpy(dtype=float)

    matched_python, matched_cadence, errors = match_spikes(
        python_spikes,
        cadence_spikes,
    )

    metrics = {
        "neuron_index": neuron_index,
        "python_spike_count": len(python_spikes),
        "cadence_spike_count": len(cadence_spikes),
        "matched_spike_count": len(errors),
        "mean_abs_timing_error_ms": float(np.mean(np.abs(errors)) * 1e3) if len(errors) else np.nan,
        "rmse_timing_error_ms": float(np.sqrt(np.mean(errors ** 2)) * 1e3) if len(errors) else np.nan,
        "max_abs_timing_error_ms": float(np.max(np.abs(errors)) * 1e3) if len(errors) else np.nan,
    }

    pd.DataFrame([metrics]).to_csv(
        os.path.join(out_dir, f"neuron_{neuron_index}_comparison_metrics.csv"),
        index=False,
    )

    if len(errors):
        pd.DataFrame(
            {
                "python_spike_time_s": matched_python,
                "cadence_spike_time_s": matched_cadence,
                "timing_error_s": errors,
                "timing_error_ms": errors * 1e3,
            }
        ).to_csv(
            os.path.join(out_dir, f"neuron_{neuron_index}_matched_spike_errors.csv"),
            index=False,
        )

    plt.figure(figsize=(10, 2.5))

    plt.eventplot(
        [cadence_spikes, python_spikes],
        orientation="horizontal",
        lineoffsets=[1, 0],
        linelengths=0.8,
    )

    plt.yticks([0, 1], ["Python", "Cadence"])
    plt.xlabel("Time (s)")
    plt.title(f"Task 5 Neuron {neuron_index}: Cadence vs Python Spike Timing")
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"neuron_{neuron_index}_spike_raster_comparison.png"),
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(10, 4))

    if len(errors):
        plt.plot(np.arange(len(errors)), errors * 1e3, marker="o")

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Matched spike index")
    plt.ylabel("Timing error (ms)")
    plt.title(f"Task 5 Neuron {neuron_index}: Timing Error")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"neuron_{neuron_index}_timing_error.png"),
        dpi=200,
    )

    plt.close()

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--python_spike_csv", required=True)
    parser.add_argument("--cadence_spike_csv", required=True)
    parser.add_argument("--neuron_index", type=int, default=0)

    args = parser.parse_args()

    out_dir = os.path.join(BASE_OUTPUT_DIR, "task5_compare")

    metrics = compare_neuron(
        python_spike_csv=args.python_spike_csv,
        cadence_spike_csv=args.cadence_spike_csv,
        neuron_index=args.neuron_index,
        out_dir=out_dir,
    )

    print(metrics)
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()