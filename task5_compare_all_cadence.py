import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PYTHON_SPIKE_DIR = "remaining_tasks_outputs/task5_export"

CADENCE_SPIKE_FILES = {
    0: "/Users/rojindawood/Downloads/cadence_task5_neuron_0_spike_times.csv",
    1: "/Users/rojindawood/Downloads/cadence_task5_neuron_1_spike_times.csv",
    2: "/Users/rojindawood/Downloads/cadence_task5_neuron_2_spike_times.csv",
    3: "/Users/rojindawood/Downloads/cadence_task5_neuron_3_spike_times.csv",
}

OUTPUT_DIR = "remaining_tasks_outputs/task5_compare"

SPIKE_MATCH_TOLERANCE_S = 5e-3  # 5 ms


# ============================================================
# HELPERS
# ============================================================

def load_spike_times(csv_path: str, column_name: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    if column_name not in df.columns:
        raise ValueError(
            f"{csv_path} must contain column '{column_name}'. "
            f"Found columns: {list(df.columns)}"
        )

    return df[column_name].dropna().to_numpy(dtype=float)


def match_spikes(python_spikes, cadence_spikes, tolerance_s):
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


def plot_spike_raster(python_spikes, cadence_spikes, neuron_index, out_path):
    plt.figure(figsize=(10, 2.8))

    plt.eventplot(
        [cadence_spikes, python_spikes],
        orientation="horizontal",
        lineoffsets=[1, 0],
        linelengths=0.8,
        colors=["tab:blue", "tab:orange"],
    )

    plt.yticks([0, 1], ["Python", "Cadence"])
    plt.xlabel("Time (s)")
    plt.title(f"Task 5 Neuron {neuron_index}: Cadence vs Python Spike Timing")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_timing_error(errors_s, neuron_index, out_path):
    plt.figure(figsize=(10, 4))

    if len(errors_s) > 0:
        plt.plot(
            np.arange(len(errors_s)),
            errors_s * 1e3,
            marker="o",
            color="tab:purple",
            linewidth=1.5,
        )

    plt.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    plt.xlabel("Matched spike index")
    plt.ylabel("Timing error (ms)")
    plt.title(f"Task 5 Neuron {neuron_index}: Timing Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compare_one_neuron(neuron_index: int):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    python_csv = os.path.join(
        PYTHON_SPIKE_DIR,
        f"neuron_{neuron_index}_python_spike_times.csv",
    )

    cadence_csv = CADENCE_SPIKE_FILES[neuron_index]

    python_spikes = load_spike_times(
        python_csv,
        column_name="spike_time_s",
    )

    cadence_spikes = load_spike_times(
        cadence_csv,
        column_name="spike_time_s",
    )

    matched_python, matched_cadence, errors = match_spikes(
        python_spikes,
        cadence_spikes,
        SPIKE_MATCH_TOLERANCE_S,
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

    metrics_path = os.path.join(
        OUTPUT_DIR,
        f"neuron_{neuron_index}_comparison_metrics.csv",
    )

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    matched_path = os.path.join(
        OUTPUT_DIR,
        f"neuron_{neuron_index}_matched_spike_errors.csv",
    )

    pd.DataFrame(
        {
            "python_spike_time_s": matched_python,
            "cadence_spike_time_s": matched_cadence,
            "timing_error_s": errors,
            "timing_error_ms": errors * 1e3,
        }
    ).to_csv(matched_path, index=False)

    plot_spike_raster(
        python_spikes=python_spikes,
        cadence_spikes=cadence_spikes,
        neuron_index=neuron_index,
        out_path=os.path.join(
            OUTPUT_DIR,
            f"neuron_{neuron_index}_spike_raster_comparison.png",
        ),
    )

    plot_timing_error(
        errors_s=errors,
        neuron_index=neuron_index,
        out_path=os.path.join(
            OUTPUT_DIR,
            f"neuron_{neuron_index}_timing_error.png",
        ),
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []

    for neuron_index in [0, 1, 2, 3]:
        metrics = compare_one_neuron(neuron_index)
        all_metrics.append(metrics)

        print(
            f"Neuron {neuron_index} | "
            f"Python spikes: {metrics['python_spike_count']} | "
            f"Cadence spikes: {metrics['cadence_spike_count']} | "
            f"Matched: {metrics['matched_spike_count']} | "
            f"Mean abs error: {metrics['mean_abs_timing_error_ms']:.4f} ms"
        )

    summary = pd.DataFrame(all_metrics)

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "task5_all_neuron_summary_metrics.csv"),
        index=False,
    )

    print()
    print("Task 5 comparison complete.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print()
    print(summary)


if __name__ == "__main__":
    main()