import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILE PATHS
# ============================================================

FI_CSV_PATH = "/Users/rojindawood/Downloads/inputCurrent_vs_outputFrequency_IF.csv"

# Cadence spike-time CSVs
CADENCE_CONSTANT_SPIKES_CSV = "/Users/rojindawood/Downloads/cadence_constant_spike_times_50nA_20ms.csv"
CADENCE_PULSE_SPIKES_CSV = "/Users/rojindawood/Downloads/cadence_pulse_spike_times_50nA_500us_5ms_100ms.csv"

# Cadence vout waveform CSVs
CADENCE_CONSTANT_VOUT_CSV = "/Users/rojindawood/Downloads/50nA_spike_timing_20ms.csv"
CADENCE_PULSE_VOUT_CSV = "/Users/rojindawood/Downloads/50nA_pulse_spike_timing_100ms.csv"

# Cadence vmem waveform CSVs
CADENCE_CONSTANT_VMEM_CSV = "/Users/rojindawood/Downloads/50nA_vmem_timing.csv"
CADENCE_PULSE_VMEM_CSV = "/Users/rojindawood/Downloads/50nA_pulse_vmem_timing.csv"

OUTPUT_DIR = "task3_outputs"


# ============================================================
# SIMULATION SETTINGS
# ============================================================

DT = 1e-5
TAU_MEM = 5e-3
V_THRESHOLD = 1.0
V_RESET = 0.0
REFRACTORY_STEPS = 0

# Constant-current case
CONST_SIM_TIME_S = 0.020
CONSTANT_CURRENT_NA = 50.0

# Pulse-train case
PULSE_SIM_TIME_S = 0.100
PULSE_BASELINE_NA = 0.0
PULSE_AMPLITUDE_NA = 50.0
PULSE_WIDTH_S = 500e-6
PULSE_PERIOD_S = 5e-3
PULSE_DELAY_S = 0.0

SPIKE_MATCH_TOLERANCE_S = 5e-4


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class SimulationResult:
    input_current_nA: np.ndarray
    predicted_rate_hz: np.ndarray
    achieved_rate_hz: float
    spike_times_s: np.ndarray
    spike_train: np.ndarray
    voltage_trace: np.ndarray
    time_s: np.ndarray


# ============================================================
# LOAD f-I DATA
# ============================================================

class FITable:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"f-I CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        if {"Current (A) X", "Frequency (f) Y"}.issubset(df.columns):
            current_nA = df["Current (A) X"].to_numpy(dtype=float) * 1e9
            firing_rate_hz = df["Frequency (f) Y"].to_numpy(dtype=float)

        elif {"Frequency (f) X", "Frequency (f) Y"}.issubset(df.columns):
            current_nA = df["Frequency (f) X"].to_numpy(dtype=float) * 1e9
            firing_rate_hz = df["Frequency (f) Y"].to_numpy(dtype=float)

        elif {"current_nA", "firing_rate_Hz"}.issubset(df.columns):
            current_nA = df["current_nA"].to_numpy(dtype=float)
            firing_rate_hz = df["firing_rate_Hz"].to_numpy(dtype=float)

        elif {"current_A", "firing_rate_Hz"}.issubset(df.columns):
            current_nA = df["current_A"].to_numpy(dtype=float) * 1e9
            firing_rate_hz = df["firing_rate_Hz"].to_numpy(dtype=float)

        else:
            raise ValueError(
                "CSV columns not recognized.\n"
                f"Found columns: {list(df.columns)}"
            )

        valid = ~(np.isnan(current_nA) | np.isnan(firing_rate_hz))
        current_nA = current_nA[valid]
        firing_rate_hz = firing_rate_hz[valid]

        if len(current_nA) < 2:
            raise ValueError("f-I CSV must contain at least 2 valid rows.")

        order = np.argsort(current_nA)

        self.current_nA = current_nA[order]
        self.firing_rate_hz = firing_rate_hz[order]

        self.min_current_nA = float(np.min(self.current_nA))
        self.max_current_nA = float(np.max(self.current_nA))
        self.max_rate_hz = float(np.max(self.firing_rate_hz))

    def interp_freq(self, current_nA):
        x = np.asarray(current_nA, dtype=float)
        x_clip = np.clip(x, self.current_nA[0], self.current_nA[-1])
        return np.interp(x_clip, self.current_nA, self.firing_rate_hz)


# ============================================================
# PYTHON CIRCUIT-INFORMED NEURON
# ============================================================

class CircuitInformedIFNeuron:
    def __init__(
        self,
        fi_table: FITable,
        dt: float = DT,
        tau_mem: float = TAU_MEM,
        v_threshold: float = V_THRESHOLD,
        v_reset: float = V_RESET,
        refractory_steps: int = REFRACTORY_STEPS,
    ):
        self.fi_table = fi_table
        self.dt = dt
        self.tau_mem = tau_mem
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.refractory_steps = refractory_steps

    def current_to_drive(self, current_nA: float, predicted_rate_hz: float) -> float:
        current_norm = max(current_nA, 0.0) / max(self.fi_table.max_current_nA, 1e-12)
        rate_norm = max(predicted_rate_hz, 0.0) / max(self.fi_table.max_rate_hz, 1e-12)

        drive = 0.25 + 2.8 * current_norm + 1.8 * rate_norm
        return max(drive, 0.0)

    def simulate_waveform(self, input_current_nA):
        input_current_nA = np.asarray(input_current_nA, dtype=float)

        steps = len(input_current_nA)
        time_s = np.arange(steps) * self.dt

        predicted_rate_hz = self.fi_table.interp_freq(np.maximum(input_current_nA, 0.0))

        voltage_trace = np.zeros(steps, dtype=float)
        spike_train = np.zeros(steps, dtype=float)
        spike_times = []

        v = self.v_reset
        refractory_counter = 0
        phase = 0.0

        for t in range(steps):
            current = max(input_current_nA[t], 0.0)
            f_pred = max(predicted_rate_hz[t], 0.0)

            if refractory_counter > 0:
                refractory_counter -= 1
                v = self.v_reset
                voltage_trace[t] = v
                continue

            drive = self.current_to_drive(current, f_pred)

            dv = (-v + drive) * (self.dt / self.tau_mem)
            v += dv

            phase += f_pred * self.dt

            spike = False

            if v >= self.v_threshold:
                spike = True

            elif phase >= 1.0 and current > 0:
                spike = True

            if spike:
                spike_train[t] = 1.0
                spike_times.append(time_s[t])

                v = self.v_reset
                phase -= 1.0

                if phase < 0.0:
                    phase = 0.0

                refractory_counter = self.refractory_steps

            voltage_trace[t] = v

        achieved_rate_hz = float(np.sum(spike_train) / max(steps * self.dt, 1e-12))

        return SimulationResult(
            input_current_nA=input_current_nA,
            predicted_rate_hz=predicted_rate_hz,
            achieved_rate_hz=achieved_rate_hz,
            spike_times_s=np.array(spike_times, dtype=float),
            spike_train=spike_train,
            voltage_trace=voltage_trace,
            time_s=time_s,
        )

    def simulate_constant(self, current_nA: float, sim_time_s: float):
        steps = int(np.round(sim_time_s / self.dt))
        waveform = np.full(steps, current_nA, dtype=float)
        return self.simulate_waveform(waveform)


# ============================================================
# WAVEFORMS
# ============================================================

def make_pulse_train_waveform(
    sim_time_s: float,
    dt: float,
    baseline_nA: float,
    pulse_nA: float,
    pulse_width_s: float,
    pulse_period_s: float,
    pulse_delay_s: float = 0.0,
):
    steps = int(np.round(sim_time_s / dt))
    time_s = np.arange(steps) * dt

    waveform = np.full(steps, baseline_nA, dtype=float)

    active_time = time_s - pulse_delay_s

    pulse_region = active_time >= 0
    phase = np.mod(active_time[pulse_region], pulse_period_s)

    waveform[pulse_region] = np.where(
        phase < pulse_width_s,
        pulse_nA,
        baseline_nA,
    )

    return waveform


# ============================================================
# LOAD CADENCE DATA
# ============================================================

def load_spike_times(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cadence spike-time CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "spike_time_s" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain a column named 'spike_time_s'"
        )

    spikes = df["spike_time_s"].dropna().to_numpy(dtype=float)
    return np.sort(spikes)


def load_cadence_vout(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cadence vout CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "/vout X" not in df.columns or "/vout Y" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain columns '/vout X' and '/vout Y'. "
            f"Found columns: {list(df.columns)}"
        )

    time_s = df["/vout X"].dropna().to_numpy(dtype=float)
    vout = df["/vout Y"].dropna().to_numpy(dtype=float)

    return time_s, vout


def load_cadence_vmem(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cadence vmem CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if "/vmem X" not in df.columns or "/vmem Y" not in df.columns:
        raise ValueError(
            f"{csv_path} must contain columns '/vmem X' and '/vmem Y'. "
            f"Found columns: {list(df.columns)}"
        )

    time_s = df["/vmem X"].dropna().to_numpy(dtype=float)
    vmem = df["/vmem Y"].dropna().to_numpy(dtype=float)

    return time_s, vmem


# ============================================================
# METRICS
# ============================================================

def compute_rate(spike_times_s, sim_time_s):
    return float(len(spike_times_s) / max(sim_time_s, 1e-12))


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
        np.array(matched_python, dtype=float),
        np.array(matched_cadence, dtype=float),
        np.array(errors, dtype=float),
    )


# ============================================================
# PLOTTING
# ============================================================

def make_output_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_input_waveform(result: SimulationResult, out_path: str, title: str):
    plt.figure(figsize=(10, 3))
    plt.plot(
        result.time_s * 1e3,
        result.input_current_nA,
        color="black",
        linewidth=1.5,
        label="Input current",
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Input current (nA)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_python_voltage(result: SimulationResult, out_path: str, title: str):
    plt.figure(figsize=(10, 4))
    plt.plot(
        result.time_s * 1e3,
        result.voltage_trace,
        color="tab:orange",
        linewidth=1.5,
        label="Python membrane voltage",
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Python membrane voltage")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_spike_raster_comparison(
    python_spikes,
    cadence_spikes,
    out_path: str,
    title: str,
):
    plt.figure(figsize=(10, 2.5))

    plt.eventplot(
        [cadence_spikes * 1e3, python_spikes * 1e3],
        orientation="horizontal",
        lineoffsets=[1, 0],
        linelengths=0.8,
        colors=["tab:blue", "tab:orange"],
    )

    plt.yticks([0, 1], ["Python", "Cadence"])
    plt.xlabel("Time (ms)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_spike_timing_error(errors_s, out_path: str, title: str):
    plt.figure(figsize=(10, 4))

    if len(errors_s) > 0:
        plt.plot(
            np.arange(len(errors_s)),
            errors_s * 1e6,
            marker="o",
            color="tab:purple",
            linewidth=1.5,
        )

    plt.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    plt.xlabel("Matched spike index")
    plt.ylabel("Timing error (µs)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_firing_rate_bar(cadence_rate, python_rate, out_path: str, title: str):
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Cadence", "Python"],
        [cadence_rate, python_rate],
        color=["tab:blue", "tab:orange"],
    )
    plt.ylabel("Firing rate (Hz)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_input_and_spikes_overlay(
    result: SimulationResult,
    cadence_spikes,
    out_path: str,
    title: str,
):
    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(
        result.time_s * 1e3,
        result.input_current_nA,
        color="black",
        linewidth=1.5,
        label="Input current",
    )
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Input current (nA)")
    ax1.grid(True)

    ax2 = ax1.twinx()

    if len(result.spike_times_s) > 0:
        ax2.vlines(
            result.spike_times_s * 1e3,
            0.0,
            1.0,
            color="tab:orange",
            linewidth=1.2,
            label="Python spikes",
        )

    if len(cadence_spikes) > 0:
        ax2.vlines(
            cadence_spikes * 1e3,
            1.2,
            2.2,
            color="tab:blue",
            linewidth=1.2,
            label="Cadence spikes",
        )

    ax2.set_ylabel("Spike events")
    ax2.set_yticks([0.5, 1.7])
    ax2.set_yticklabels(["Python", "Cadence"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_vout_overlay(
    result: SimulationResult,
    cadence_time_s,
    cadence_vout,
    out_path: str,
    title: str,
):
    """
    Plot Cadence vout and Python output spikes on one graph.
    Python does not have an analog vout, so Python output is shown as
    scaled spike pulses.
    """
    python_vout = result.spike_train.copy()

    if len(cadence_vout) > 0 and np.max(cadence_vout) > 0:
        python_vout = python_vout * np.max(cadence_vout)

    plt.figure(figsize=(11, 4))

    plt.plot(
        cadence_time_s * 1e3,
        cadence_vout,
        color="tab:blue",
        linewidth=1.5,
        label="Cadence vout",
    )

    plt.step(
        result.time_s * 1e3,
        python_vout,
        where="post",
        color="tab:orange",
        linestyle="--",
        linewidth=1.3,
        label="Python output spikes",
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Output voltage / scaled spikes")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_vmem_overlay(
    result: SimulationResult,
    cadence_vmem_time_s,
    cadence_vmem,
    out_path: str,
    title: str,
):
    """
    Plot Cadence membrane voltage and Python membrane voltage.
    They use different voltage scales, so this uses two y-axes.
    """
    fig, ax1 = plt.subplots(figsize=(11, 4))

    ax1.plot(
        cadence_vmem_time_s * 1e3,
        cadence_vmem,
        color="tab:blue",
        linewidth=1.5,
        label="Cadence vmem",
    )

    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Cadence vmem (V)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)

    ax2 = ax1.twinx()

    ax2.plot(
        result.time_s * 1e3,
        result.voltage_trace,
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label="Python membrane voltage",
    )

    ax2.set_ylabel("Python membrane voltage", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# CASE COMPARISON
# ============================================================

def compare_case(
    case_name: str,
    result: SimulationResult,
    cadence_spikes,
    sim_time_s: float,
    output_dir: str,
    cadence_vout_time_s=None,
    cadence_vout=None,
    cadence_vmem_time_s=None,
    cadence_vmem=None,
):
    case_dir = os.path.join(output_dir, case_name)
    make_output_dir(case_dir)

    python_spikes = result.spike_times_s

    cadence_rate = compute_rate(cadence_spikes, sim_time_s)
    python_rate = compute_rate(python_spikes, sim_time_s)

    matched_python, matched_cadence, errors = match_spikes(
        python_spikes,
        cadence_spikes,
        SPIKE_MATCH_TOLERANCE_S,
    )

    metrics = {
        "case": case_name,
        "cadence_rate_Hz": cadence_rate,
        "python_rate_Hz": python_rate,
        "rate_abs_error_Hz": abs(python_rate - cadence_rate),
        "rate_percent_error": abs(python_rate - cadence_rate) / max(abs(cadence_rate), 1e-12) * 100.0,
        "cadence_spike_count": len(cadence_spikes),
        "python_spike_count": len(python_spikes),
        "matched_spike_count": len(errors),
        "mean_abs_timing_error_us": float(np.mean(np.abs(errors)) * 1e6) if len(errors) > 0 else np.nan,
        "rmse_timing_error_us": float(np.sqrt(np.mean(errors ** 2)) * 1e6) if len(errors) > 0 else np.nan,
        "max_abs_timing_error_us": float(np.max(np.abs(errors)) * 1e6) if len(errors) > 0 else np.nan,
    }

    pd.DataFrame([metrics]).to_csv(
        os.path.join(case_dir, "metrics.csv"),
        index=False,
    )

    pd.DataFrame({"python_spike_time_s": python_spikes}).to_csv(
        os.path.join(case_dir, "python_spike_times.csv"),
        index=False,
    )

    pd.DataFrame({"cadence_spike_time_s": cadence_spikes}).to_csv(
        os.path.join(case_dir, "cadence_spike_times.csv"),
        index=False,
    )

    if len(errors) > 0:
        pd.DataFrame(
            {
                "python_spike_time_s": matched_python,
                "cadence_spike_time_s": matched_cadence,
                "timing_error_s": errors,
                "timing_error_us": errors * 1e6,
            }
        ).to_csv(
            os.path.join(case_dir, "matched_spike_errors.csv"),
            index=False,
        )

    plot_input_waveform(
        result,
        os.path.join(case_dir, "input_waveform.png"),
        f"{case_name}: Input Current",
    )

    plot_python_voltage(
        result,
        os.path.join(case_dir, "python_voltage.png"),
        f"{case_name}: Python Membrane Voltage",
    )

    plot_spike_raster_comparison(
        python_spikes,
        cadence_spikes,
        os.path.join(case_dir, "spike_raster_comparison.png"),
        f"{case_name}: Cadence vs Python Spike Timing",
    )

    plot_spike_timing_error(
        errors,
        os.path.join(case_dir, "spike_timing_error.png"),
        f"{case_name}: Spike Timing Error",
    )

    plot_firing_rate_bar(
        cadence_rate,
        python_rate,
        os.path.join(case_dir, "firing_rate_comparison.png"),
        f"{case_name}: Firing Rate Comparison",
    )

    plot_input_and_spikes_overlay(
        result,
        cadence_spikes,
        os.path.join(case_dir, "input_and_spikes_overlay.png"),
        f"{case_name}: Input and Spike Comparison",
    )

    if cadence_vout_time_s is not None and cadence_vout is not None:
        plot_vout_overlay(
            result=result,
            cadence_time_s=cadence_vout_time_s,
            cadence_vout=cadence_vout,
            out_path=os.path.join(case_dir, "vout_overlay.png"),
            title=f"{case_name}: Cadence vout vs Python output spikes",
        )

    if cadence_vmem_time_s is not None and cadence_vmem is not None:
        plot_vmem_overlay(
            result=result,
            cadence_vmem_time_s=cadence_vmem_time_s,
            cadence_vmem=cadence_vmem,
            out_path=os.path.join(case_dir, "vmem_overlay.png"),
            title=f"{case_name}: Cadence vmem vs Python membrane voltage",
        )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    make_output_dir(OUTPUT_DIR)

    fi_table = FITable(FI_CSV_PATH)

    neuron = CircuitInformedIFNeuron(
        fi_table=fi_table,
        dt=DT,
        tau_mem=TAU_MEM,
        v_threshold=V_THRESHOLD,
        v_reset=V_RESET,
        refractory_steps=REFRACTORY_STEPS,
    )

    # --------------------------------------------------------
    # Constant-current case: 50 nA for 20 ms
    # --------------------------------------------------------

    cadence_const_spikes = load_spike_times(CADENCE_CONSTANT_SPIKES_CSV)

    cadence_const_vout_time_s, cadence_const_vout = load_cadence_vout(
        CADENCE_CONSTANT_VOUT_CSV
    )

    cadence_const_vmem_time_s, cadence_const_vmem = load_cadence_vmem(
        CADENCE_CONSTANT_VMEM_CSV
    )

    const_result = neuron.simulate_constant(
        current_nA=CONSTANT_CURRENT_NA,
        sim_time_s=CONST_SIM_TIME_S,
    )

    const_metrics = compare_case(
        case_name="constant_current_50nA_20ms",
        result=const_result,
        cadence_spikes=cadence_const_spikes,
        sim_time_s=CONST_SIM_TIME_S,
        output_dir=OUTPUT_DIR,
        cadence_vout_time_s=cadence_const_vout_time_s,
        cadence_vout=cadence_const_vout,
        cadence_vmem_time_s=cadence_const_vmem_time_s,
        cadence_vmem=cadence_const_vmem,
    )

    # --------------------------------------------------------
    # Pulsed-current case: 50 nA, 500 us width, 5 ms period
    # --------------------------------------------------------

    pulse_waveform = make_pulse_train_waveform(
        sim_time_s=PULSE_SIM_TIME_S,
        dt=DT,
        baseline_nA=PULSE_BASELINE_NA,
        pulse_nA=PULSE_AMPLITUDE_NA,
        pulse_width_s=PULSE_WIDTH_S,
        pulse_period_s=PULSE_PERIOD_S,
        pulse_delay_s=PULSE_DELAY_S,
    )

    pd.DataFrame(
        {
            "time_s": np.arange(len(pulse_waveform)) * DT,
            "input_current_nA": pulse_waveform,
        }
    ).to_csv(
        os.path.join(OUTPUT_DIR, "pulse_train_input_waveform.csv"),
        index=False,
    )

    cadence_pulse_spikes = load_spike_times(CADENCE_PULSE_SPIKES_CSV)

    cadence_pulse_vout_time_s, cadence_pulse_vout = load_cadence_vout(
        CADENCE_PULSE_VOUT_CSV
    )

    cadence_pulse_vmem_time_s, cadence_pulse_vmem = load_cadence_vmem(
        CADENCE_PULSE_VMEM_CSV
    )

    pulse_result = neuron.simulate_waveform(pulse_waveform)

    pulse_metrics = compare_case(
        case_name="pulse_train_50nA_500us_5ms_100ms",
        result=pulse_result,
        cadence_spikes=cadence_pulse_spikes,
        sim_time_s=PULSE_SIM_TIME_S,
        output_dir=OUTPUT_DIR,
        cadence_vout_time_s=cadence_pulse_vout_time_s,
        cadence_vout=cadence_pulse_vout,
        cadence_vmem_time_s=cadence_pulse_vmem_time_s,
        cadence_vmem=cadence_pulse_vmem,
    )

    summary = pd.DataFrame([const_metrics, pulse_metrics])

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "summary_metrics.csv"),
        index=False,
    )

    print("Done.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print()
    print(summary)


if __name__ == "__main__":
    main()