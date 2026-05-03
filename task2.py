import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

CSV_PATH = "/Users/rojindawood/Downloads/inputCurrent_vs_outputFrequency_IF.csv"
OUTPUT_DIR = "task2_outputs"

# simulation settings
DT = 1e-4                 # 0.1 ms
TAU_MEM = 5e-3            # 5 ms
V_THRESHOLD = 1.0
V_RESET = 0.0
REFRACTORY_STEPS = 0

# how long to simulate each constant-current test
SIM_TIME_S = 1.0

# example currents for voltage/spike plots
EXAMPLE_CURRENTS_NA = [10.0, 20.0, 50.0]


# ============================================================
# DATA STRUCTURES
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
# f-I TABLE
# ============================================================

class FITable:
    """
    Supports these CSV header styles:

    1) Current (A) X, Frequency (f) Y
    2) Frequency (f) X, Frequency (f) Y
    3) current_nA, firing_rate_Hz
    4) current_A, firing_rate_Hz
    """

    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

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
                f"Found columns: {list(df.columns)}\n\n"
                "Accepted formats are:\n"
                "  ['Current (A) X', 'Frequency (f) Y']\n"
                "  ['Frequency (f) X', 'Frequency (f) Y']\n"
                "  ['current_nA', 'firing_rate_Hz']\n"
                "  ['current_A', 'firing_rate_Hz']"
            )

        valid = ~(np.isnan(current_nA) | np.isnan(firing_rate_hz))
        current_nA = current_nA[valid]
        firing_rate_hz = firing_rate_hz[valid]

        if len(current_nA) < 2:
            raise ValueError("CSV must contain at least 2 valid data rows.")

        order = np.argsort(current_nA)
        self.current_nA = current_nA[order]
        self.firing_rate_hz = firing_rate_hz[order]

        if np.any(np.diff(self.current_nA) <= 0):
            raise ValueError("Current values must be strictly increasing after sorting.")

    def interp_freq(self, current_nA):
        x = np.asarray(current_nA, dtype=float)
        x_clip = np.clip(x, self.current_nA[0], self.current_nA[-1])
        return np.interp(x_clip, self.current_nA, self.firing_rate_hz)

    def min_current(self):
        return float(self.current_nA[0])

    def max_current(self):
        return float(self.current_nA[-1])

    def max_rate(self):
        return float(np.max(self.firing_rate_hz))


# ============================================================
# CIRCUIT-INFORMED PYTHON NEURON
# ============================================================

class CircuitInformedIFNeuron:
    """
    Practical Part 2 neuron:
    - integrates membrane voltage
    - spikes at threshold
    - resets after spike
    - uses measured f-I data to calibrate spiking behavior
    """

    def __init__(
        self,
        fi_table: FITable,
        dt: float = 1e-4,
        tau_mem: float = 5e-3,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        refractory_steps: int = 3,
    ):
        self.fi_table = fi_table
        self.dt = dt
        self.tau_mem = tau_mem
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.refractory_steps = refractory_steps

        self.max_current_nA = fi_table.max_current()

    def current_to_drive(self, current_nA: float, predicted_rate_hz: float) -> float:
        current_norm = max(current_nA, 0.0) / max(self.max_current_nA, 1e-12)
        rate_norm = max(predicted_rate_hz, 0.0) / max(self.fi_table.max_rate(), 1e-12)

        # empirical drive to ensure threshold crossing and preserve f-I behavior
        drive = 0.25 + 2.8 * current_norm + 1.8 * rate_norm
        return max(drive, 0.0)

    def simulate_constant(self, current_nA: float, sim_time_s: float) -> SimulationResult:
        steps = int(np.round(sim_time_s / self.dt))
        input_current_nA = np.full(steps, current_nA, dtype=float)
        time_s = np.arange(steps) * self.dt

        predicted_rate_hz = self.fi_table.interp_freq(input_current_nA)

        voltage_trace = np.zeros(steps, dtype=float)
        spike_train = np.zeros(steps, dtype=float)
        spike_times = []

        v = self.v_reset
        refractory_counter = 0
        phase = 0.0

        for t in range(steps):
            current = input_current_nA[t]
            f_pred = predicted_rate_hz[t]

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

        achieved_rate_hz = float(np.sum(spike_train) / max(sim_time_s, 1e-12))

        return SimulationResult(
            input_current_nA=input_current_nA,
            predicted_rate_hz=predicted_rate_hz,
            achieved_rate_hz=achieved_rate_hz,
            spike_times_s=np.array(spike_times, dtype=float),
            spike_train=spike_train,
            voltage_trace=voltage_trace,
            time_s=time_s,
        )


# ============================================================
# METRICS
# ============================================================

def compute_fi_metrics(currents_nA, measured_rates_hz, python_rates_hz):
    abs_err = np.abs(python_rates_hz - measured_rates_hz)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((python_rates_hz - measured_rates_hz) ** 2)))
    denom = np.maximum(np.abs(measured_rates_hz), 1e-12)
    mape = float(np.mean(abs_err / denom) * 100.0)

    return {
        "MAE_Hz": mae,
        "RMSE_Hz": rmse,
        "MAPE_percent": mape,
    }


# ============================================================
# PLOTTING
# ============================================================

def make_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_fi_curve(fi_table: FITable, neuron: CircuitInformedIFNeuron, output_dir: str):
    test_currents = np.linspace(fi_table.min_current(), fi_table.max_current(), 50)
    measured_rates = fi_table.interp_freq(test_currents)

    python_rates = []
    for current in test_currents:
        result = neuron.simulate_constant(current, SIM_TIME_S)
        python_rates.append(result.achieved_rate_hz)
    python_rates = np.array(python_rates)

    plt.figure(figsize=(8, 5))
    plt.plot(fi_table.current_nA, fi_table.firing_rate_hz, "o", label="Measured Cadence points")
    plt.plot(test_currents, measured_rates, "-", label="Interpolated measured f-I")
    plt.plot(test_currents, python_rates, "--", label="Python neuron f-I")
    plt.xlabel("Input current (nA)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("Python f-I Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "python_fi_curve.png"), dpi=200)
    plt.close()

    metrics = compute_fi_metrics(test_currents, measured_rates, python_rates)

    df = pd.DataFrame({
        "current_nA": test_currents,
        "measured_rate_Hz": measured_rates,
        "python_rate_Hz": python_rates,
        "abs_error_Hz": np.abs(python_rates - measured_rates),
        "percent_error": np.abs(python_rates - measured_rates) / np.maximum(np.abs(measured_rates), 1e-12) * 100.0,
    })
    df.to_csv(os.path.join(output_dir, "python_fi_curve_data.csv"), index=False)

    return metrics, df


def plot_voltage_trace(result: SimulationResult, output_path: str, title: str):
    plt.figure(figsize=(10, 4))
    plt.plot(result.time_s, result.voltage_trace)
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane voltage")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_spike_train(result: SimulationResult, output_path: str, title: str):
    plt.figure(figsize=(10, 2))
    plt.eventplot(result.spike_times_s, orientation="horizontal")
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_timeseries_csv(result: SimulationResult, output_path: str):
    pd.DataFrame({
        "time_s": result.time_s,
        "input_current_nA": result.input_current_nA,
        "predicted_rate_hz": result.predicted_rate_hz,
        "voltage": result.voltage_trace,
        "spike": result.spike_train,
    }).to_csv(output_path, index=False)


def save_spike_times_csv(result: SimulationResult, output_path: str):
    pd.DataFrame({
        "spike_time_s": result.spike_times_s
    }).to_csv(output_path, index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    make_output_dir(OUTPUT_DIR)

    fi_table = FITable(CSV_PATH)
    neuron = CircuitInformedIFNeuron(
        fi_table=fi_table,
        dt=DT,
        tau_mem=TAU_MEM,
        v_threshold=V_THRESHOLD,
        v_reset=V_RESET,
        refractory_steps=REFRACTORY_STEPS,
    )

    print("Loaded CSV successfully.")
    print(f"Current range: {fi_table.min_current():.3f} nA to {fi_table.max_current():.3f} nA")
    print(f"Max measured firing rate: {fi_table.max_rate():.3f} Hz")

    # 1) Python f-I curve
    fi_metrics, fi_df = plot_fi_curve(fi_table, neuron, OUTPUT_DIR)

    print("\nPython f-I curve metrics:")
    for k, v in fi_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 2) voltage traces + spike trains for example currents
    summary_rows = []

    for current_nA in EXAMPLE_CURRENTS_NA:
        result = neuron.simulate_constant(current_nA, SIM_TIME_S)

        base = f"{str(current_nA).replace('.', 'p')}nA"

        plot_voltage_trace(
            result,
            os.path.join(OUTPUT_DIR, f"{base}_voltage_trace.png"),
            f"Voltage Trace at {current_nA} nA",
        )

        plot_spike_train(
            result,
            os.path.join(OUTPUT_DIR, f"{base}_spike_train.png"),
            f"Spike Train at {current_nA} nA",
        )

        save_timeseries_csv(
            result,
            os.path.join(OUTPUT_DIR, f"{base}_timeseries.csv"),
        )

        save_spike_times_csv(
            result,
            os.path.join(OUTPUT_DIR, f"{base}_spike_times.csv"),
        )

        measured_rate = float(fi_table.interp_freq([current_nA])[0])
        abs_error = abs(result.achieved_rate_hz - measured_rate)
        percent_error = abs_error / max(abs(measured_rate), 1e-12) * 100.0

        summary_rows.append({
            "current_nA": current_nA,
            "measured_rate_Hz": measured_rate,
            "python_rate_Hz": result.achieved_rate_hz,
            "abs_error_Hz": abs_error,
            "percent_error": percent_error,
            "spike_count": int(np.sum(result.spike_train)),
        })

        print(
            f"\nCurrent = {current_nA:.2f} nA | "
            f"Measured = {measured_rate:.3f} Hz | "
            f"Python = {result.achieved_rate_hz:.3f} Hz | "
            f"Abs Error = {abs_error:.3f} Hz"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "example_current_summary.csv"), index=False)

    print(f"\nAll Part 2 outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()