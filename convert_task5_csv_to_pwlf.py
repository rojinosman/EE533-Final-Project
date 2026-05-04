import os
import pandas as pd


INPUT_DIR = "remaining_tasks_outputs/task5_export"
OUTPUT_DIR = "remaining_tasks_outputs/task5_export/cadence_pwlf"

NEURON_INDICES = [0, 1, 2, 3]


def convert_csv_to_pwlf(input_csv, output_txt):
    df = pd.read_csv(input_csv)

    if "time_s" not in df.columns or "input_current_nA" not in df.columns:
        raise ValueError(
            f"{input_csv} must contain columns: time_s,input_current_nA"
        )

    time_s = df["time_s"].to_numpy(dtype=float)
    current_a = df["input_current_nA"].to_numpy(dtype=float) * 1e-9

    with open(output_txt, "w") as f:
        for t, i in zip(time_s, current_a):
            f.write(f"{t:.9e} {i:.9e}\n")

    print(f"Created: {output_txt}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx in NEURON_INDICES:
        input_csv = os.path.join(
            INPUT_DIR,
            f"neuron_{idx}_input_waveform.csv",
        )

        output_txt = os.path.join(
            OUTPUT_DIR,
            f"neuron_{idx}_input_waveform_pwlf.txt",
        )

        convert_csv_to_pwlf(input_csv, output_txt)


if __name__ == "__main__":
    main()