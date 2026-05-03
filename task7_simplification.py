import os

import pandas as pd

from common_snn import (
    BASE_OUTPUT_DIR,
    HIDDEN_SIZES,
    QUANT_BITS,
    RESOLUTIONS,
    TIME_STEPS_LIST,
)

from task6_sweep import run_single_experiment


def run_task7(
    epochs_per_run: int = 2,
    target_acc: float = 75.0,
):
    out_dir = os.path.join(BASE_OUTPUT_DIR, "task7")
    os.makedirs(out_dir, exist_ok=True)

    candidate_configs = []

    for hidden in HIDDEN_SIZES:
        for time_steps in TIME_STEPS_LIST:
            for quant_bits in QUANT_BITS:
                for resolution in RESOLUTIONS:
                    complexity = hidden * time_steps * quant_bits * resolution

                    candidate_configs.append(
                        (
                            complexity,
                            hidden,
                            time_steps,
                            quant_bits,
                            resolution,
                        )
                    )

    candidate_configs.sort(key=lambda x: x[0])

    rows = []
    best_valid = None

    for complexity, hidden, time_steps, quant_bits, resolution in candidate_configs:
        row = run_single_experiment(
            neuron_type="circuit",
            resolution=resolution,
            time_steps=time_steps,
            quant_bits=quant_bits,
            hidden_size=hidden,
            epochs=epochs_per_run,
        )

        row["complexity_score"] = complexity

        rows.append(row)

        print(f"Task 7 | {row}")

        if row["test_acc"] >= target_acc and best_valid is None:
            best_valid = row.copy()

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(out_dir, "task7_simplification_results.csv"),
        index=False,
    )

    if best_valid is not None:
        pd.DataFrame([best_valid]).to_csv(
            os.path.join(out_dir, "task7_best_config.csv"),
            index=False,
        )

        print("Best config above target:")
        print(best_valid)

    else:
        print("No config reached the target accuracy.")

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    run_task7()