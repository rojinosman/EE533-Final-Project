import os

import pandas as pd

from common_snn import (
    BASE_OUTPUT_DIR,
    QUANT_BITS,
    RESOLUTIONS,
    TIME_STEPS_LIST,
)

from task6_sweep import run_single_experiment


# ============================================================
# TASK 7 SETTINGS
# ============================================================

TASK7_EPOCHS_PER_RUN = 10
TASK7_TARGET_ACC = 75.0

# Search hidden sizes from smaller to larger
TASK7_HIDDEN_SIZES = [32, 64, 128]

TASK7_RESOLUTIONS = RESOLUTIONS
TASK7_TIME_STEPS_LIST = TIME_STEPS_LIST
TASK7_QUANT_BITS = QUANT_BITS


def complexity_score(hidden_size, time_steps, quant_bits, resolution):
    return hidden_size * time_steps * quant_bits * resolution


def run_task7(
    epochs_per_run: int = TASK7_EPOCHS_PER_RUN,
    target_acc: float = TASK7_TARGET_ACC,
):
    out_dir = os.path.join(BASE_OUTPUT_DIR, "task7")
    os.makedirs(out_dir, exist_ok=True)

    candidate_configs = []

    for hidden in TASK7_HIDDEN_SIZES:
        for resolution in TASK7_RESOLUTIONS:
            for time_steps in TASK7_TIME_STEPS_LIST:
                for quant_bits in TASK7_QUANT_BITS:
                    score = complexity_score(
                        hidden_size=hidden,
                        time_steps=time_steps,
                        quant_bits=quant_bits,
                        resolution=resolution,
                    )

                    candidate_configs.append(
                        {
                            "complexity_score": score,
                            "hidden_size": hidden,
                            "resolution": resolution,
                            "time_steps": time_steps,
                            "quant_bits": quant_bits,
                        }
                    )

    candidate_configs = sorted(
        candidate_configs,
        key=lambda x: x["complexity_score"],
    )

    rows = []
    best_valid = None

    for cfg in candidate_configs:
        print()
        print(
            "Task 7 running | "
            f"hidden={cfg['hidden_size']}, "
            f"resolution={cfg['resolution']}, "
            f"time_steps={cfg['time_steps']}, "
            f"quant_bits={cfg['quant_bits']}, "
            f"complexity={cfg['complexity_score']}, "
            f"epochs={epochs_per_run}"
        )

        row = run_single_experiment(
            neuron_type="circuit",
            resolution=cfg["resolution"],
            time_steps=cfg["time_steps"],
            quant_bits=cfg["quant_bits"],
            hidden_size=cfg["hidden_size"],
            epochs=epochs_per_run,
        )

        row["complexity_score"] = cfg["complexity_score"]

        rows.append(row)

        df_partial = pd.DataFrame(rows)
        df_partial.to_csv(
            os.path.join(out_dir, "task7_simplification_results.csv"),
            index=False,
        )

        print(f"Task 7 result | {row}")

        if row["test_acc"] >= target_acc and best_valid is None:
            best_valid = row.copy()

            pd.DataFrame([best_valid]).to_csv(
                os.path.join(out_dir, "task7_best_config.csv"),
                index=False,
            )

            print()
            print("First simplified config above target accuracy:")
            print(best_valid)

            break

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(out_dir, "task7_simplification_results.csv"),
        index=False,
    )

    if best_valid is not None:
        print()
        print("Best simplified config above target:")
        print(best_valid)
    else:
        print()
        print("No config reached the target accuracy.")

    print()
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    run_task7()