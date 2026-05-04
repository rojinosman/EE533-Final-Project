import os

import pandas as pd
import torch

from common_snn import (
    BASE_OUTPUT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEVICE,
    FI_CSV_PATH,
    QUANT_BITS,
    RESOLUTIONS,
    TIME_STEPS_LIST,
    OneHiddenLayerSNN,
    evaluate,
    get_mnist_loaders,
    train_one_epoch,
    maybe_quantize_weights,
)


# ============================================================
# TASK 6 SETTINGS
# ============================================================

TASK6_EPOCHS_PER_RUN = 3
TASK6_HIDDEN_SIZE = 128

# Smaller subset keeps the sweep reasonable.
# Increase these for more accurate but slower results.
TASK6_TRAIN_LIMIT = 10000
TASK6_TEST_LIMIT = 2000


def run_single_experiment(
    neuron_type: str,
    resolution: int,
    time_steps: int,
    quant_bits: int,
    hidden_size: int = TASK6_HIDDEN_SIZE,
    epochs: int = TASK6_EPOCHS_PER_RUN,
):
    train_loader, test_loader = get_mnist_loaders(
        batch_size=DEFAULT_BATCH_SIZE,
        train_limit=TASK6_TRAIN_LIMIT,
        test_limit=TASK6_TEST_LIMIT,
    )

    model = OneHiddenLayerSNN(
        fi_csv_path=FI_CSV_PATH,
        hidden_size=hidden_size,
        time_steps=time_steps,
        neuron_type=neuron_type,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=DEFAULT_LR,
    )

    last_train_loss = None
    last_train_acc = None

    # --------------------------------------------------------
    # Train without quantization first
    # --------------------------------------------------------
    for epoch in range(epochs):
        last_train_loss, last_train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            resolution=resolution,
            quant_bits=None,
        )

        print(
            f"    Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {last_train_loss:.4f} | "
            f"Train Acc: {last_train_acc:.2f}%"
        )

    # --------------------------------------------------------
    # Evaluate before quantization
    # --------------------------------------------------------
    test_acc_before_quant = evaluate(
        model=model,
        loader=test_loader,
        device=DEVICE,
        resolution=resolution,
    )

    # --------------------------------------------------------
    # Apply quantization once after training
    # --------------------------------------------------------
    maybe_quantize_weights(model, quant_bits)

    # --------------------------------------------------------
    # Evaluate after quantization
    # --------------------------------------------------------
    test_acc_after_quant = evaluate(
        model=model,
        loader=test_loader,
        device=DEVICE,
        resolution=resolution,
    )

    return {
        "neuron_type": neuron_type,
        "resolution": resolution,
        "time_steps": time_steps,
        "quant_bits": quant_bits,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "train_loss": last_train_loss,
        "train_acc": last_train_acc,
        "test_acc_before_quant": test_acc_before_quant,
        "test_acc": test_acc_after_quant,
        "quantization_drop": test_acc_before_quant - test_acc_after_quant,
    }


def run_task6(epochs_per_run: int = TASK6_EPOCHS_PER_RUN):
    out_dir = os.path.join(BASE_OUTPUT_DIR, "task6")
    os.makedirs(out_dir, exist_ok=True)

    rows = []

    for neuron_type in ["default", "circuit"]:
        for resolution in RESOLUTIONS:
            for time_steps in TIME_STEPS_LIST:
                for quant_bits in QUANT_BITS:
                    print()
                    print(
                        f"Task 6 running | "
                        f"neuron_type={neuron_type}, "
                        f"resolution={resolution}, "
                        f"time_steps={time_steps}, "
                        f"quant_bits={quant_bits}, "
                        f"epochs={epochs_per_run}"
                    )

                    row = run_single_experiment(
                        neuron_type=neuron_type,
                        resolution=resolution,
                        time_steps=time_steps,
                        quant_bits=quant_bits,
                        hidden_size=TASK6_HIDDEN_SIZE,
                        epochs=epochs_per_run,
                    )

                    rows.append(row)

                    print(f"Task 6 result | {row}")

                    # Save after every run so progress is not lost if interrupted.
                    df_partial = pd.DataFrame(rows)
                    df_partial.to_csv(
                        os.path.join(out_dir, "task6_results.csv"),
                        index=False,
                    )

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(out_dir, "task6_results.csv"),
        index=False,
    )

    print()
    print(f"Saved outputs to: {out_dir}")
    print()
    print(df)


if __name__ == "__main__":
    run_task6()