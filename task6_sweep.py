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
)


def run_single_experiment(
    neuron_type: str,
    resolution: int,
    time_steps: int,
    quant_bits: int,
    hidden_size: int = 128,
    epochs: int = 1,
):
    train_loader, test_loader = get_mnist_loaders(
        batch_size=DEFAULT_BATCH_SIZE,
        train_limit=10000,
        test_limit=2000,
    )

    model = OneHiddenLayerSNN(
        fi_csv_path=FI_CSV_PATH,
        hidden_size=hidden_size,
        time_steps=time_steps,
        neuron_type=neuron_type,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)

    last_train_loss = None
    last_train_acc = None

    for _ in range(epochs):
        last_train_loss, last_train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            resolution=resolution,
            quant_bits=quant_bits,
        )

    test_acc = evaluate(
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
        "test_acc": test_acc,
    }


def run_task6(epochs_per_run: int = 1):
    out_dir = os.path.join(BASE_OUTPUT_DIR, "task6")
    os.makedirs(out_dir, exist_ok=True)

    rows = []

    for neuron_type in ["default", "circuit"]:
        for resolution in RESOLUTIONS:
            for time_steps in TIME_STEPS_LIST:
                for quant_bits in QUANT_BITS:
                    row = run_single_experiment(
                        neuron_type=neuron_type,
                        resolution=resolution,
                        time_steps=time_steps,
                        quant_bits=quant_bits,
                        hidden_size=128,
                        epochs=epochs_per_run,
                    )

                    rows.append(row)

                    print(f"Task 6 | {row}")

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(out_dir, "task6_results.csv"),
        index=False,
    )

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    run_task6()