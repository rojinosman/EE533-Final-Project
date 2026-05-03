import os

import pandas as pd
import torch

from common_snn import (
    BASE_OUTPUT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LR,
    DEFAULT_TIME_STEPS,
    DEVICE,
    FI_CSV_PATH,
    OneHiddenLayerSNN,
    evaluate,
    get_mnist_loaders,
    train_one_epoch,
)


def run_task4(
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    time_steps: int = DEFAULT_TIME_STEPS,
    epochs: int = DEFAULT_EPOCHS,
):
    out_dir = os.path.join(BASE_OUTPUT_DIR, "task4")
    os.makedirs(out_dir, exist_ok=True)

    train_loader, test_loader = get_mnist_loaders(DEFAULT_BATCH_SIZE)

    model = OneHiddenLayerSNN(
        fi_csv_path=FI_CSV_PATH,
        hidden_size=hidden_size,
        time_steps=time_steps,
        neuron_type="circuit",
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)

    rows = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            resolution=28,
            quant_bits=None,
        )

        test_acc = evaluate(
            model=model,
            loader=test_loader,
            device=DEVICE,
            resolution=28,
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "hidden_size": hidden_size,
            "time_steps": time_steps,
        }

        rows.append(row)

        print(
            f"Task 4 | Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    hist = pd.DataFrame(rows)

    hist.to_csv(
        os.path.join(out_dir, "task4_training_history.csv"),
        index=False,
    )

    torch.save(
        model.state_dict(),
        os.path.join(out_dir, "task4_model.pt"),
    )

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    run_task4()
