import os
import argparse

import numpy as np
import pandas as pd
import torch

from common_snn import (
    BASE_OUTPUT_DIR,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_TIME_STEPS,
    DEVICE,
    DT,
    FI_CSV_PATH,
    OneHiddenLayerSNN,
    get_mnist_loaders,
    preprocess_batch,
)


def export_hidden_neuron_waveforms(
    model: OneHiddenLayerSNN,
    image_tensor: torch.Tensor,
    neuron_indices,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    model.reset_state()

    with torch.no_grad():
        img = preprocess_batch(image_tensor.unsqueeze(0).to(DEVICE), 28)

        _, hidden_spikes, hidden_currents = model(
            img,
            record_hidden=True,
        )

    hidden_currents_np = torch.stack(hidden_currents, dim=0).squeeze(1).numpy()
    hidden_spikes_np = torch.stack(hidden_spikes, dim=0).squeeze(1).numpy()

    time_s = np.arange(hidden_currents_np.shape[0]) * DT

    rows = []

    for idx in neuron_indices:
        current_nA = hidden_currents_np[:, idx]
        spikes = hidden_spikes_np[:, idx]

        pd.DataFrame(
            {
                "time_s": time_s,
                "input_current_nA": current_nA,
            }
        ).to_csv(
            os.path.join(out_dir, f"neuron_{idx}_input_waveform.csv"),
            index=False,
        )

        pd.DataFrame(
            {
                "time_s": time_s,
                "python_spike": spikes,
            }
        ).to_csv(
            os.path.join(out_dir, f"neuron_{idx}_python_spikes_over_time.csv"),
            index=False,
        )

        pd.DataFrame(
            {
                "spike_time_s": time_s[spikes > 0.5],
            }
        ).to_csv(
            os.path.join(out_dir, f"neuron_{idx}_python_spike_times.csv"),
            index=False,
        )

        rows.append(
            {
                "neuron_index": idx,
                "num_python_spikes": int((spikes > 0.5).sum()),
                "waveform_csv": f"neuron_{idx}_input_waveform.csv",
                "python_spike_time_csv": f"neuron_{idx}_python_spike_times.csv",
            }
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        os.path.join(out_dir, "task5_export_summary.csv"),
        index=False,
    )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--time_steps", type=int, default=DEFAULT_TIME_STEPS)
    parser.add_argument("--neuron_indices", type=str, default="0,1,2,3")

    args = parser.parse_args()

    out_dir = os.path.join(BASE_OUTPUT_DIR, "task5_export")
    os.makedirs(out_dir, exist_ok=True)

    model = OneHiddenLayerSNN(
        fi_csv_path=FI_CSV_PATH,
        hidden_size=args.hidden_size,
        time_steps=args.time_steps,
        neuron_type="circuit",
    ).to(DEVICE)

    task4_model_path = os.path.join(
        BASE_OUTPUT_DIR,
        "task4",
        "task4_model.pt",
    )

    if os.path.exists(task4_model_path):
        model.load_state_dict(
            torch.load(task4_model_path, map_location=DEVICE)
        )

        print(f"Loaded trained Task 4 model: {task4_model_path}")

    else:
        print("WARNING: No trained Task 4 model found. Exporting from an untrained model.")

    train_loader, _ = get_mnist_loaders(batch_size=1)

    dataset = train_loader.dataset

    image, label = dataset[args.image_index]

    neuron_indices = [int(x.strip()) for x in args.neuron_indices.split(",")]

    summary = export_hidden_neuron_waveforms(
        model=model,
        image_tensor=image,
        neuron_indices=neuron_indices,
        out_dir=out_dir,
    )

    print(summary)
    print(f"Image label: {label}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()