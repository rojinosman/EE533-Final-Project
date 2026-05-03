import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# USER SETTINGS
# ============================================================

CSV_PATH = "/Users/rojindawood/Downloads/inputCurrent_vs_outputFrequency_IF.csv"

OUTPUT_DIR = "task2_snn_outputs"

BATCH_SIZE = 64
HIDDEN_SIZE = 128
TIME_STEPS = 10
EPOCHS = 3
LEARNING_RATE = 1e-3

DT = 1e-4
TAU_MEM = 5e-3
V_THRESHOLD = 1.0
V_RESET = 0.0

# Set to 0 because your Cadence f-I curve matched better without refractory delay
REFRACTORY_STEPS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# f-I TABLE
# ============================================================

class FITable:
    """
    Loads the Cadence-measured input-current vs firing-rate data.

    Supports your current CSV format:
        Frequency (f) X, Frequency (f) Y

    where:
        Frequency (f) X = input current in amps
        Frequency (f) Y = firing frequency in Hz
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
                "Accepted formats:\n"
                "  ['Frequency (f) X', 'Frequency (f) Y']\n"
                "  ['Current (A) X', 'Frequency (f) Y']\n"
                "  ['current_nA', 'firing_rate_Hz']\n"
                "  ['current_A', 'firing_rate_Hz']"
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

    def interp_freq_np(self, current_nA_np: np.ndarray) -> np.ndarray:
        x = np.asarray(current_nA_np, dtype=float)
        x_clip = np.clip(x, self.current_nA[0], self.current_nA[-1])
        return np.interp(x_clip, self.current_nA, self.firing_rate_hz)


# ============================================================
# SURROGATE SPIKE FUNCTION
# ============================================================

class SurrogateSpike(torch.autograd.Function):
    """
    Forward pass:
        hard spike: 1 if v >= threshold, else 0

    Backward pass:
        smooth sigmoid-based surrogate gradient
    """

    @staticmethod
    def forward(ctx, v_minus_threshold: torch.Tensor):
        ctx.save_for_backward(v_minus_threshold)
        return (v_minus_threshold >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v_minus_threshold,) = ctx.saved_tensors

        slope = 10.0
        sig = torch.sigmoid(slope * v_minus_threshold)
        surrogate_grad = slope * sig * (1.0 - sig)

        return grad_output * surrogate_grad


def spike_fn(x: torch.Tensor) -> torch.Tensor:
    return SurrogateSpike.apply(x)


# ============================================================
# CIRCUIT-INFORMED NEURON
# ============================================================

class CircuitInformedNeuron(nn.Module):
    """
    Circuit-informed spiking neuron.

    It uses:
    - measured Cadence f-I data
    - membrane integration
    - threshold spiking
    - reset
    - optional refractory behavior

    The network activation is mapped into the measured current range
    of the Cadence neuron, approximately 1 nA to 100 nA.
    """

    def __init__(
        self,
        fi_table: FITable,
        dt: float = DT,
        tau_mem: float = TAU_MEM,
        v_threshold: float = V_THRESHOLD,
        v_reset: float = V_RESET,
        refractory_steps: int = REFRACTORY_STEPS,
    ):
        super().__init__()

        self.fi_table = fi_table
        self.dt = dt
        self.tau_mem = tau_mem
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.refractory_steps = refractory_steps

        self.v = None
        self.refrac = None
        self.phase = None

    def reset_state(self):
        self.v = None
        self.refrac = None
        self.phase = None

    def activation_to_current(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps arbitrary network activation to the Cadence current range.

        The Cadence f-I data is measured from about 1 nA to 100 nA.
        This maps the activation smoothly into that range.
        """

        min_i = self.fi_table.min_current_nA
        max_i = self.fi_table.max_current_nA

        current_nA = min_i + (max_i - min_i) * torch.sigmoid(x)

        return current_nA

    def current_to_drive(self, current_nA: torch.Tensor):
        """
        Uses the measured f-I curve to create a membrane drive and
        predicted firing rate.
        """

        current_np = current_nA.detach().cpu().numpy()

        predicted_rate_np = self.fi_table.interp_freq_np(current_np)

        predicted_rate = torch.tensor(
            predicted_rate_np,
            dtype=current_nA.dtype,
            device=current_nA.device,
        )

        current_norm = current_nA / max(self.fi_table.max_current_nA, 1e-12)
        rate_norm = predicted_rate / max(self.fi_table.max_rate_hz, 1e-12)

        drive = 0.25 + 2.8 * current_norm + 1.8 * rate_norm

        return drive, predicted_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(x)
            self.refrac = torch.zeros_like(x)
            self.phase = torch.zeros_like(x)

        current_nA = self.activation_to_current(x)
        drive, predicted_rate = self.current_to_drive(current_nA)

        active = (self.refrac <= 0).float()

        # Membrane integration
        dv = (-self.v + drive) * (self.dt / self.tau_mem)
        self.v = self.v + dv * active

        # Phase accumulator helps preserve the measured f-I behavior
        self.phase = self.phase + predicted_rate * self.dt * active

        threshold_spike = spike_fn(self.v - self.v_threshold)
        phase_spike = spike_fn(self.phase - 1.0)

        spike = torch.clamp(threshold_spike + phase_spike, max=1.0)

        # Reset
        self.v = self.v * (1.0 - spike) + self.v_reset * spike

        # Reduce phase after spike
        self.phase = torch.clamp(self.phase - spike, min=0.0)

        # Refractory update
        if self.refractory_steps > 0:
            self.refrac = torch.clamp(self.refrac - 1, min=0) + spike * self.refractory_steps
        else:
            self.refrac = torch.zeros_like(self.refrac)

        return spike


# ============================================================
# CIRCUIT-INFORMED SNN
# ============================================================

class CircuitInformedSNN(nn.Module):
    """
    MNIST SNN using the circuit-informed neuron.

    Structure:
        input image
        flatten
        linear hidden layer
        circuit-informed spiking neuron
        linear output layer
        class logits

    Important:
        The output layer is not spiking here.
        This makes training much more stable while still embedding
        the circuit-informed neuron in the SNN.
    """

    def __init__(
        self,
        fi_csv_path: str,
        hidden_size: int = HIDDEN_SIZE,
        time_steps: int = TIME_STEPS,
    ):
        super().__init__()

        self.T = time_steps

        self.fi_table = FITable(fi_csv_path)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(28 * 28, hidden_size)

        self.neuron1 = CircuitInformedNeuron(
            self.fi_table,
            dt=DT,
            tau_mem=TAU_MEM,
            v_threshold=V_THRESHOLD,
            v_reset=V_RESET,
            refractory_steps=REFRACTORY_STEPS,
        )

        self.fc2 = nn.Linear(hidden_size, 10)

    def reset_state(self):
        self.neuron1.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        logits_sum = 0.0

        for _ in range(self.T):
            h_current = self.fc1(x)
            h_spike = self.neuron1(h_current)

            logits = self.fc2(h_spike)

            logits_sum = logits_sum + logits

        return logits_sum / self.T


# ============================================================
# DATA
# ============================================================

def get_dataloaders(batch_size: int):
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        model.reset_state()

        optimizer.zero_grad()

        outputs = model(images)

        loss = F.cross_entropy(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / total_count
    acc = 100.0 * total_correct / total_count

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        model.reset_state()

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    acc = 100.0 * total_correct / total_count

    return acc


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    model = CircuitInformedSNN(
        fi_csv_path=CSV_PATH,
        hidden_size=HIDDEN_SIZE,
        time_steps=TIME_STEPS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    print("Starting training...")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
        )

        test_acc = evaluate(
            model=model,
            loader=test_loader,
            device=DEVICE,
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "hidden_size": HIDDEN_SIZE,
            "time_steps": TIME_STEPS,
        }

        history.append(row)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        os.path.join(OUTPUT_DIR, "task2_snn_training_history.csv"),
        index=False,
    )

    torch.save(
        model.state_dict(),
        os.path.join(OUTPUT_DIR, "task2_snn_model.pt"),
    )

    print("Done.")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()