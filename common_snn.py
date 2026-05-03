import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# COMMON USER CONFIG
# ============================================================

FI_CSV_PATH = "/Users/rojindawood/Downloads/inputCurrent_vs_outputFrequency_IF.csv"
BASE_OUTPUT_DIR = "remaining_tasks_outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DT = 1e-4
TAU_MEM = 5e-3
V_THRESHOLD = 1.0
V_RESET = 0.0
REFRACTORY_STEPS = 0

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-3
DEFAULT_TIME_STEPS = 10
DEFAULT_HIDDEN_SIZE = 128

HIDDEN_SIZES = [512, 256, 128, 64, 32]
RESOLUTIONS = [4, 7, 14, 28]
TIME_STEPS_LIST = [5, 10, 20]
QUANT_BITS = [2, 4, 8]


# ============================================================
# f-I TABLE
# ============================================================

class FITable:
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
            raise ValueError(f"CSV columns not recognized: {list(df.columns)}")

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
    @staticmethod
    def forward(ctx, v_minus_thr: torch.Tensor):
        ctx.save_for_backward(v_minus_thr)
        return (v_minus_thr >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v_minus_thr,) = ctx.saved_tensors

        slope = 10.0
        sig = torch.sigmoid(slope * v_minus_thr)
        grad = slope * sig * (1.0 - sig)

        return grad_output * grad


def spike_fn(x: torch.Tensor) -> torch.Tensor:
    return SurrogateSpike.apply(x)


# ============================================================
# QUANTIZATION
# ============================================================

def quantize_tensor_symmetric(x: torch.Tensor, bits: Optional[int]) -> torch.Tensor:
    if bits is None:
        return x

    qmax = 2 ** (bits - 1) - 1

    if qmax <= 0:
        return torch.zeros_like(x)

    scale = x.detach().abs().max()

    if scale.item() == 0:
        return x

    step = scale / qmax

    return torch.round(x / step) * step


def maybe_quantize_weights(model: nn.Module, bits: Optional[int]):
    if bits is None:
        return

    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" in name:
                p.copy_(quantize_tensor_symmetric(p, bits))


# ============================================================
# NEURONS
# ============================================================

class CircuitInformedNeuron(nn.Module):
    """
    Circuit-informed spiking neuron.

    Uses the measured Cadence f-I curve to map input current to expected
    firing behavior while preserving:
    - membrane integration
    - threshold spiking
    - reset
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

        self.last_input_current = None
        self.last_predicted_rate = None
        self.last_spike = None

    def reset_state(self):
        self.v = None
        self.refrac = None
        self.phase = None

        self.last_input_current = None
        self.last_predicted_rate = None
        self.last_spike = None

    def activation_to_current(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map arbitrary neural-network activation into the measured Cadence
        current range.

        This is important because fc1(x) is not naturally in nA.
        """
        min_i = self.fi_table.min_current_nA
        max_i = self.fi_table.max_current_nA

        current_nA = min_i + (max_i - min_i) * torch.sigmoid(x)

        return current_nA

    def current_to_drive(self, current_nA: torch.Tensor):
        """
        Convert current to:
        - predicted firing rate from measured Cadence f-I curve
        - membrane drive term
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

        # Phase accumulator keeps firing-rate behavior close to measured f-I curve
        self.phase = self.phase + predicted_rate * self.dt * active

        threshold_spike = spike_fn(self.v - self.v_threshold)
        phase_spike = spike_fn(self.phase - 1.0)

        spike = torch.clamp(threshold_spike + phase_spike, max=1.0)

        # Reset after spike
        self.v = self.v * (1.0 - spike) + self.v_reset * spike
        self.phase = torch.clamp(self.phase - spike, min=0.0)

        if self.refractory_steps > 0:
            self.refrac = torch.clamp(self.refrac - 1, min=0) + spike * self.refractory_steps
        else:
            self.refrac = torch.zeros_like(self.refrac)

        self.last_input_current = current_nA.detach().clone()
        self.last_predicted_rate = predicted_rate.detach().clone()
        self.last_spike = spike.detach().clone()

        return spike


class DefaultSpikingNeuron(nn.Module):
    """
    Generic default spiking neuron for Task 6 comparison.
    """

    def __init__(
        self,
        dt: float = DT,
        tau_mem: float = TAU_MEM,
        v_threshold: float = V_THRESHOLD,
        v_reset: float = V_RESET,
    ):
        super().__init__()

        self.dt = dt
        self.tau_mem = tau_mem
        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.v = None

    def reset_state(self):
        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(x)

        x_pos = torch.relu(x)

        drive = 0.25 + 3.0 * x_pos / max(x_pos.detach().abs().max().item(), 1e-12)

        dv = (-self.v + drive) * (self.dt / self.tau_mem)
        self.v = self.v + dv

        spike = spike_fn(self.v - self.v_threshold)

        self.v = self.v * (1.0 - spike) + self.v_reset * spike

        return spike


# ============================================================
# SNN MODEL
# ============================================================

class OneHiddenLayerSNN(nn.Module):
    """
    One-hidden-layer SNN.

    Fixed architecture:

        input image
        -> flatten
        -> fc1
        -> spiking neuron
        -> fc2
        -> class logits

    Important:
    The output layer is not spiking. This keeps training stable while
    still using the circuit-informed neuron inside the SNN.
    """

    def __init__(
        self,
        fi_csv_path: str,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        time_steps: int = DEFAULT_TIME_STEPS,
        neuron_type: str = "circuit",
    ):
        super().__init__()

        self.T = time_steps
        self.hidden_size = hidden_size
        self.neuron_type = neuron_type

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

        if neuron_type == "circuit":
            self.fi_table = FITable(fi_csv_path)
            self.neuron1 = CircuitInformedNeuron(self.fi_table)

        elif neuron_type == "default":
            self.fi_table = None
            self.neuron1 = DefaultSpikingNeuron()

        else:
            raise ValueError("neuron_type must be 'circuit' or 'default'")

    def reset_state(self):
        self.neuron1.reset_state()

    def forward(self, x: torch.Tensor, record_hidden: bool = False):
        x = self.flatten(x)

        logits_sum = 0.0

        hidden_spikes_over_time = []
        hidden_currents_over_time = []

        for _ in range(self.T):
            h_current = self.fc1(x)
            h_spike = self.neuron1(h_current)

            if record_hidden:
                hidden_spikes_over_time.append(h_spike.detach().cpu())

                if hasattr(self.neuron1, "last_input_current") and self.neuron1.last_input_current is not None:
                    hidden_currents_over_time.append(self.neuron1.last_input_current.detach().cpu())
                else:
                    hidden_currents_over_time.append(torch.relu(h_current.detach()).cpu())

            logits = self.fc2(h_spike)

            logits_sum = logits_sum + logits

        if record_hidden:
            return logits_sum / self.T, hidden_spikes_over_time, hidden_currents_over_time

        return logits_sum / self.T


# ============================================================
# DATA HELPERS
# ============================================================

def resize_tensor_image(x: torch.Tensor, resolution: int) -> torch.Tensor:
    if resolution == 28:
        return x

    return F.interpolate(
        x,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    )


def pad_back_to_28(x: torch.Tensor) -> torch.Tensor:
    r = x.shape[-1]

    if r == 28:
        return x

    pad_total = 28 - r
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    return F.pad(x, (pad_left, pad_right, pad_left, pad_right))


def preprocess_batch(images: torch.Tensor, resolution: int) -> torch.Tensor:
    if resolution != 28:
        images = resize_tensor_image(images, resolution)
        images = pad_back_to_28(images)

    return images


def get_mnist_loaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
):
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

    if train_limit is not None:
        train_set = Subset(train_set, list(range(min(train_limit, len(train_set)))))

    if test_limit is not None:
        test_set = Subset(test_set, list(range(min(test_limit, len(test_set)))))

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
# TRAIN / EVAL
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device,
    resolution: int = 28,
    quant_bits: Optional[int] = None,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = preprocess_batch(images.to(device), resolution)
        labels = labels.to(device)

        model.reset_state()
        optimizer.zero_grad()

        outputs = model(images)

        loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

        maybe_quantize_weights(model, quant_bits)

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, 100.0 * total_correct / total_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device,
    resolution: int = 28,
):
    model.eval()

    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = preprocess_batch(images.to(device), resolution)
        labels = labels.to(device)

        model.reset_state()

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    return 100.0 * total_correct / total_count