"""Train and export the energy-management LME ICNN model."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EM_DIR = SCRIPT_DIR.parent
DATA_DIR = EM_DIR / "datasets"
MODEL_DIR = EM_DIR / "model"

TRAIN_DATA_PATH = DATA_DIR / "eco_mpc-rho=0.1-train.npz"
TEST_DATA_PATH = DATA_DIR / "eco_mpc-rho=0.1-test.npz"
MODEL_PATH = MODEL_DIR / "neco_mpc-rho=0.1"

WIDTHS = [16, 16]
LEARNING_RATE = 1e-3
GRAD_WEIGHT = 5.0
L2_REG = 0.0
BATCH_SIZE = 16
EPOCHS = 5000
SEED = 0

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ICNN_em import (
    act_p,
    batched_forward,
    batched_grad_wrt_x,
    to_serializable,
    train_icnn,
)


def load_dataset(path: Path):
    """Load an energy-management LME dataset in learner-friendly orientation."""
    with np.load(path) as data:
        return data["input"].T, data["enve"], data["grad"].T, data["rho"].item()


def evaluate(params, Xva, yva, gva):
    """Compute held-out value and gradient mean-square errors."""
    y_pred = batched_forward(params, jnp.asarray(Xva))
    g_pred = batched_grad_wrt_x(params, jnp.asarray(Xva))

    val_mse = jnp.mean((y_pred - jnp.asarray(yva)) ** 2)
    grad_mse = jnp.mean(jnp.sum((g_pred - jnp.asarray(gva)) ** 2, axis=1))
    return val_mse, grad_mse


def prepare_for_export(params, rho):
    """Apply nonnegative ICNN projections expected by the Julia model loader."""
    params["rho"] = rho
    params["v"] = act_p(params["v"])

    for i in range(1, len(params["W"])):
        params["W"][i] = act_p(params["W"][i])

    return params


def save_model(params, path: Path) -> None:
    """Save the trained ICNN in both pickle and JSON formats."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(params, f)

    with open(path.with_suffix(".json"), "w") as f:
        json.dump(to_serializable(params), f)


if __name__ == "__main__":
    print(jax.devices())

    Xtr, ytr, gtr, rho = load_dataset(TRAIN_DATA_PATH)
    Xva, yva, gva, _ = load_dataset(TEST_DATA_PATH)

    n_features, n_samples = Xtr.shape[1], Xtr.shape[0]
    print(f"Number of data: {n_samples}")

    params = train_icnn(
        Xtr,
        ytr,
        gtr,
        n_in=n_features,
        widths=WIDTHS,
        lr=LEARNING_RATE,
        grad_weight=GRAD_WEIGHT,
        l2_reg=L2_REG,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        seed=SEED,
    )

    val_mse, grad_mse = evaluate(params, Xva, yva, gva)
    print(f"[TEST] value MSE: {val_mse:.4e} | grad MSE: {grad_mse:.4e}")

    save_model(prepare_for_export(params, rho), MODEL_PATH)
