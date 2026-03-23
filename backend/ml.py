"""
Synthetic dataset generation and model training for server digital twin.
Physics-inspired rules:
  - Temperature rises with CPU load and ambient temp, cooled by fan speed
  - Power consumption is driven by CPU and memory utilization
  - Fan speed responds to temperature with some lag
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

RANDOM_SEED = 42
N_SAMPLES = 2000
DT = 30  # seconds per step


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Workload patterns ---
    t = np.arange(n_samples) * DT
    # Diurnal-ish workload cycle + random spikes
    base_load = 30 + 20 * np.sin(2 * np.pi * t / (24 * 3600)) + rng.normal(0, 5, n_samples)
    spike = rng.random(n_samples) < 0.05  # 5% chance of burst
    cpu_util = np.clip(base_load + spike * rng.uniform(30, 50, n_samples), 5, 100)

    # Memory utilization: correlated with cpu but slower moving
    mem_util = np.clip(40 + 0.35 * cpu_util + rng.normal(0, 4, n_samples), 10, 95)

    # Ambient temperature (slow drift)
    ambient_temp = 22 + 3 * np.sin(2 * np.pi * t / (24 * 3600 * 2)) + rng.normal(0, 0.5, n_samples)

    # --- Simulate dynamics with simple Euler integration ---
    temp = np.zeros(n_samples)
    fan_speed = np.zeros(n_samples)
    power = np.zeros(n_samples)

    temp[0] = 45.0
    fan_speed[0] = 40.0

    for i in range(1, n_samples):
        # Power: base + cpu contribution + memory + noise
        power[i - 1] = (
            80
            + 1.2 * cpu_util[i - 1]
            + 0.4 * mem_util[i - 1]
            + rng.normal(0, 3)
        )

        # Fan responds to temperature (PID-like) with lag
        target_fan = np.clip(20 + 2.5 * (temp[i - 1] - ambient_temp[i - 1]), 10, 100)
        fan_speed[i] = 0.8 * fan_speed[i - 1] + 0.2 * target_fan + rng.normal(0, 1)
        fan_speed[i] = np.clip(fan_speed[i], 10, 100)

        # Temperature: heating from power, cooling from fan and ambient
        heat_in = 0.003 * power[i - 1]
        cool_fan = 0.008 * fan_speed[i] * (temp[i - 1] - ambient_temp[i - 1])
        cool_ambient = 0.002 * (temp[i - 1] - ambient_temp[i - 1])
        dtemp = heat_in - cool_fan - cool_ambient + rng.normal(0, 0.2)
        temp[i] = np.clip(temp[i - 1] + dtemp, ambient_temp[i] + 5, 95)

    power[n_samples - 1] = power[n_samples - 2]

    # Next-step temperature (prediction target)
    next_temp = np.roll(temp, -1)
    next_temp[-1] = temp[-1]

    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n_samples, freq=f"{DT}s"),
        "cpu_util": np.round(cpu_util, 2),
        "mem_util": np.round(mem_util, 2),
        "ambient_temp": np.round(ambient_temp, 2),
        "fan_speed": np.round(fan_speed, 2),
        "power_w": np.round(power, 2),
        "temperature": np.round(temp, 2),
        "next_temperature": np.round(next_temp, 2),
    })

    return df


FEATURES = ["cpu_util", "mem_util", "ambient_temp", "fan_speed", "power_w", "temperature"]
TARGET = "next_temperature"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


def train_model(df: pd.DataFrame) -> Pipeline:
    X = df[FEATURES].values
    y = df[TARGET].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )),
    ])
    model.fit(X, y)

    # Save
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def load_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


def predict(model: Pipeline, features: dict) -> float:
    X = np.array([[features[f] for f in FEATURES]])
    return float(model.predict(X)[0])


def delete_model():
    """Remove stale model file (e.g. built on a different numpy version)."""
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print(f"Deleted stale model at {MODEL_PATH}")


if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df.describe())
    print("\nTraining model...")
    model = train_model(df)
    sample = df.iloc[-50]
    pred = predict(model, sample.to_dict())
    print(f"\nSample prediction: {pred:.2f}°C (actual next: {sample['next_temperature']:.2f}°C)")