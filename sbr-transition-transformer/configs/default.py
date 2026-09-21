from __future__ import annotations

from dataclasses import dataclass

LANDMARK_NAMES: tuple[str, ...] = ("valley", "elbow", "knee", "apex")
CHANNEL_NAMES: tuple[str, ...] = ("ph", "orp", "do")


@dataclass
class SimConfig:
    """Parameters for SBRCycleSimulator. All time values are in minutes."""

    dt: float = 5.0
    t_min: float = 150.0
    t_max: float = 240.0

    # Nominal landmark position as a fraction of total cycle length T, in
    # chronological order: valley -> elbow (end of nitrification, aerobic
    # phase) -> knee -> apex (end of denitrification, anoxic phase). This
    # assumes the common post-anoxic SBR configuration (aerobic phase runs
    # first); a pre-anoxic plant would need this order reversed.
    valley_frac: float = 0.25
    elbow_frac: float = 0.35
    knee_frac: float = 0.55
    apex_frac: float = 0.60

    jitter_std_frac: float = 0.03
    min_gap_frac: float = 0.05
    max_jitter_resamples: int = 50

    # Channel baseline drift: value(t) = start + slope * t, start/slope drawn
    # per cycle from Uniform(*_range).
    ph_start_range: tuple[float, float] = (6.8, 7.4)
    ph_slope_range: tuple[float, float] = (-0.002, 0.002)
    orp_start_range: tuple[float, float] = (-50.0, 50.0)
    orp_slope_range: tuple[float, float] = (-0.3, 0.3)
    do_start_range: tuple[float, float] = (0.1, 0.5)
    do_slope_range: tuple[float, float] = (0.0, 0.01)

    # Landmark shape parameters.
    valley_depth_range: tuple[float, float] = (0.15, 0.45)
    valley_sigma_range: tuple[float, float] = (4.0, 10.0)
    apex_height_range: tuple[float, float] = (15.0, 40.0)
    apex_sigma_range: tuple[float, float] = (4.0, 10.0)
    knee_slope_before_range: tuple[float, float] = (-2.5, -1.0)
    knee_slope_after_range: tuple[float, float] = (-0.3, 0.1)
    knee_sharpness_range: tuple[float, float] = (3.0, 8.0)
    elbow_slope_before_range: tuple[float, float] = (0.0, 0.01)
    elbow_slope_after_range: tuple[float, float] = (0.02, 0.08)
    elbow_sharpness_range: tuple[float, float] = (3.0, 8.0)

    # Measurement noise, additive Gaussian, std per channel in the channel's
    # own units (pH noise is small in absolute terms; ORP/DO noisier).
    ph_noise_std: float = 0.03
    orp_noise_std: float = 2.5
    do_noise_std: float = 0.08

    # Probability that one (uniformly chosen) landmark is entirely absent
    # from a cycle: its presence flag is False and no shape feature for it
    # is injected. Kept at 0 for the base train/val/test split and >0 only
    # for the "hard" stress split (see seed blocks below).
    p_missing_landmark: float = 0.0

    use_derivatives: bool = True
    max_len: int = 64


# Disjoint seed *blocks* per split, fixed once here so that regenerating a
# split (e.g. with more cycles) can never accidentally overlap another
# split's seeds.
TRAIN_SEEDS = range(0, 8_000)
VAL_SEEDS = range(100_000, 101_500)
TEST_SEEDS = range(200_000, 201_500)
TEST_HARD_SEEDS = range(300_000, 300_500)

# A small slice of each block, used by the fast end-to-end sanity run.
TINY_TRAIN_SEEDS = range(0, 1_500)
TINY_VAL_SEEDS = range(100_000, 100_300)
TINY_TEST_SEEDS = range(200_000, 200_300)


def hard_config() -> SimConfig:
    """Wider noise/jitter and a chance of a missing landmark, for the test_hard split."""
    return SimConfig(
        jitter_std_frac=0.06,
        min_gap_frac=0.04,
        ph_noise_std=0.06,
        orp_noise_std=4.0,
        do_noise_std=0.15,
        p_missing_landmark=0.15,
    )


@dataclass
class ModelConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_len: int = 64
    num_landmarks: int = 4
    head: str = "heatmap"  # "heatmap" or "pointer"


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 30
    heatmap_sigma_steps: float = 3.0
    lambda_presence: float = 0.5
    lambda_l1: float = 0.1
    device: str = "cpu"
    seed: int = 0


def tiny_configs() -> tuple[SimConfig, ModelConfig, TrainConfig]:
    """Small model/short run for the fast end-to-end sanity check (see README)."""
    sim = SimConfig()
    model = ModelConfig(d_model=32, nhead=2, num_layers=2, dim_feedforward=64)
    train = TrainConfig(batch_size=32, epochs=5)
    return sim, model, train
