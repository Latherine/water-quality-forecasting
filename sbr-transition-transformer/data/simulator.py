from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from configs.default import LANDMARK_NAMES, SimConfig
from data.landmarks import gaussian_bump, gaussian_dip, smooth_piecewise_linear


@dataclass
class SimulatedCycle:
    t: np.ndarray  # (L,) minutes
    ph: np.ndarray  # (L,)
    orp: np.ndarray  # (L,)
    do: np.ndarray  # (L,)
    landmarks: dict[str, Optional[float]]  # minutes, or None if absent
    presence: dict[str, bool]
    seed: int
    meta: dict = field(default_factory=dict)


class SBRCycleSimulator:
    """Generates synthetic SBR-cycle pH/ORP/DO curves with exactly-known
    phase-transition ("bending point") ground truth.

    Landmark chronology within a cycle (as configured fractions of total
    cycle length T): valley -> elbow (end of nitrification) -> knee -> apex
    (end of denitrification) — the common post-anoxic configuration (aerobic
    phase first). Channel assignment: pH carries the ammonia valley, ORP
    carries the nitrate knee + nitrate apex, DO carries the DO elbow.
    """

    def __init__(self, config: SimConfig):
        self.config = config

    def generate_cycle(self, seed: int) -> SimulatedCycle:
        cfg = self.config
        rng = np.random.default_rng(seed)

        total_t = rng.uniform(cfg.t_min, cfg.t_max)
        n_steps = int(round(total_t / cfg.dt)) + 1
        t = np.arange(n_steps) * cfg.dt

        nominal = {
            "valley": cfg.valley_frac * total_t,
            "elbow": cfg.elbow_frac * total_t,
            "knee": cfg.knee_frac * total_t,
            "apex": cfg.apex_frac * total_t,
        }
        jitter_std = cfg.jitter_std_frac * total_t
        min_gap = cfg.min_gap_frac * total_t

        positions = self._sample_ordered_positions(rng, nominal, jitter_std, min_gap)

        presence = {name: True for name in LANDMARK_NAMES}
        if rng.random() < cfg.p_missing_landmark:
            dropped = rng.choice(LANDMARK_NAMES)
            presence[dropped] = False

        ph = self._baseline(rng, t, cfg.ph_start_range, cfg.ph_slope_range)
        orp = self._baseline(rng, t, cfg.orp_start_range, cfg.orp_slope_range)
        do = self._baseline(rng, t, cfg.do_start_range, cfg.do_slope_range)

        if presence["valley"]:
            depth = rng.uniform(*cfg.valley_depth_range)
            sigma = rng.uniform(*cfg.valley_sigma_range)
            ph = ph + gaussian_dip(t, positions["valley"], depth, sigma)

        if presence["knee"]:
            slope_before = rng.uniform(*cfg.knee_slope_before_range)
            slope_after = rng.uniform(*cfg.knee_slope_after_range)
            sharpness = rng.uniform(*cfg.knee_sharpness_range)
            orp = orp + smooth_piecewise_linear(
                t, positions["knee"], slope_before, slope_after, sharpness
            )

        if presence["apex"]:
            height = rng.uniform(*cfg.apex_height_range)
            sigma = rng.uniform(*cfg.apex_sigma_range)
            orp = orp + gaussian_bump(t, positions["apex"], height, sigma)

        if presence["elbow"]:
            slope_before = rng.uniform(*cfg.elbow_slope_before_range)
            slope_after = rng.uniform(*cfg.elbow_slope_after_range)
            sharpness = rng.uniform(*cfg.elbow_sharpness_range)
            do = do + smooth_piecewise_linear(
                t, positions["elbow"], slope_before, slope_after, sharpness
            )

        ph = ph + rng.normal(0.0, cfg.ph_noise_std, size=n_steps)
        orp = orp + rng.normal(0.0, cfg.orp_noise_std, size=n_steps)
        do = do + rng.normal(0.0, cfg.do_noise_std, size=n_steps)
        do = np.clip(do, 0.0, None)  # DO cannot be negative

        landmarks = {name: (positions[name] if presence[name] else None) for name in LANDMARK_NAMES}

        return SimulatedCycle(
            t=t,
            ph=ph,
            orp=orp,
            do=do,
            landmarks=landmarks,
            presence=presence,
            seed=seed,
            meta={"dt": cfg.dt, "total_t": total_t, "n_steps": n_steps},
        )

    def generate_dataset(self, seeds) -> list[SimulatedCycle]:
        return [self.generate_cycle(seed) for seed in seeds]

    @staticmethod
    def _baseline(rng, t, start_range, slope_range) -> np.ndarray:
        start = rng.uniform(*start_range)
        slope = rng.uniform(*slope_range)
        return start + slope * t

    @staticmethod
    def _sample_ordered_positions(rng, nominal: dict, jitter_std: float, min_gap: float) -> dict:
        """Jitter each nominal position independently, rejection-resampling
        until the landmarks stay in the order given by `nominal`'s insertion
        order (valley < elbow < knee < apex) with at least `min_gap` between
        consecutive landmarks. Falls back to the nominal positions (which
        already satisfy the ordering by construction of the *_frac defaults)
        if resampling doesn't converge within the configured attempt budget.
        """
        names = list(nominal.keys())
        max_tries = 50
        for _ in range(max_tries):
            jittered = {name: nominal[name] + rng.normal(0.0, jitter_std) for name in names}
            ordered = [jittered[name] for name in names]
            if all(b - a >= min_gap for a, b in zip(ordered, ordered[1:])):
                return jittered
        return dict(nominal)
