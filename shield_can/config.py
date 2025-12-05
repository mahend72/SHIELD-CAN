# shield_can/config.py
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FeatureConfig:
    timing_bins: int = 32
    timing_hist_size: int = 1024      # N_Δt
    byte_window: int = 4096          # W_bytes
    entropy_half_life_s: float = 20.0
    dyn_half_life_s: float = 10.0
    sigma_min: float = 0.05

    kalman_q: float = 1e-4
    kalman_r: float = 1e-2
    kalman_sigma_r: float = 0.5
    kalman_sigma_rdot: float = 0.1
    kalman_delta_r_floor: float = 0.1

    toggle_half_life_frames: int = 256  # for payload byte toggling EMA


@dataclass
class ModelConfig:
    feature_dim: int = 9
    d_model: int = 192
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    window_size: int = 80
    num_classes: int = 5
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-2
    device: str = "cuda"


@dataclass
class PolicyConfig:
    # thresholds loosely following the paper
    static_sigma_mult_tier1: float = 3.0
    static_sigma_mult_tier2: float = 3.5
    dyn_tier1: float = 2.0
    dyn_tier2: float = 3.0
    conf_tier1: float = 0.6
    conf_tier2: float = 0.8

    rho_threshold: float = 1.5
    rho_persist_ms: float = 50.0

    # time constants
    tier2_persist_ms: float = 500.0

    # token bucket style
    shape_rate_multiplier: float = 1.1   # relative to baseline
    shape_duration_s: float = 2.0
    hysteresis_ms: float = 200.0

    drop_interval_ms: float = 200.0
    drop_burst: int = 2
    drop_hold_s: float = 1.0

    # ECU quarantine
    ecu_window_s: float = 1.0
    ecu_quarantine_threshold: int = 2

    # safe mode
    safe_mode_window_ms: float = 500.0


@dataclass
class SafetyConfig:
    safety_critical_ids: List[int]
    id_to_ecu: Dict[int, str]
