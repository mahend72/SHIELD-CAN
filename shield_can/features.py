# shield_can/features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import numpy as np

from .config import FeatureConfig
from .utils import EWMAStat


@dataclass
class KalmanState:
    x: np.ndarray          # shape (2,) -> [rate, rate_dot]
    P: np.ndarray          # shape (2, 2)
    last_t: Optional[float]


@dataclass
class PerIDState:
    last_t: Optional[float]
    kalman: KalmanState
    dlc_hist: np.ndarray            # shape (16,)
    dlc_modes: List[int]
    toggle_ref: Optional[np.ndarray]
    toggle_ema: float


class StreamingFeatureExtractor:
    """
    Stateful streaming feature pipeline implementing the SHIELD-CAN ideas:
    - global timing entropy over Δt histogram
    - global payload byte entropy
    - per-ID Kalman filter over message rate
    - DLC drift
    - byte-position toggling EMA

    update(frame) -> 9-D feature vector for that frame.
    """

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

        # global timing histogram over Δt bins
        self.timing_bins = cfg.timing_bins
        self.timing_hist = np.zeros(self.timing_bins, dtype=np.int64)
        self.timing_ring = np.full(cfg.timing_hist_size, -1, dtype=np.int32)
        self.timing_head = 0
        self.timing_filled = 0

        # global byte histogram (256 possible values)
        self.byte_hist = np.zeros(256, dtype=np.int64)
        self.byte_ring = np.full(cfg.byte_window, -1, dtype=np.int16)
        self.byte_head = 0
        self.byte_filled = 0

        # per-feature EWMA stats for entropy & dynamics
        self.entropy_time_ewma = EWMAStat(cfg.entropy_half_life_s, cfg.sigma_min)
        self.entropy_data_ewma = EWMAStat(cfg.entropy_half_life_s, cfg.sigma_min)
        self.ratio_ewma = EWMAStat(cfg.dyn_half_life_s, cfg.sigma_min)

        # per-ID state
        self.per_id: Dict[int, PerIDState] = {}

        # precompute toggle α from "half life" in frames
        self.toggle_alpha = 1.0 - math.exp(
            -math.log(2.0) / float(cfg.toggle_half_life_frames)
        )

        # global baseline for Δt binning (we'll adapt as we see traffic)
        self.min_dt_us = 100.0
        self.max_dt_us = 1e6

    # ------------ helpers ------------

    def _get_or_init_id_state(self, can_id: int, t: float) -> PerIDState:
        if can_id in self.per_id:
            return self.per_id[can_id]

        # Kalman initialisation: we don't know rate yet; start sub-optimally
        r0 = 10.0  # Hz, arbitrary; will adapt
        x = np.array([r0, 0.0], dtype=np.float64)
        P = np.diag([self.cfg.kalman_sigma_r**2, self.cfg.kalman_sigma_rdot**2])

        st = PerIDState(
            last_t=None,
            kalman=KalmanState(x=x, P=P, last_t=t),
            dlc_hist=np.zeros(16, dtype=np.int64),
            dlc_modes=[8],  # classical CAN default
            toggle_ref=None,
            toggle_ema=0.0,
        )
        self.per_id[can_id] = st
        return st

    def _dt_bin_index(self, dt_us: float) -> int:
        # Clamp & log-scale into [0, timing_bins)
        dt_us = max(dt_us, 1.0)
        self.min_dt_us = min(self.min_dt_us, dt_us)
        self.max_dt_us = max(self.max_dt_us, dt_us * 1.1)

        log_min = math.log10(self.min_dt_us)
        log_max = math.log10(self.max_dt_us + 1e-6)
        if log_max <= log_min:
            return 0
        x = (math.log10(dt_us) - log_min) / (log_max - log_min)
        idx = int(x * self.timing_bins)
        idx = max(0, min(self.timing_bins - 1, idx))
        return idx

    def _update_timing_entropy(self, dt_us: float) -> float:
        b = self._dt_bin_index(dt_us)

        if self.timing_filled == self.cfg.timing_hist_size:
            old = self.timing_ring[self.timing_head]
            if old >= 0:
                self.timing_hist[old] -= 1
        else:
            self.timing_filled += 1

        self.timing_ring[self.timing_head] = b
        self.timing_head = (self.timing_head + 1) % self.cfg.timing_hist_size
        self.timing_hist[b] += 1

        eps = 1e-6
        total = float(self.timing_filled) + self.timing_bins * eps
        probs = (self.timing_hist.astype(np.float64) + eps) / total
        H = -np.sum(probs * np.log2(probs))
        return float(H)

    def _update_byte_entropy(self, payload: bytes) -> float:
        for v in payload:
            if self.byte_filled == self.cfg.byte_window:
                old = self.byte_ring[self.byte_head]
                if old >= 0:
                    self.byte_hist[old] -= 1
            else:
                self.byte_filled += 1

            self.byte_ring[self.byte_head] = v
            self.byte_head = (self.byte_head + 1) % self.cfg.byte_window
            self.byte_hist[v] += 1

        eps = 1e-6
        total = float(self.byte_filled) + 256.0 * eps
        probs = (self.byte_hist.astype(np.float64) + eps) / total
        H = -np.sum(probs * np.log2(probs))
        return float(H)

    def _kalman_update(self, st: PerIDState, dt_s: float) -> Tuple[float, float, float]:
        """2D constant-velocity rate model. Returns (rho_KF, NIS, rate_hat)."""
        kcfg = self.cfg
        x = st.kalman.x
        P = st.kalman.P

        # measurement: instantaneous rate
        z = 1.0 / max(dt_s, 1e-6)

        # state transition
        F = np.array([[1.0, dt_s], [0.0, 1.0]], dtype=np.float64)
        Q = kcfg.kalman_q * np.array(
            [[dt_s**3 / 3.0, dt_s**2 / 2.0], [dt_s**2 / 2.0, dt_s]],
            dtype=np.float64,
        )
        H = np.array([[1.0, 0.0]], dtype=np.float64)
        R = np.array([[kcfg.kalman_r]], dtype=np.float64)

        # predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # update
        y = z - (H @ x_pred)[0]
        S = float(H @ P_pred @ H.T + R)
        K = (P_pred @ H.T) / S  # shape (2,1)

        x_new = x_pred + (K * y).reshape(2)
        I = np.eye(2)
        P_new = (I - K @ H) @ P_pred

        st.kalman.x = x_new
        st.kalman.P = P_new

        rate_hat = max(x_new[0], 1e-6)
        rho = z / max(rate_hat, kcfg.kalman_delta_r_floor)
        nis = (y * y) / S

        return float(rho), float(nis), float(rate_hat)

    def _update_dlc_state(self, st: PerIDState, dlc: int) -> float:
        st.dlc_hist[dlc] += 1
        # recompute top-2 modes occasionally (cheap here)
        modes = np.argsort(st.dlc_hist)[-2:]
        st.dlc_modes = list(sorted(modes))
        drift = min(abs(dlc - m) for m in st.dlc_modes)
        return float(drift)

    def _update_toggling(self, st: PerIDState, payload: bytes) -> Tuple[float, float]:
        dlc = len(payload)
        if st.toggle_ref is not None and len(st.toggle_ref) == dlc:
            ref = st.toggle_ref
            pl = np.frombuffer(payload, dtype=np.uint8)
            inst = float(np.mean(pl != ref))
        else:
            inst = 1.0  # we have no good ref yet

        st.toggle_ema = (1.0 - self.toggle_alpha) * st.toggle_ema + self.toggle_alpha * inst

        # reference maintenance: update when pattern "stable"
        if st.toggle_ref is None or len(st.toggle_ref) != dlc or inst < 0.2:
            st.toggle_ref = np.frombuffer(payload, dtype=np.uint8).copy()

        return inst, st.toggle_ema

    # ------------ main API ------------

    def update(
        self,
        t_s: float,
        can_id: int,
        dlc: int,
        payload: bytes,
    ) -> np.ndarray:
        """
        Update the streaming state with a new CAN frame and
        return the 9-D feature vector z_i for this frame:

        [H_time, H_data, rho_KF, NIS, DLC_drift,
         toggle_ema, id_norm, dlc, dt_norm]
        """
        st = self._get_or_init_id_state(can_id, t_s)

        # Δt
        if st.last_t is None:
            dt_s = 0.01  # arbitrary small; first observation
        else:
            dt_s = max(t_s - st.last_t, 1e-6)
        st.last_t = t_s
        dt_us = dt_s * 1e6

        # global entropies
        H_time = self._update_timing_entropy(dt_us)
        H_data = self._update_byte_entropy(payload)

        # EWMA z-scores for entropies
        z_H_time = self.entropy_time_ewma.update(t_s, H_time)
        z_H_data = self.entropy_data_ewma.update(t_s, H_data)

        # Kalman cues
        rho_kf, nis, rate_hat = self._kalman_update(st, dt_s)
        log_rho = math.log(max(rho_kf, 1e-6))
        z_rho = self.ratio_ewma.update(t_s, log_rho)

        # DLC drift
        dlc_drift = self._update_dlc_state(st, dlc)

        # byte toggling
        toggle_inst, toggle_ema = self._update_toggling(st, payload)

        # build feature vector (raw + some normalisation)
        id_norm = float(can_id) / 2048.0  # rough scaling
        dt_norm = math.log10(dt_us + 1.0) / 6.0  # ~[0,1] for 1–1e6us

        z = np.array(
            [
                H_time,
                H_data,
                rho_kf,
                nis,
                dlc_drift,
                toggle_ema,
                id_norm,
                float(dlc),
                dt_norm,
            ],
            dtype=np.float32,
        )

        return z
