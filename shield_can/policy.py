# shield_can/policy.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import time

import numpy as np

from .config import PolicyConfig, SafetyConfig


@dataclass
class IDRuntimeState:
    last_shape_until: float = 0.0
    last_drop_until: float = 0.0
    last_hysteresis_until: float = 0.0
    last_tier2_start: Optional[float] = None
    last_dyn_high: Optional[float] = None


@dataclass
class ECURuntimeState:
    events: Dict[float, int] = field(default_factory=dict)  # t -> count


class SelfHealingPolicy:
    """
    Stateless w.r.t. ML model; stateful over IDs/ECUs and time.

    Inputs per frame:
      t, can_id, ecu, S_static, S_dyn, conf, rho_kf, nis
    Outputs:
      tier (0-4) and action string.
    """

    def __init__(self, pcfg: PolicyConfig, scfg: SafetyConfig):
        self.pcfg = pcfg
        self.scfg = scfg

        self.id_state: Dict[int, IDRuntimeState] = {}
        self.ecu_state: Dict[str, ECURuntimeState] = {}

        # baseline stats for S (here just simple EWMA)
        self.S_mean = 0.0
        self.S_var = 0.0
        self.S_n = 0

    def _now(self) -> float:
        return time.time()

    def _get_id_state(self, can_id: int) -> IDRuntimeState:
        if can_id not in self.id_state:
            self.id_state[can_id] = IDRuntimeState()
        return self.id_state[can_id]

    def _get_ecu_state(self, ecu: str) -> ECURuntimeState:
        if ecu not in self.ecu_state:
            self.ecu_state[ecu] = ECURuntimeState()
        return self.ecu_state[ecu]

    def _update_S_stats(self, S: float):
        self.S_n += 1
        delta = S - self.S_mean
        self.S_mean += delta / self.S_n
        self.S_var += delta * (S - self.S_mean)

    @property
    def S_sigma(self) -> float:
        if self.S_n < 2:
            return 1.0
        return float(np.sqrt(max(self.S_var / (self.S_n - 1), 1e-6)))

    def decide(
        self,
        t: float,
        can_id: int,
        ecu: str,
        S_static: float,
        S_dyn: float,
        conf: float,
        rho_kf: float,
        nis: float,
        predicted_class: int,
    ) -> Tuple[int, str]:
        """
        Return (tier, action_string).
        predicted_class is an int index; you can map it as you like
        (e.g. 0=normal, 1=DoS, 2=fuzzy, 3=malfunction, 4=spoof).
        """

        # update global S stats
        self._update_S_stats(S_static)
        mu_S = self.S_mean
        sigma_S = self.S_sigma

        zS = 0.0 if sigma_S == 0 else abs(S_static - mu_S) / sigma_S

        pcfg = self.pcfg
        idst = self._get_id_state(can_id)
        action = "log"
        tier = 0

        safety_critical = can_id in self.scfg.safety_critical_ids

        # soft ignore if everything looks benign
        if (
            conf < pcfg.conf_tier1
            and zS < 2.0
            and S_dyn < pcfg.dyn_tier1
        ):
            return 0, "log"

        # Tier 1: shaping
        strong_dyn = S_dyn >= pcfg.dyn_tier1
        strong_S = zS >= pcfg.static_sigma_mult_tier1
        strong_rho = rho_kf > pcfg.rho_threshold

        if strong_dyn or strong_S or strong_rho:
            # never shape safety-critical IDs, but still log
            if not safety_critical:
                now = t
                idst.last_shape_until = max(
                    idst.last_shape_until, now + pcfg.shape_duration_s
                )
                idst.last_hysteresis_until = max(
                    idst.last_hysteresis_until,
                    now + pcfg.hysteresis_ms / 1000.0,
                )
                tier = max(tier, 1)
                action = "shape"

        # Tier 2: dropping / ECU-local quarantine
        very_strong_dyn = S_dyn >= pcfg.dyn_tier2
        very_strong_S = zS >= pcfg.static_sigma_mult_tier2
        class_sus = predicted_class in (1, 2, 3) and conf >= pcfg.conf_tier2

        if very_strong_dyn or very_strong_S or class_sus:
            if not safety_critical:
                now = t
                idst.last_drop_until = max(
                    idst.last_drop_until, now + pcfg.drop_hold_s
                )
                tier = max(tier, 2)
                action = "drop"

                # ECU escalation
                ecu_st = self._get_ecu_state(ecu)
                ecu_st.events[now] = ecu_st.events.get(now, 0) + 1

                # clean old events
                cutoff = now - pcfg.ecu_window_s
                ecu_st.events = {
                    tt: c for tt, c in ecu_st.events.items() if tt >= cutoff
                }
                total = sum(ecu_st.events.values())
                if total >= pcfg.ecu_quarantine_threshold:
                    if not safety_critical:
                        tier = max(tier, 3)
                        action = f"quarantine_ecu:{ecu}"

        # Tier 3/4: safe mode (very simplified trigger)
        if (
            predicted_class == 4 and conf >= 0.9
        ) or (S_dyn >= pcfg.dyn_tier2 and very_strong_S):
            # safe-mode trigger guarded by safety constraints
            if not safety_critical:
                tier = max(tier, 4)
                action = "safe_mode"

        return tier, action

    # Convenience helpers for gateway loop ----------------

    def is_dropped(self, can_id: int, now: float) -> bool:
        if can_id not in self.id_state:
            return False
        st = self.id_state[can_id]
        return now < st.last_drop_until

    def is_shaped(self, can_id: int, now: float) -> bool:
        if can_id not in self.id_state:
            return False
        st = self.id_state[can_id]
        return now < st.last_shape_until
