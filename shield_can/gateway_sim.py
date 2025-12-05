# shield_can/gateway_sim.py
from __future__ import annotations
from typing import Dict, Callable, Optional

import time
import numpy as np
import torch

from .config import FeatureConfig, ModelConfig, PolicyConfig, SafetyConfig
from .features import StreamingFeatureExtractor
from .model import EdgeTransformer
from .policy import SelfHealingPolicy


class GatewaySimulator:
    """
    Simple event-loop style gateway:
    - ingest frames (e.g., from CSV or socketCAN callback)
    - compute features
    - run model
    - pass scores to self-healing policy
    - decide whether to forward, shape, drop, etc.
    """

    def __init__(
        self,
        feat_cfg: FeatureConfig,
        model_cfg: ModelConfig,
        policy_cfg: PolicyConfig,
        safety_cfg: SafetyConfig,
        num_classes: int,
        device: str = "cpu",
    ):
        self.fe = StreamingFeatureExtractor(feat_cfg)
        self.model = EdgeTransformer(model_cfg).to(device)
        self.device = device
        self.policy = SelfHealingPolicy(policy_cfg, safety_cfg)
        self.W = model_cfg.window_size
        self.num_classes = num_classes

        self._window_buf: list[np.ndarray] = []

    def load_weights(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

    def process_frame(
        self,
        t_s: float,
        can_id: int,
        dlc: int,
        payload: bytes,
        ecu: str,
    ) -> Dict[str, object]:
        """
        Process a single frame, returning a dict with:
          - forward (bool): whether to forward frame on CAN
          - tier (int), action (str)
          - logits, conf, predicted_class
        """
        z = self.fe.update(t_s=t_s, can_id=can_id, dlc=dlc, payload=payload)
        self._window_buf.append(z)

        if len(self._window_buf) < self.W:
            # not enough context yet; just pass through
            return {"forward": True, "tier": 0, "action": "warmup"}

        if len(self._window_buf) > self.W:
            self._window_buf = self._window_buf[-self.W :]

        x = np.stack(self._window_buf, axis=0)  # (W, F)
        x_t = torch.from_numpy(x).unsqueeze(0).float().to(self.device)  # (1, W, F)

        with torch.no_grad():
            logits = self.model(x_t)[0]  # (C,)
            probs = torch.softmax(logits, dim=-1)
            conf, pred_idx = torch.max(probs, dim=-1)

        conf_val = float(conf.item())
        pred_class = int(pred_idx.item())

        # simple anomaly scores:
        # we use entropies + rho as proxies (can be refined)
        H_time, H_data, rho_kf, nis, dlc_drift, toggle_ema, id_norm, dlc_f, dt_norm = (
            self._window_buf[-1]
        )
        S_static = float(H_time + H_data)
        S_dyn = float(abs(rho_kf - 1.0) + nis)

        t = t_s
        ecu_name = ecu or "unknown"

        tier, action = self.policy.decide(
            t=t,
            can_id=can_id,
            ecu=ecu_name,
            S_static=S_static,
            S_dyn=S_dyn,
            conf=conf_val,
            rho_kf=float(rho_kf),
            nis=float(nis),
            predicted_class=pred_class,
        )

        now = t
        if self.policy.is_dropped(can_id, now):
            forward = False
        else:
            forward = True

        return {
            "forward": forward,
            "tier": tier,
            "action": action,
            "logits": logits.cpu().numpy(),
            "conf": conf_val,
            "predicted_class": pred_class,
        }
