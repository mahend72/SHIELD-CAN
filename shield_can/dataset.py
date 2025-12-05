# shield_can/dataset.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import FeatureConfig, ModelConfig
from .features import StreamingFeatureExtractor


class CANWindowDataset(Dataset):
    """
    Offline dataset that replays a CAN log through the streaming
    feature extractor, then builds sliding windows of length W.

    Assumes CSV columns:
    - timestamp (float, seconds or ms; if ms, normalise outside)
    - id (int or hex string)
    - dlc (int)
    - data0..data7 (int in [0,255], optional depending on dlc)
    - label (string or int)
    """

    def __init__(
        self,
        csv_path: str,
        feat_cfg: FeatureConfig,
        model_cfg: ModelConfig,
        label_map: Dict[str, int],
        step: int = 1,
    ):
        super().__init__()
        self.feat_cfg = feat_cfg
        self.model_cfg = model_cfg
        self.label_map = label_map
        self.step = step

        df = pd.read_csv(csv_path)
        self._normalise_df(df)

        self.features, self.labels = self._build_features(df)

    def _normalise_df(self, df: pd.DataFrame):
        # Ensure timestamp is float seconds
        if df["timestamp"].max() > 1e6:
            # assume ms
            df["timestamp"] = df["timestamp"].astype(float) / 1000.0
        else:
            df["timestamp"] = df["timestamp"].astype(float)

        # normalise ID; accept hex strings
        if df["id"].dtype == object:
            df["id"] = df["id"].apply(lambda x: int(str(x), 16))
        else:
            df["id"] = df["id"].astype(int)

        df["dlc"] = df["dlc"].astype(int)

    def _extract_payload(self, row: pd.Series) -> bytes:
        dlc = int(row["dlc"])
        vals = []
        for i in range(dlc):
            col = f"data{i}"
            if col in row:
                vals.append(int(row[col]))
            else:
                vals.append(0)
        return bytes(vals)

    def _build_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.sort_values("timestamp").reset_index(drop=True)
        fe = StreamingFeatureExtractor(self.feat_cfg)

        feat_list: List[np.ndarray] = []
        label_list: List[int] = []

        for _, row in df.iterrows():
            t = float(row["timestamp"])
            can_id = int(row["id"])
            dlc = int(row["dlc"])
            payload = self._extract_payload(row)
            z = fe.update(t_s=t, can_id=can_id, dlc=dlc, payload=payload)
            feat_list.append(z)

            lab_val = row["label"]
            if isinstance(lab_val, str):
                label_list.append(self.label_map[lab_val])
            else:
                label_list.append(int(lab_val))

        feats = np.stack(feat_list, axis=0)  # (N, F)
        labels = np.array(label_list, dtype=np.int64)

        # build windows
        W = self.model_cfg.window_size
        X_list = []
        y_list = []

        for i in range(W - 1, len(feats), self.step):
            X_list.append(feats[i - W + 1 : i + 1, :])
            y_list.append(labels[i])

        X = np.stack(X_list, axis=0)  # (M, W, F)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.features[idx]).float()
        y = torch.tensor(self.labels[idx]).long()
        return x, y
