# SHIELD-CAN: A Self-Healing Edge-AI Intrusion Detection System for Automotive CAN Gateways

This repository contains a reference Python implementation of **SHIELD-CAN**, a self-healing edge-AI intrusion detection and response system designed for **automotive CAN gateways**.

The system combines:

- A **streaming, protocol-agnostic feature pipeline** with O(1) state per active CAN ID  
- An **edge-optimised encoder-only Transformer** operating on short windows of features  
- A **deterministic, multi-tier self-healing policy** that maps anomaly scores and predictions into concrete gateway actions (rate limiting, selective dropping, ECU quarantine) while enforcing explicit **safety invariants**

The implementation is structured to be deployable on embedded ARM platforms and usable for research on CAN IDS/IPS, self-healing policies, and safety-aware mitigation.

---

## Key Features

- **Streaming feature extraction**
  - Protocol-agnostic, operates directly on raw CAN frames
  - O(1) state per CAN ID; constant-time per-frame updates
  - Dual entropy measures (timing & payload), Kalman-based timing analysis, DLC drift, byte-position toggling

- **Edge-optimised Transformer model**
  - Encoder-only Transformer over short windows of features
  - Single STAT token for window-level classification
  - Static shapes and compact architecture suitable for quantisation and edge deployment

- **Deterministic self-healing policy**
  - Multi-tier response: logging → shaping (rate limiting) → dropping → ECU quarantine → (optional) safe-mode
  - Enforces safety invariants (e.g. never drop safety-critical IDs, bounded degraded modes)

- **Gateway-oriented design**
  - Streaming feature extractor + model + policy + gateway simulator
  - Designed to sit at an in-vehicle gateway between CAN segments

- **Research-friendly**
  - Works on standard CAN IDS datasets (e.g., Car-Hacking, IVN-IDS)
  - Training script, dataset loader, and example CSV schema included

---

## Repository Structure

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── examples/
│   └── sample_can_log.csv
├── train.py
└── shield_can/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── features.py
    ├── gateway_sim.py
    ├── model.py
    ├── policy.py
    └── utils.py
```

## Core Components

- `config.py` – dataclasses for feature, model, training, policy, and safety configs  
- `features.py` – streaming feature extractor (per-frame → 9-D feature vector)  
- `model.py` – encoder-only Transformer (`EdgeTransformer`)  
- `policy.py` – multi-tier self-healing policy with safety-aware actions  
- `dataset.py` – CAN log → feature windows for training  
- `gateway_sim.py` – example gateway loop using feature extractor, model, and policy  
- `train.py` – training script (PyTorch) for the Transformer  

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/shield-can.git
cd shield-can
```

### Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

### 3. Install dependencies

Using requirements.txt:

pip install --upgrade pip
pip install -r requirements.txt


Or install as an editable package:

pip install -e .


Python version: 3.9+ is recommended.

CAN Log Format

The implementation assumes CAN logs in a CSV format similar to:

timestamp,id,dlc,data0,data1,data2,data3,data4,data5,data6,data7,label
0.000000,0x130,8,0,0,0,0,0,0,0,0,Normal
0.010000,0x130,8,0,0,16,0,0,0,0,0,Normal
0.020000,0x130,8,0,0,32,0,0,0,0,0,Normal
0.030000,0x1A0,8,10,20,30,40,50,60,70,80,DoS


Expected columns:

timestamp – float (seconds). If in ms, it will be converted internally.

id – CAN ID as integer or hex string (e.g. 0x130).

dlc – data length code (0–8 for classical CAN).

data0 .. data7 – payload bytes (0–255); only the first dlc are used.

label – class label (string or int). Example label mapping (default in train.py):

{
    "Normal": 0,
    "DoS": 1,
    "Fuzzy": 2,
    "Malfunction": 3,
    "Spoof": 4,
}


You can adjust the label mapping in train.py to match your dataset.

Training the Transformer

The main training entrypoint is train.py.

Basic usage
python train.py \
  --train_csv data/car_hacking_train.csv \
  --val_csv data/car_hacking_val.csv \
  --out_dir runs/shield_can \
  --epochs 30 \
  --batch_size 128 \
  --lr 3e-4 \
  --device cuda


Arguments:

--train_csv – path to training CSV

--val_csv – path to validation CSV

--out_dir – output directory for checkpoints and logs

--epochs – number of training epochs

--batch_size – batch size

--lr – learning rate

--device – cuda or cpu

The script will:

Build streaming features using StreamingFeatureExtractor

Construct sliding windows of length W (configurable in ModelConfig.window_size)

Train an EdgeTransformer using cross-entropy

Compute macro-F1 on the validation set

Save the best model’s weights to best_model.pt in out_dir

Using the Gateway Simulator

shield_can.gateway_sim.GatewaySimulator demonstrates how to combine feature extraction, model inference, and policy decisions in an online gateway-like loop.

Example
from shield_can.config import FeatureConfig, ModelConfig, PolicyConfig, SafetyConfig
from shield_can.gateway_sim import GatewaySimulator

# Configure safety-critical IDs and ECU mapping for your platform
safety_cfg = SafetyConfig(
    safety_critical_ids=[0x130, 0x1A0],  # example IDs
    id_to_ecu={
        0x130: "powertrain_ecu",
        0x1A0: "brake_ecu",
        # ...
    },
)

feat_cfg = FeatureConfig()
model_cfg = ModelConfig()
policy_cfg = PolicyConfig()

gateway = GatewaySimulator(
    feat_cfg=feat_cfg,
    model_cfg=model_cfg,
    policy_cfg=policy_cfg,
    safety_cfg=safety_cfg,
    num_classes=model_cfg.num_classes,
    device="cpu",
)

gateway.load_weights("runs/shield_can/best_model.pt")

# Process a single frame (e.g. from socketCAN or a log replay)
result = gateway.process_frame(
    t_s=0.123,              # timestamp in seconds
    can_id=0x130,
    dlc=8,
    payload=bytes([0, 0, 16, 0, 0, 0, 0, 0]),
    ecu="powertrain_ecu",
)

print(result)
# {
#   "forward": True/False,
#   "tier": 0..4,
#   "action": "log" | "shape" | "drop" | "quarantine_ecu:<id>" | "safe_mode",
#   "logits": [...],
#   "conf": float,
#   "predicted_class": int,
# }


In a real deployment, you would wire this into a CAN interface (e.g. via python-can / socketCAN) and enforce the policy’s forward / drop / shape decisions.
