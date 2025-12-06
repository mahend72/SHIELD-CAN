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

