# SHIELD-CAN: A Self-Healing Edge-AI Intrusion Detection System for Automotive CAN Gateways

This repository provides the reference implementation of **SHIELD-CAN**, a self-healing edge-AI intrusion detection and response system for automotive CAN gateways.

Modern vehicles comprise a large number of interconnected Electronic Control Units (ECUs) communicating over the legacy Controller Area Network (CAN) bus, which remains vulnerable to cyber–physical attacks that may compromise both safety and availability. Deploying machine learning–based intrusion detection systems (IDS) directly on in-vehicle gateways can provide timely, context-aware detection, but also imposes strict constraints on latency, memory, interpretability, and safe interaction with safety-critical workloads. Detection outputs must be mapped to well-specified mitigation actions whose impact on the system’s safety and performance can be analysed.

**SHIELD-CAN** addresses these challenges by combining:

- A **streaming, protocol-agnostic feature-extraction pipeline** over raw CAN frames.
- A **compact encoder-only Transformer** operating on short windows of traffic statistics, designed for edge deployment.
- A **deterministic, multi-tier self-healing policy** that translates anomaly scores and class predictions into gateway actions such as rate limiting, selective dropping, and ECU quarantine, while enforcing explicit safety invariants.

---
## Key features

At a high level, SHIELD-CAN consists of three tightly coupled components:

1. **Streaming Feature Extractor**  
   - Processes raw CAN frames in chronological order.  
   - Maintains **O(1)** state per active CAN ID and computes lightweight traffic statistics, including timing and payload entropy, local inter-arrival dynamics (e.g. Kalman-style residuals), DLC drift and byte-level toggling behaviour.  
   - Produces a fixed-length feature vector per frame, independent of OEM-specific payload semantics.

2. **Edge Transformer Model (`EdgeTransformer`)**  
   - Encoder-only Transformer applied to short windows of feature vectors.  
   - Uses a single STAT token for window-level classification.  
   - Designed with static shapes, moderate depth and width to support **8-bit quantisation** and efficient inference on ARM-based gateway hardware.

3. **Self-Healing Policy and Gateway Loop**  
   - Maps model outputs (logits, anomaly scores, class labels) into a **multi-tier response**:
     - Tier 0: monitoring and logging  
     - Tier 1: traffic shaping / rate limiting  
     - Tier 2: selective frame dropping  
     - Tier 3: ECU-level quarantine  
     - Tier 4: optional gateway safe mode  
   - Enforces **safety invariants**, e.g. safety-critical frame IDs may never be dropped, and degraded modes are bounded in duration and severity.
   - Can be integrated at an in-vehicle gateway to enforce decisions in real time.

---
## Architecture

If you have an architecture diagram in `Images/Shield-CAN-arch.png`, you can render it as:

<p align="center">
  <img src="Images/Shield-CAN-arch.png" alt="SHIELD-CAN architecture" width="500">
</p>

---

## Datasets

You can download the example CAN intrusion datasets used with this code from:

- [Car Hacking Dataset](https://www.kaggle.com/datasets/pranavjha24/car-hacking-dataset)
- [IVN-IDS Dataset](https://www.kaggle.com/datasets/daksh0511/ivn-ids/code)

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── setup.py
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

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```
### 3. Install dependencies

Using `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Or install as an editable package:

```bash
pip install -e.
```

**Python version**: 3.9+ is recommended.

## CAN Log Format

The implementation assumes CAN logs in a CSV format similar to:

```bash
timestamp,id,dlc,data0,data1,data2,data3,data4,data5,data6,data7,label
0.000000,0x130,8,0,0,0,0,0,0,0,0,Normal
0.010000,0x130,8,0,0,16,0,0,0,0,0,Normal
0.020000,0x130,8,0,0,32,0,0,0,0,0,Normal
0.030000,0x1A0,8,10,20,30,40,50,60,70,80,DoS
```

**Expected columns**:

  - `timestamp` – float (seconds). If in ms, it will be converted internally.
  - `id` – CAN ID as integer or hex string (e.g. 0x130).
  - `dlc` – data length code (0–8 for classical CAN).
  - `data0 .. data7` – payload bytes (0–255); only the first dlc are used.
  - `label` – class label (string or int). Example label mapping (default in train.py):

```bash
{
    "Normal": 0,
    "DoS": 1,
    "Fuzzy": 2,
    "Malfunction": 3,
    "Spoof": 4,
}
```

You can adjust the label mapping in `train.py` to match your dataset.

## Training the Transformer

The main training entrypoint is `train.py`.

```bash
Basic usage
python train.py \
  --train_csv data/car_hacking_train.csv \
  --val_csv data/car_hacking_val.csv \
  --out_dir runs/shield_can \
  --epochs 30 \
  --batch_size 128 \
  --lr 3e-4 \
  --device cuda
```

## Configuration

Key configuration classes are defined in shield_can/config.py:

**FeatureConfig**: Timing histogram size, Entropy half-lives, Kalman parameters, Toggling half-life.

**ModelConfig**: Feature dimensionality, Transformer depth, Number of heads, Window size, Number of classes, Dropout

**TrainingConfig**: Batch size, Number of epochs, Learning rate, Device

**PolicyConfig**: Thresholds and time constants for tiered responses, Shaping (rate limiting), Dropping, ECU quarantine, Safe mode

**SafetyConfig**
  - `safety_critical_ids`: CAN IDs that must never be dropped.
  - `id_to_ecu`: mapping from CAN IDs to ECU names used by the self-healing policy.



## Related Publications

Below are selected publications related to SHIELD-CAN, cyber-physical security, additive manufacturing security, threat intelligence, and trusted AI systems. These give additional background and context around secure architectures, intrusion detection, self-healing, and risk assessment.

### Automotive, VANET & Cyber-Physical / ICS Security

- **[Secure and Anonymous Batch Authentication and Key Exchange Protocols for 6G Enabled VANETs](https://ieeexplore.ieee.org/document/10972137/)**  
  **Mahender Kumar** and Carsten Maple
  *IEEE Transactions on Intelligent Transportation Systems, 2025*

- **[ICSThreatQA: A Knowledge-Graph Enhanced Question Answering Model for Industrial Control System Threat Intelligence](https://www.sciencedirect.com/science/article/abs/pii/S0957417425037959)**  
  Ruby Rani, **Mahender Kumar**, Gregory Epiphaniou and Carsten Maple  
  *Expert Systems with Applications, 2025*.  

- **[Securing connected and autonomous vehicles: analysing attack methods, mitigation strategies, and the role of large language models](https://digital-library.theiet.org/doi/abs/10.1049/icp.2024.2534)**  
  **Mahender Kumar**, Ruby Rani, Gregory Epiphaniou, Carsten Maple
  *IET Conference Proceedings, 2024*.  

### Additive Manufacturing & Industrial Security

- **[Securing additive manufacturing with blockchain-based cryptographic anchoring and dual-lock integrity auditing](https://www.sciencedirect.com/science/article/pii/S0166361525001605)**  
  **Mahender Kumar**, Gregory Epiphaniou, Carsten Maple 
  *Computers in Industry, 173 (2025): 104395*.  

- **[Security of cyber-physical Additive Manufacturing supply chain: Survey, attack taxonomy and solutions](https://www.sciencedirect.com/science/article/pii/S0167404825002469)**  
  **Mahender Kumar**, Gregory Epiphaniou, Carsten Maple
  *Computers & Security, 2025: 104557*.  

- **[SPM-SeCTIS: Severity Pattern Matching for Secure Computable Threat Information Sharing in Intelligent Additive Manufacturing](id.elsevier.com/as/authorization.oauth2?platSite=SD%2Fscience&additionalPlatSites=GH%2Fgeneralhospital%2CMDY%2Fmendeley%2CSC%2Fscopus%2CRX%2Freaxys&scope=openid%20email%20profile%20els_auth_info%20els_idp_info%20els_idp_analytics_attrs%20urn%3Acom%3Aelsevier%3Aidp%3Apolicy%3Aproduct%3Ainst_assoc&response_type=code&redirect_uri=https%3A%2F%2Fwww.sciencedirect.com%2Fuser%2Fidentity%2Flanding&authType=SINGLE_SIGN_IN&prompt=none&client_id=SDFE-v4&state=retryCounter%3D0%26csrfToken%3De0f7da65-4312-406d-b8b1-5d0077a91e35%26idpPolicy%3Durn%253Acom%253Aelsevier%253Aidp%253Apolicy%253Aproduct%253Ainst_assoc%26returnUrl%3D%252Fscience%252Farticle%252Fpii%252FS2542660524002750%26prompt%3Dnone%26cid%3Darp-3211915e-c7b0-4f78-a79d-8fc3c10269d1)**
  **Mahender Kumar, Gregory Epiphaniou, Carsten Maple**  
  *Internet of Things, 28 (2024): 101334*.  

- **[Comprehensive threat analysis in additive manufacturing supply chain: a hybrid qualitative and quantitative risk assessment framework](https://link.springer.com/article/10.1007/s11740-024-01283-1)**  
  **Mahender Kumar**, Gregory Epiphaniou, Carsten Maple
  *Production Engineering, 18(6): 955–973, 2024*.  

- **[Leveraging Semantic Relationships to Prioritise Indicators of Compromise in Additive Manufacturing Systems](https://link.springer.com/chapter/10.1007/978-3-031-41181-6_18)**  
  **Mahender Kumar**, Gregory Epiphaniou, Carsten Maple
  *International Conference on Applied Cryptography and Network Security, 2023*. 



### Healthcare, IoMT & Blockchain Systems

- **[A Provable Secure and Lightweight Smart Healthcare Cyber-Physical System With Public Verifiability](https://ieeexplore.ieee.org/document/9624169)**  
  **Mahender Kumar**, Satish Chand
  *IEEE Systems Journal, 16(4): 5501–5508, 2022*.  

- **[A Secure and Efficient Cloud-Centric Internet-of-Medical-Things-Enabled Smart Healthcare System With Public Verifiability](https://ieeexplore.ieee.org/document/9131770)**  
  **Mahender Kumar**, Satish Chand
  *IEEE Internet of Things Journal, 7(10): 10457–10465, 2020*.  

- **[MedHypChain: A patient-centered interoperability hyperledger-based medical healthcare system: Regulation in COVID-19 pandemic](https://www.sciencedirect.com/science/article/pii/S1084804521000023)**  
  **Mahender Kumar**, Satish Chand
  *Journal of Network and Computer Applications, 179:102975, 2021*.  

- **[A Lightweight Cloud-Assisted Identity-Based Anonymous Authentication and Key Agreement Protocol for Secure Wireless Body Area Network](https://ieeexplore.ieee.org/document/9099043)**  
  **Mahender Kumar**, Satish Chand
  *IEEE Systems Journal, 15(2): 1646–1657, 2021*.  


### Cryptography, Ontologies & Threat Intelligence Foundations

- **[Science and Technology Ontology: A Taxonomy of Emerging Topics](https://arxiv.org/abs/2305.04055)**  
  **Mahender Kumar**, Ruby Rani, Mirko Botarelli, Gregory Epiphaniou, Carsten Maple
  *arXiv preprint arXiv:2305.04055, 2023*.  

- **[Pairing for Greenhorn: Survey and Future Perspective](https://arxiv.org/abs/2108.12392)**  
  **Mahender Kumar**, Satish Chand
  arXiv preprint, 2021.  

For a complete and up-to-date list of publications, please refer to my full publication list or Google Scholar profile. [Google scholar](https://scholar.google.com/citations?hl=en&user=Ppmct6EAAAAJ&view_op=list_works&sortby=pubdate)

---

## Citing

If you use SHIELD-CAN framework, please cite the corresponding work:

```bash
@article{kumar2025shieldcan,
  title   = {SHIELD-CAN: A Self-Healing Edge-AI Intrusion Detection System for Automotive CAN Gateways},
  author  = {Mahender Kumar and Gregory Epiphaniou and Carsten Maple},
  journal = {Preprint},
  year    = {2025}
}
```

## Contributor 
  - [Mahender Kumar](https://scholar.google.com/citations?user=Ppmct6EAAAAJ&hl=en)
  - [Gregory Epiphaniou](https://warwick.ac.uk/fac/sci/wmg/about/our-people/profile/?wmgid=2175)
  - [Carsten Maple](https://warwick.ac.uk/fac/sci/wmg/about/our-people/profile/?wmgid=1102)
