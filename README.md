# SHIELD-CAN: A Self-Healing Edge-AI Intrusion Detection System for Automotive CAN Gateways


Modern vehicles rely on many interconnected Electronic Control Units (ECUs) that communicate over the legacy Controller Area Network (CAN) bus, which is still vulnerable to cyber-physical attacks that can impact both safety and availability. Deploying machine learning-based intrusion detection systems (IDS) directly on in-vehicle gateways can improve protection, but also introduces tight constraints on latency, memory footprint, and safe interaction with critical workloads. It also requires that detection outputs be translated into clear, analysable mitigation actions.

**SHIELD-CAN** is a self-healing edge-AI intrusion detection and response system designed specifically for automotive CAN gateways and dependability:

- Combines a streaming, protocol-agnostic feature pipeline with a compact encoder-only Transformer operating on short windows of traffic statistics.
- Supports `O(1)` per-frame feature updates and sub-millisecond inference on embedded ARM hardware using 8-bit quantisation.
- Uses a deterministic, multi-tier self-healing policy that maps anomaly scores and class predictions into:
  - rate limiting,
  - selective dropping, and
  - ECU-level quarantine,
  while enforcing safety invariants that prevent interference with safety-critical messages and bound degraded modes.


<p align="center">
  <img src="Images/Shield-CAN-arch.png" alt="SHIELD-CAN architecture" width="500">
</p>


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
 
## Datasets

You can download the example CAN intrusion datasets used with this code from:

- [Car Hacking Dataset](https://www.kaggle.com/datasets/pranavjha24/car-hacking-dataset)
- [IVN-IDS Dataset](https://www.kaggle.com/datasets/daksh0511/ivn-ids/code)

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
pip install -e .
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

**FeatureConfig**
  - Timing histogram size
  - Entropy half-lives
  - Kalman parameters
  - Toggling half-life, etc.

**ModelConfig**
  - Feature dimensionality
  - Transformer depth
  - Number of heads
  - Window size
  - Number of classes
  - Dropout

**TrainingConfig**
  - Batch size
  - Number of epochs
  - Learning rate
  - Device

**PolicyConfig**
  - Thresholds and time constants for tiered responses
  - Shaping (rate limiting)
  - Dropping
  - ECU quarantine
  - Safe mode

**SafetyConfig**
  - `safety_critical_ids`: CAN IDs that must never be dropped.
  - `id_to_ecu`: mapping from CAN IDs to ECU names used by the self-healing policy.


## Related Publications

Below are selected publications related to SHIELD-CAN, cyber-physical security, and trusted AI systems. These give additional background and context around secure architectures, intrusion detection, and self-healing systems.

### Automotive & Cyber-Physical Security

- **Securing connected and autonomous vehicles: analysing attack methods, mitigation strategies, and the role of large language models**  
  *Mahender Kumar, Ruby Rani, Gregory Epiphaniou, Carsten Maple*  
  IET Conference Proceedings, 2024.  
  [Publisher link](https://digital-library.theiet.org/doi/10.1049/icp.2024.2534) :contentReference[oaicite:0]{index=0}  

- **Leveraging Semantic Relationships to Prioritise Indicators of Compromise in Additive Manufacturing Systems**  
  *Mahender Kumar, Gregory Epiphaniou, Carsten Maple*  
  Lecture Notes in Computer Science (ACNS Workshops), 2023.  
  [Springer link](https://link.springer.com/chapter/10.1007/978-3-031-41181-6_18) • [arXiv preprint](https://arxiv.org/abs/2305.04102) :contentReference[oaicite:1]{index=1}  

- **A Resilient Cyber-Physical Demand Forecasting System for Critical Infrastructures against Stealthy False Data Injection Attacks**  
  *Mahender Kumar et al.*  
  (Cyber-physical resilience / critical infrastructure security.)  
  [ResearchGate](https://www.researchgate.net/publication/364257202_A_Resilient_Cyber-Physical_Demand_Forecasting_System_for_Critical_Infrastructures_against_Stealthy_False_Data_Injection_Attacks) :contentReference[oaicite:2]{index=2}  


### Secure Healthcare, IoMT & Blockchain

- **A Provable Secure and Lightweight Smart Healthcare Cyber-Physical System With Public Verifiability**  
  *Mahender Kumar, Satish Chand*  
  IEEE Systems Journal, 16(4): 5501–5508, 2022.  
  [IEEE Xplore](https://ieeexplore.ieee.org/document/9624169) :contentReference[oaicite:3]{index=3}  

- **A Secure and Efficient Cloud-Centric Internet-of-Medical-Things-Enabled Smart Healthcare System With Public Verifiability**  
  *Mahender Kumar, Satish Chand*  
  IEEE Internet of Things Journal, 7(10): 10457–10465, 2020.  
  [IEEE Xplore](https://doi.org/10.1109/JIOT.2020.3006523) :contentReference[oaicite:4]{index=4}  

- **MedHypChain: A patient-centered interoperability hyperledger-based medical healthcare system: Regulation in COVID-19 pandemic**  
  *Mahender Kumar, Satish Chand*  
  Journal of Network and Computer Applications, 179:102975, 2021.  
  [ScienceDirect](https://doi.org/10.1016/j.jnca.2021.102975) :contentReference[oaicite:5]{index=5}  

- **A Lightweight Cloud-Assisted Identity-Based Anonymous Authentication and Key Agreement Protocol for Secure Wireless Body Area Network**  
  *Mahender Kumar, Satish Chand*  
  IEEE Systems Journal, 15(2): 1646–1657, 2021.  
  [IEEE Xplore](https://ieeexplore.ieee.org/document/9099043) :contentReference[oaicite:6]{index=6}  

- **A Lightweight Cloud-Assisted Identity-Based Anonymous Authentication and Key Agreement Protocol for Secure Wireless Body Area Network**  
  (Arising note / follow-up work referenced in later literature.)  
  [Overview via Semantic Scholar](https://www.semanticscholar.org/paper/2a32090025fe6275abe2acb58cb26a51c897aedb) :contentReference[oaicite:7]{index=7}  


### Cryptography & Privacy-Preserving Systems

- **Pairing for Greenhorn: Survey and Future Perspective**  
  *Mahender Kumar, Satish Chand*  
  arXiv preprint, 2021.  
  [arXiv](https://arxiv.org/abs/2108.12392) :contentReference[oaicite:8]{index=8}  

- **A Provable Secure and Lightweight Smart Healthcare Cyber-Physical System With Public Verifiability**  
  (See Systems Journal entry above; also discussed in multiple survey and follow-up works.) :contentReference[oaicite:9]{index=9}  

- **A patient-centered interoperability hyperledger-based medical healthcare system: Regulation in Covid-19 pandemic**  
  (MedHypChain; frequently cited in blockchain / privacy preserving healthcare frameworks.) :contentReference[oaicite:10]{index=10}  


### Additive Manufacturing & Industrial Security

- **Securing Additive Manufacturing Systems: A Threat-Centric Risk Assessment Framework**  
  *Mahender Kumar, Gregory Epiphaniou, Carsten Maple*  
  (Communicated; threat-centric risk assessment for AM systems.) – link to be added when publicly available. :contentReference[oaicite:11]{index=11}  

- **Blockchain-Based G-Code Protection with Physical-to-Digital Cryptographic Anchor in Additive Manufacturing System**  
  *Mahender Kumar, Gregory Epiphaniou, Carsten Maple*  
  (Communicated.) – link to be added once published. :contentReference[oaicite:12]{index=12}  


---

For a complete and always-up-to-date list of my publications, see:  

- **Full publication list:** [https://mahend72.github.io/Biography/Publication.html](https://mahend72.github.io/Biography/Publication.html) :contentReference[oaicite:13]{index=13}  



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
