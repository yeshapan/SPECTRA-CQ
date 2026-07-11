# SPECTRA-CQ: Spectrogram Pattern Evaluation via Classical and Quantum Architectures

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-000000?style=for-the-badge&logo=penny-lane&logoColor=white)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)

**SPECTRA-CQ** is a machine learning research repository dedicated to anomaly detection in Structural Health Monitoring (SHM) for bridge.

This project leverages data from the openLAB research bridge to evaluate and benchmark the performance of Classical Convolutional Autoencoders against Hybrid Quantum Autoencoders (HQAE) for identifying structural degradation.

## Overview
The core hypothesis of SPECTRA-CQ is that a Variational Quantum Circuit (VQC) acting as the latent bottleneck in an autoencoder architecture can learn highly entangled, robust representations of a "healthy" structural state. By training exclusively on baseline structural data, the system flags future damage as anomalies characterized by a spike in reconstruction error (Mean Squared Error).

As the project scales from single-sensor evaluations to multi-sensor spatial fusion, it investigates a critical intersection in modern machine learning: **Geometric Spatial Context vs. NISQ-era Quantum Scaling (Barren Plateaus).**

---

## Dataset Characteristics
Data is sourced from the openLAB research bridge reference phase. Crucial dataset parameters include:
* **Triggered Windows:** The raw data consists of 70-second triggered acceleration measurements captured during periods of increased vibration activity.
* **Pre-filtered Signals:** The source data has been preprocessed by the dataset authors, including median subtraction and the application of a 4th-order Butterworth bandpass filter ($0.5\text{ Hz}$ to $100\text{ Hz}$).
* **Transformation:** 1D acceleration signals are converted into 2D time-frequency energy density representations (`(64, 1000)` matrices) using the **Continuous Wavelet Transform (CWT)**.

This dataset is available at [10.25532/OPARA-660](https://doi.org/10.25532/OPARA-660)

---

## Experimental Protocol
To ensure statistical stability and reproducibility, all training loops and benchmark evaluations adhere strictly to a rigorous evaluation methodology:
* **Seed Locking:** A strictly enforced **3-seed** initialization protocol ensures reproducibility across classical PyTorch and highly stochastic PennyLane quantum state-vector simulators.
* **Epoch Restraint:** Training is locked to **15 epochs** to mitigate multi-channel convolutional overfitting and account for rapid quantum convergence.
* **The Compression Bottleneck:** Multi-sensor spatial models are strictly constrained to a `dim=2` per-sensor capacity bottleneck, enforcing a 1-to-1 parity between classical latent spaces and the maximum representational capacity of a 2-qubit register.
* **Incipient Anomaly Proxy:** Evaluated against an Acoustic Emission burst (Amplitude = $0.35$) using Target Node Isolation to mathematically simulate early-stage micro-cracking.

---

## Repository Structure

```
SPECTRA-CQ/
├── configs/                            # Configuration management (multiple experiment variants)
│
├── docs/                               # Project Documentation
│   ├── ARCHITECTURE.md                 # System architecture and tensor dimensional flow
│   ├── DATASET.md                      # Dataset characteristics
│   ├── IMPLEMENTATION.md               # Barren plateau mitigation, topology bounds and constraints
│   ├── OPENLAB_DATASET_README.md       # Official OpenLAB Bridge Dataset README (added for reference)
│   ├── PIPELINE.md                     # Data processing pipeline
│   └── RESULTS.md                      # Experiment results, observations and inferences
│
├── notebooks/                          # Jupyter Notebooks for execution
│   ├── phase1_foundation/              # Phase 1: EDA + Classical Baseline (single sensor) (dim=8)
│   │  
│   ├── phase2_barren_plaeaus/          # Phase 2: HQAE Ablation Optimization Failure (single sensor) (dim=8) 
│   │   
│   ├── phase3_quantum_regularization/  # Phase 3: Barren Plateau Mitigation (dim=8) -> Sensor Bottleneck (dim=2) -> Synthetic Anomaly Evaluation (For both dim=2 and dim=8)
│   │ 
│   └── phase4_multi_sensor_fusion/     # Phase 4: 6-Sensor Spatial Graph
│      
│
├── scripts/                            # Execution scripts
│   ├── evaluate_synthetic.py           # Evaluation script for synthetic anomalies
│   └── run_experiment.py               # Master script to execute models via YAML configs
│
├── src/                                # Core Source Code
│   ├── __init__.py
│   ├── data/                           # Data pipeline for structural signal processing + degradation simulation
│   │   ├── __init__.py
│   │   ├── dataloader.py               # Siamese folding and universal dimension injection
│   │   ├── preprocess.py               # CWT mathematical transformations
│   │   └── synthetic.py                # Incipient damage proxy (Acoustic Emission micro-crack injection)
│   │
│   ├── engine/                         # Training logic and reproducibility
│   │   ├── __init__.py
│   │   ├── evaluator.py                # Evaluation logic (Target Node Isolation)
│   │   ├── seed_control.py             # Global determinism locks
│   │   └── trainer.py                  # Gradient clipping and Adam optimization
│   │
│   ├── models/                         # Network architectures
│   │   ├── __init__.py
│   │   ├── classical/
│   │   │   ├── __init__.py
│   │   │   └── cae.py                  # Convolutional Autoencoder (Classical Baseline)
│   │   ├── hybrid/
│   │   │   ├── __init__.py
│   │   │   ├── hqae.py                 # PyTorch-to-PennyLane Hybrid Wrapper
│   │   │   └── quantum_layer.py        # VQCTorchLayer and CPU/GPU State-Vector computational bridge
│   │   └── quantum/
│   │       ├── __init__.py
│   │       ├── ansatz.py               # Topologies (none, basic, strong, bridge_soft, bridge_hard)
│   │       ├── embedding.py            # Amplitude-to-Angle mathematical mapping
│   │       └── vqc.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── viz.py                      # Standardized visualization suite
│
├── study-notes/                        # Research and planning notes
│
├── .gitignore
├── README.md
└── requirements.txt                    # Python dependencies (PyTorch, PennyLane, Qiskit)
```
---

## Development Environment
* Version Control + Development: Authored in local IDE (Google Antigravity) and version controlled via GitHub.
* Execution Engine: Google Colab (T4 GPU environment) utilizing dynamic CPU offloading for PennyLane state-vector simulators.
* Data Storage: All massive datasets (openLAB raw .npy files) and trained model weights (.pth checkpoints) were hosted externally on Google Drive and imported via NVMe compressed zip streaming.
* Frameworks: PyTorch (Classical CNNs and Optimizers) and PennyLane (Hybrid Quantum Neural Networks).

---

## Key Research Findings
* **Single-Sensor Spatial Blindness:** Isolated sensors mathematically lack the spatial context to distinguish between macroscopic environmental noise and localized incipient micro-cracking (Acoustic Emissions) under a strict representation bottleneck (capacity of $2$).
* **The Classical Spatial Advantage:** By expanding the architecture to a 6-sensor spatial graph, classical continuous networks successfully utilize the "geometric peer pressure" of healthy adjacent nodes to significantly boost localized anomaly detection.
* **The Quantum Scaling Wall:** While 2-qubit quantum models achieve parity with classical models, scaling all-to-all entanglement to a 12-qubit spatial graph triggers catastrophic gradient collapse (Barren Plateaus). This proves that forced global entanglement limits current structural quantum computing applications.
* **The PINN Paradox:** Physics-Informed Neural Networks (`bridge_soft`) explicitly penalize non-physical wave states. When evaluated on localized synthetic data that disobeys continuous elastic wave propagation, the PINN correctly "rejects" the data. This paradox highlights the necessity of real-world physical load testing for physics-constrained architectures.