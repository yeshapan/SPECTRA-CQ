# SPECTRA-CQ: Spectrogram Pattern Evaluation via Classical and Quantum Architectures

**SPECTRA-CQ** is a machine learning research repository dedicated to anomaly detection in Structural Health Monitoring (SHM). This project leverages data from the openLAB research bridge to evaluate and benchmark the performance of Classical Convolutional Autoencoders against Hybrid Quantum Autoencoders (HQAE) for identifying structural degradation.

## Overview
The core hypothesis of SPECTRA-CQ is that a **Variational Quantum Circuit (VQC)** acting as the latent bottleneck in an autoencoder architecture can learn highly entangled, robust representations of a "healthy" structural state. 
By training exclusively on baseline structural data, the system flags future damage as anomalies characterized by a spike in reconstruction error (Mean Squared Error).

This repository is structured to establish a rigorous classical baseline before introducing quantum integration.

## Dataset Characteristics
Data is sourced from the openLAB research bridge reference phase. Crucial dataset parameters include:
* **Triggered Windows:** The raw data consists of 70-second triggered acceleration measurements captured during periods of increased vibration activity.
* **Pre-filtered Signals:** The source data has been preprocessed by the dataset authors, including median subtraction and the application of a 4th-order Butterworth bandpass filter ($0.5\text{ Hz}$ to $100\text{ Hz}$).
* **Transformation:** 1D acceleration signals are converted into 2D time-frequency energy density representations using the **Continuous Wavelet Transform (CWT)**.

This dataset is available at [10.25532/OPARA-660](https://doi.org/10.25532/OPARA-660)

## Experimental Protocol
To ensure statistical stability and reproducibility, especially concerning the parameter initialization of the Variational Quantum Circuits, all training loops and benchmark evaluations adhere strictly to a **3-seed, 15-epoch** experimental setup.

## Repository Structure

```
SPECTRA-CQ/
├── configs/                        # Configuration management (multiple experiment variants)
│
├── docs/                           # Documentation
│   ├── ARCHITECTURE.md             # System architecture and design
│   ├── DATASET.md                  # Dataset characteristics
│   ├── OPENLAB_DATASET_README.md   # OpenLab dataset details
│   ├── PIPELINE.md                 # Data processing pipeline
│   └── RESULTS.md                  # Experiment results and tables
│
├── notebooks/                      # Exploration and execution dashboards
│   ├── phase1_foundation/          # Phase 1: EDA and Classical Baseline
│   │   ├── 01_eda_and_cwt.ipynb
│   │   └── 02_cae_baseline.ipynb
│   ├── phase2_barren_plaeaus/      # Phase 2: Barren Plateaus analysis
│   │   ├── 03a_ablation_topology.ipynb
│   │   └── 03b_ablation_depth.ipynb
│   └── phase3_quantum_regularization/ # Phase 3: Quantum Regularization models
│       ├── 04a_ablation_topology.ipynb
│       ├── 04b_ablation_depth.ipynb
│       └── 05_synthetic_anomaly_eval.ipynb
│
├── scripts/                        # Execution scripts
│   ├── evaluate_synthetic.py       # Evaluation script for synthetic anomalies
│   └── run_experiment.py           # Master script to execute models via YAML configs
│
├── src/                            # Core Source Code
│   ├── __init__.py
│   ├── data/                       # Data pipeline
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   ├── preprocess.py
│   │   └── synthetic.py            # Synthetic anomaly generation
│   │
│   ├── engine/                     # Training logic and reproducibility
│   │   ├── __init__.py
│   │   ├── evaluator.py            # Evaluation logic
│   │   ├── seed_control.py
│   │   └── trainer.py
│   │
│   ├── models/                     # Network architectures
│   │   ├── __init__.py
│   │   ├── classical/
│   │   │   ├── __init__.py
│   │   │   └── cae.py
│   │   ├── hybrid/
│   │   │   ├── __init__.py
│   │   │   ├── hqae.py
│   │   │   └── quantum_layer.py
│   │   └── quantum/
│   │       ├── __init__.py
│   │       ├── ansatz.py
│   │       ├── embedding.py
│   │       └── vqc.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── viz.py
│
├── study-notes/                    # Research and planning notes
│
├── .gitignore
├── README.md
└── requirements.txt                # Python dependencies (PyTorch, PennyLane, Qiskit)
```

* **Version Control & Development**: Authored in local IDE (Google Antigravity) and version controlled via GitHub
* **Execution Engine**: Google Colab Pro (GPU environment).
* **Data Storage**: All massive datasets (`openLAB` raw files, generated images) and trained model weights (`.pt` files) are hosted externally on Google Drive.
* **Frameworks**: PyTorch (Classical CNN) and PennyLane (Hybrid Quantum Neural Networks)