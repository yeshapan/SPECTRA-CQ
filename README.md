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
├── configs/                        # Configuration management
│   ├── baseline_cae.yaml           # Config for the Classical Convolutional Autoencoder
│   └── baseline_hqae.yaml          # Config for the Hybrid Quantum Autoencoder
│
├── data/                           # Data directory
│   └── 2024-02-01_2024-10-31_ida_ki_export/
│
├── docs/                           # Documentation
│   ├── ARCHITECTURE.md             # System architecture and design
│   ├── DATASET.md                  # Dataset characteristics
│   └── PIPELINE.md                 # Data processing pipeline
│
├── notebooks/                      # Exploration and execution dashboards
│   ├── 01_eda_and_cwt.ipynb        # Visualizing raw 70-sec bursts and CWT spectrograms
│   ├── 02_cae_baseline.ipynb       # Colab execution wrapper for classical baseline
│   └── 03_hqae_baseline.ipynb      # Colab execution wrapper for hybrid quantum model
│
├── scripts/                        # Execution scripts
│   └── run_experiment.py           # Master script to execute models via YAML configs
│
├── src/                            # Core Source Code
│   ├── __init__.py
│   │
│   ├── data/                       # Data pipeline
│   │   ├── __init__.py
│   │   ├── dataloader.py           # PyTorch Dataset for loading spectrograms
│   │   └── preprocess.py           # Converts 70-sec CSV bursts to CWT .npy matrices
│   │
│   ├── engine/                     # Training logic and reproducibility
│   │   ├── __init__.py
│   │   ├── seed_control.py         # Enforces the strict 3-seed determinism protocol
│   │   └── trainer.py              # Standardized PyTorch training loop
│   │
│   ├── models/                     # Network architectures
│   │   ├── __init__.py
│   │   ├── classical/              # Classical baselines
│   │   │   ├── __init__.py
│   │   │   └── cae.py              # Classical Convolutional Autoencoder
│   │   ├── hybrid/                 # Hybrid Integration
│   │   │   ├── __init__.py
│   │   │   ├── hqae.py             # Connects Classical CNN encoder to VQC bottleneck
│   │   │   └── quantum_layer.py    # Quantum layer integration
│   │   └── quantum/                # Pure QML math
│   │       ├── __init__.py
│   │       ├── ansatz.py           # Variational Quantum Circuits
│   │       ├── embedding.py        # Data encoding strategies
│   │       └── vqc.py              # Variational Quantum Circuit definition
│   │
│   └── utils/                      # Helper modules
│       ├── __init__.py
│       └── viz.py                  # Scripts for generating visualizations
│
├── study-notes/                    # Research and planning notes
│   ├── 01_theoretical_justification.md
│   ├── 02_model_architectures.md
│   └── 03_implementation_plan.md
│
├── .gitignore
├── README.md
└── requirements.txt                # Python dependencies (PyTorch, PennyLane, Qiskit)
```

* **Version Control & Development**: Authored in local IDE (Google Antigravity) and version controlled via GitHub
* **Execution Engine**: Google Colab Pro (GPU environment).
* **Data Storage**: All massive datasets (`openLAB` raw files, generated images) and trained model weights (`.pt` files) are hosted externally on Google Drive.
* **Frameworks**: PyTorch (Classical CNN) and PennyLane (Hybrid Quantum Neural Networks)