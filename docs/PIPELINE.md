# Data Processing Pipeline (ETL)

## Overview
This document outlines the data transformation pipeline that converts the raw 1D time-domain acceleration bursts from the OpenLAB dataset into 2D energy density matrices suitable for ingestion by Convolutional Autoencoders.

## Phase 1: Mathematical Transformation (`preprocess.py`)
To capture the frequency-domain features of structural degradation $\rightarrow$ the 1D signals are transformed using a Continuous Wavelet Transform (CWT).

* **Kernel Selection:** Complex Morlet Wavelet
* **Admissibility Parameter:** $\omega_0 = 6.0$ (Selected to symmetrically balance time-domain and frequency-domain resolution according to the Heisenberg-Gabor limit).
* **Frequency Mapping:** 64 logarithmically spaced scales bounding the 0.5 Hz to 100 Hz bandwidth.
* **Segmentation:** The continuous 70-second bursts are sliced into discrete 2-second, non-overlapping windows (1000 samples per window at 500 Hz).
* **Output Tensor:** The resulting matrices are cast to `float32` to optimize GPU VRAM and saved as `(64, 1000)` `.npy` binaries.

## Phase 2: PyTorch Ingestion (`dataloader.py`)
The pipeline utilizes a custom PyTorch `Dataset` to serve the `.npy` matrices to the GPU during training.

* **Memory Mapping:** Files are read dynamically from disk to prevent out-of-memory (OOM) crashes.
* **Sample Normalization:** Each 2D spectrogram undergoes Min-Max normalization. Energy values are tightly bounded between `[0, 1]` to ensure stable gradient descent during the model's reconstruction phase.
* **Unsupervised Targets:** As an autoencoder pipeline, the `__getitem__` method returns the normalized spectrogram as both the feature ($X$) and the target ($Y$) to compute the Mean Squared Error (MSE).
* **Determinism:** Data loading relies on a strict generator seed to ensure identical Train/Validation splits across the multi-seed experimental protocol.