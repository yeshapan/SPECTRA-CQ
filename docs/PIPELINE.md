# Data Processing Pipeline (ETL)

## Overview
This document outlines the data transformation pipeline that converts the raw 1D time-domain acceleration bursts from the OpenLAB dataset into 2D energy density matrices suitable for ingestion by Convolutional Autoencoders, and traces the subsequent data flow through both classical and quantum neural architectures.

## Stage 1: Mathematical Transformation (`preprocess.py`)
To capture the frequency-domain features of structural degradation $\rightarrow$ the 1D signals are transformed using a Continuous Wavelet Transform (CWT).

* **Kernel Selection:** Complex Morlet Wavelet
* **Admissibility Parameter:** $\omega_0 = 6.0$ (Selected to symmetrically balance time-domain and frequency-domain resolution according to the Heisenberg-Gabor limit).
* **Frequency Mapping:** 64 logarithmically spaced scales bounding the 0.5 Hz to 100 Hz bandwidth.
* **Segmentation:** The continuous 70-second bursts are sliced into discrete 2-second, non-overlapping windows (1000 samples per window at 500 Hz).
* **Output Tensor:** The resulting matrices are cast to `float32` to optimize GPU VRAM and saved as `(64, 1000)` `.npy` binaries.

## Stage 2: PyTorch Ingestion + I/O Optimization (`dataloader.py`)
The pipeline utilizes a custom PyTorch `Dataset` to serve the `.npy` matrices to the GPU during training.

* **I/O Bottleneck Mitigation (Colab NVMe):**
    * Streaming tens of thousands of `.npy` files directly from Google Drive causes severe GPU starvation (the GPU waits for network I/O)
    * So the pipeline ingests a compressed `spectra_matrices.zip` from Drive and extracts it directly to the Colab instance's local NVMe storage (`/content/local_spectra_data/`) before training begins.
* **Sample Normalization:**
    * Each 2D spectrogram undergoes Min-Max normalization.
    * Energy values are tightly bounded between `[0, 1]` to ensure stable gradient descent during the classical convolutional feature extraction.
* **Unsupervised Targets:**
    * As an autoencoder pipeline, the `__getitem__` method returns the normalized spectrogram as both the feature ($X$) and the target ($Y$) to compute the Mean Squared Error (MSE).
* **Determinism:**
    * Data loading relies on a strict generator seed (seed = 42) to ensure identical Train/Validation splits across the multi-seed experimental protocol.

## Stage 3: Classical Data Flow (CAE Pipeline)
Classical Convolutional Autoencoder (CAE) processes the matrices once the data is loaded onto the GPU:

1. **Spatial Compression (Encoder):**
    * The `(1, 64, 1000)` tensor is passed through a deep stack of `Conv2d` and `MaxPool2d` layers
    * This dynamically flattens the tensor into a 6,400-dimensional vector.
2. **The Classical Bottleneck:**
    * A purely linear dense layer (`nn.Linear`) violently compresses the 6,400 features into an 8-dimensional continuous latent space representation.
3. **Spatial Reconstruction (Decoder):**
    * The 8-dimensional vector is projected back up and passed through `ConvTranspose2d` layers + utilizes a `Sigmoid` activation at the terminal layer to output the predicted `(1, 64, 1000)` bounded reconstruction.

## Stage 4: Hybrid Quantum Data Flow (HQAE Pipeline)
The HQAE pipeline intercepts the classical data flow at the bottleneck to execute the Variational Quantum Circuit (VQC).

1. **Spatial Compression (Encoder):** 
    * Identical to Stage 3.
    * The `(1, 64, 1000)` tensor is flattened and pre-compressed via a linear layer into an 8-dimensional continuous vector.
2. **The Device Bridge (GPU $\rightarrow$ CPU):** 
    * Current PennyLane state-vector simulators (`default.qubit`) cannot reliably execute highly entangled circuits natively on PyTorch CUDA tensors without crashing.
    * The pipeline actively intercepts the 8-dimensional tensor, detaches it, and moves it to the CPU.
3. **Quantum Feature Scaling:** 
    * The 8-dimensional vector (currently unbounded due to classical dense layer processing) is mathematically scaled to strictly fit within a $[-\pi, \pi]$ bound. 
    * This is required for stable angular rotation mapping.
4. **The Quantum Bottleneck (VQC):**
    * **State Preparation:** The scaled classical features are mapped into the Hilbert space via Angle Embedding ($R_y$).
    * **Ansatz:** The state is manipulated via trainable rotations and CNOT entanglements (configurable topologies).
    * **Measurement:** The circuit collapses, returning 8 continuous expectation values (Pauli-Z) bounded between `[-1, 1]`.
5. **The Device Bridge (CPU $\rightarrow$ GPU):** 
    * The resulting 8-dimensional tensor is pushed back to the original CUDA device to resume classical processing.
6. **Spatial Reconstruction (Decoder):** 
    * Identical to Stage 3.
    * The quantum-processed features are upsampled back into the `(1, 64, 1000)` space.

## Stage 5: Evaluation & Synthetic Degradation (`synthetic.py`)
During inference, a specialized pipeline generates parallel degradation matrices to evaluate model sensitivity.
* Unseen healthy baseline matrices are ingested and normalized.
* A controlled structural degradation function injects target parameters (e.g., 2% Gaussian noise and localized frequency band masking) to simulate incipient structural fatigue.
* Both the healthy and degraded tensors are passed through the frozen evaluators to compute absolute reconstruction error (MSE) and generate comparative AUC-ROC curves.