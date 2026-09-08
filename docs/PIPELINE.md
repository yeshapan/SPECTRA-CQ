# Data Processing Pipeline (ETL)

## Overview
This document outlines the data transformation pipeline that converts raw 1D time-domain acceleration bursts from the OpenLAB 2024 Bridge dataset into 2D energy density matrices suitable for ingestion by Convolutional Autoencoders.
It traces the subsequent data flow through both classical and quantum neural architectures, detailing the transition from the unconstrained single-sensor baseline (`dim=8`) to the capacity-constrained multi-sensor spatial graph (`dim=2` per sensor).

## Stage 1: Mathematical Transformation (`preprocess.py`)
To capture the frequency-domain features of structural degradation, the 1D signals are transformed using a Continuous Wavelet Transform (CWT).

* **Kernel Selection:** Complex Morlet Wavelet
* **Admissibility Parameter:** $\omega_0 = 6.0$ (Selected to symmetrically balance time-domain and frequency-domain resolution according to the Heisenberg-Gabor limit).
* **Frequency Mapping:** 64 logarithmically spaced scales bounding the 0.5 Hz to 100 Hz bandwidth.
* **Segmentation:** The continuous 70-second bursts are sliced into discrete 2-second, non-overlapping windows (1000 samples per window at 500 Hz).
* **Spatial Synchronization (Phase 4):** For the multi-sensor spatial pipeline, this transformation is perfectly time-synchronized across 6 physical physical nodes (`PE11` through `PE23`), yielding a stacked `(6, 64, 1000)` spatial tensor.
* **Output Tensor:** The resulting matrices are cast to `float32` to optimize GPU VRAM and saved as `.npy` binaries.

## Stage 2: PyTorch Ingestion + I/O Optimization (`dataloader.py`)
The pipeline utilizes a custom PyTorch `Dataset` to serve the `.npy` matrices to the GPU during training.

* **I/O Bottleneck Mitigation (Colab NVMe):**
    * Streaming tens of thousands of `.npy` files directly from Google Drive causes severe GPU starvation (the GPU waits for network I/O).
    * The pipeline ingests a compressed `.zip` archive from Drive and extracts it directly to the Colab instance's local NVMe storage (`/content/local_spectra_data/`) before training begins.
* **Universal Tensor Alignment:**
    * The pipeline dynamically accommodates both Phase 3 and Phase 4 architectures by auto-injecting missing channel dimensions (`if x.dim() == 3: x = x.unsqueeze(1)`).
* **Sample Normalization:**
    * Each 2D spectrogram undergoes Min-Max normalization per matrix.
    * Energy values are tightly bounded between `[0, 1]` to ensure stable gradient descent during classical convolutional feature extraction.
* **Unsupervised Targets:**
    * As an autoencoder pipeline, the dataloader returns the normalized spectrogram as both the feature ($X$) and the target ($Y$) to compute the Mean Squared Error (MSE).
* **Determinism:**
    * Data loading relies on a strict generator seed (`seed = 42`) to ensure identical Train/Validation splits across the multi-seed experimental protocol.

## Stage 3: Classical Data Flow (CAE Pipeline)
The Classical Convolutional Autoencoder (CAE) processes the matrices once loaded onto the GPU using a Siamese weight-tied architecture.

1. **Siamese Folding:**
    * The batch and channel dimensions are folded (`B*C, 1, 64, 1000`) so all sensor matrices pass through identical spatial filters.
2. **Spatial Compression (Encoder):**
    * The tensor is passed through a deep stack of `Conv2d` and `MaxPool2d` layers, sequentially flattening the spatial matrix into 6,400 classical features per sensor.
3. **The Capacity Bottleneck (Divergence):**
    * A purely linear dense layer violently compresses the 6,400 features. 
    * **Phases 1, 2 and 3 (initial ablations):** Unconstrained representation $\rightarrow$ compressed to an 8-dimensional continuous vector (`dim=8`) for single-sensor (`PE11`) ablations.
    * **Phases 3 and 4:** Capacity-constrained baseline $\rightarrow$ severely suffocated down to a 2-dimensional continuous vector (`dim_per_sensor=2`).
4. **Spatial Reconstruction (Decoder):**
    * The latent vectors are projected back up to 6,400 features, passed through `ConvTranspose2d` layers, and utilize a `Sigmoid` activation to output the predicted bounded reconstruction.
5. **Siamese Unfolding:**
    * The tensor is unfolded back to its multi-sensor or single-sensor geometric shape `(B, C, 64, 1000)`.

## Stage 4: Hybrid Quantum Data Flow (HQAE Pipeline)
The HQAE pipeline intercepts the classical data flow at the bottleneck to execute the Variational Quantum Circuit (VQC).

1. **Classical Feature Extraction:** * Identical to Stage 3. The input is flattened and pre-compressed via a Siamese linear layer into either an 8-dimensional or 2-dimensional vector per sensor.
2. **Global Graph Assembly:**
    * Independent sensor vectors are flattened into a single global state to map to the quantum register.
    * *Phase 2 and 3 (initial ablations):* 8 qubits total $\leftarrow$ (1 sensor $\times$ 8 latents-per-sensor)
    * *Phase 3:* 2 qubits total $\leftarrow$ (1 sensor $\times$ 2 latents-per-sensor)
    * *Phase 4:* 12 qubits total $\leftarrow$ (6 sensors $\times$ 2 latents-per-sensor).
3. **Quantum Feature Scaling:** 
    * The continuous global vector is mathematically scaled via `tanh(x) * pi` to strictly fit within a $[-\pi, \pi]$ bound to ensure deterministic angle embedding without Bloch sphere overlap.
4. **The Device Bridge (GPU $\rightarrow$ CPU):** 
    * Because current PennyLane state-vector simulators (`default.qubit`) cannot reliably execute heavily entangled circuits natively on PyTorch CUDA tensors, the pipeline detaches the bounded tensor and moves it to the CPU.
5. **The Quantum Bottleneck (VQC):**
    * **State Preparation:** The scaled features are mapped into the Hilbert space via Angle Embedding ($R_x$).
    * **Ansatz:** The state is manipulated via trainable rotations ($R_y, R_z$) and entanglement configurations (e.g., `none`, `strong`, `bridge_soft`).
    * **Measurement:** The circuit collapses, returning continuous expectation values (Pauli-Z) bounded between `[-1, 1]`.
6. **The Device Bridge (CPU $\rightarrow$ GPU):** * The resulting quantum tensor is pushed back to the original CUDA device to resume classical processing.
7. **Graph Deconstruction & Reconstruction:** * The global quantum state is sliced back into isolated physical sensor representations (e.g., from `(B, 12)` back to `(B*C, 2)`). The features are then upsampled back into the `(B, C, 64, 1000)` space identically to Stage 3.

## Stage 5: Evaluation and Synthetic Degradation (`synthetic.py` and `evaluator.py`)
During inference, a specialized pipeline generates parallel degradation matrices to evaluate model sensitivity and spatial fusion efficacy.

* **The Incipient Anomaly Proxy:**
    * Unseen healthy baseline matrices are ingested and normalized.
    * A localized Acoustic Emission (AE) proxy (a high-frequency energy burst with an amplitude of `0.35`) is synthetically injected strictly into the `PE11` sensor channel.
    * Adjacent sensors (`PE12` through `PE23`) are left perfectly unperturbed to simulate an infinitely steep spatial gradient.
* **Target Node Isolation:**
    * During inference, the `AnomalyEvaluator` computes the Mean Squared Error (MSE) strictly on the `PE11` target node and intentionally discards the reconstruction error of the adjacent 5 healthy sensors.
    * **Purpose:**
        * This guarantees a mathematically identical (apples-to-apples) comparison between Phase 3 (single-sensor) and Phase 4 (multi-sensor graph) models
        * It helps to directly measure whether geometric spatial context lowered the false-positive rate of the targeted anomaly. 
* **Scoring:** Both healthy and degraded tensors are passed through the frozen evaluators to compute absolute reconstruction error and generate comparative AUC-ROC curves.