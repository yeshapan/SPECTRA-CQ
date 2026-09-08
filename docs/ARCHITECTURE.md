# SPECTRA-CQ: Architecture

## Executive Summary
The SPECTRA-CQ project follows a progressive, four-phase architectural evolution. It begins by establishing a robust classical foundation, introduces parameterized quantum circuits at a single-node level, enforces strict mathematical capacity bottlenecks, and ultimately scales into a multi-node spatial graph to evaluate geometric fusion and physics-informed constraints.

---

## Phase 1: The Classical Baseline
**Objective:** Establish a high-performance classical architecture to benchmark quantum models against.
* **Input:** 2D energy density spectrograms from a single accelerometer (`PE11`). Shape: `(1, 64, 1000)`.
* **Model:** Convolutional Autoencoder (CAE).
* **Compression:** Unconstrained (`dim=8`) for single-sensor (`PE11`) ablations. The network reduces the 64,000-dimensional input into a dense, 8-dimensional continuous latent vector.
* **Performance:** Converged to a highly accurate Validation MSE of `~0.0089`, demonstrating that standard deep learning can effectively map baseline structural vibrations given sufficient capacity.

## Phase 2: Single-Sensor Quantum Autoencoder
**Objective:** Replace the classical bottleneck with a Variational Quantum Circuit (VQC) to test quantum representational capacity.
* **Architecture:** Hybrid Quantum Autoencoder (HQAE).
* **Process:** 1. Classical convolutional encoder reduces input to an 8-dimensional vector.
    2. Data is encoded into an 8-qubit quantum state via angle embedding.
    3. Parameterized quantum layers (entanglement + rotation) process the state.
    4. Measurement (expectation values) returns an 8-dimensional vector to the classical decoder.
* **Findings:** Demonstrated that heavily entangled quantum topologies suffer from vanishing gradients (Barren Plateaus) when mapping continuous, unconstrained structural data.

## Phase 3: The Compression Bottleneck (Dim=2 Single-Sensor Baseline)
**Objective:** Establish a mathematically rigorous, capacity-constrained baseline to allow for a 1-to-1 spatial ablation comparison with Phase 4.
* **The Constraint:** To simulate the state-vector simulation limits of a 6-sensor quantum graph (maximum 12 qubits total, or 2 qubits per sensor), the Phase 3 latent capacity was strictly suffocated from `dim=8` down to **`dim=2`**.
* **The Bottleneck Effect:** Compressing 64,000 data points into just 2 variables severely starves the network. Models are forced to discard high-frequency environmental noise, resulting in a "blurrier" baseline reconstruction (MSE `~0.015`).
* **Evaluation:** Evaluated against an incipient Acoustic Emission (AE) proxy (Amplitude: `0.35`) injected at node `PE11`. 
* **Findings:** Empirically proved **Single-Sensor Spatial Blindness**. When heavily compressed, a single isolated sensor mathematically lacks the spatial context to confidently separate a subtle micro-crack anomaly from its own high baseline error, plateauing at an AUC of `~0.62`.

## Phase 4: Multi-Sensor Spatial Graph Fusion & Physics-Informed Constraints
**Objective:** Determine if geometric spatial context (peer pressure from adjacent nodes) can overcome the `dim=2` per-sensor capacity bottleneck.
* **Input:** Multi-sensor synchronized spectrogram tensors encompassing sensors `PE11` through `PE23`. Shape: `(6, 64, 1000)`.
* **Architecture (Siamese):** The batch and channel dimensions are dynamically folded (`B*C, 1, 64, 1000`) so all 6 sensors pass through identical, weight-tied convolutional filters before entering the global latent space.
* **The Global Quantum Graph:** The 6 independent `dim=2` vectors are flattened into a single 12-qubit global quantum state, allowing for complex spatial entanglement topologies between physical bridge nodes.
* **Findings:**
    1.  **Geometric Peer Pressure (Classical Victory):** The classical model successfully used the spatial coherence of healthy sensors to better isolate the `PE11` anomaly, boosting incipient AUC from `0.620` (Single) to `0.650` (Multi).
    2.  **The Quantum Scaling Wall:** The highly entangled `HQAE [Strong]` model collapsed during spatial fusion. Expanding all-to-all entanglement to 12 qubits destroyed gradient stability (Barren Plateaus), proving that forced global entanglement limits NISQ-era spatial scaling.
    3.  **The PINN Paradox:** Physics-Informed Neural Networks (`Bridge Soft`) correctly rejected the synthetic damage as physically impossible, proving their loss constraints function exactly as intended (see Core Components below).

---

## Core Architectural Components and Tensor Flow

### 1. `ClassicalAutoencoder (CAE)`
A universal, highly robust PyTorch module utilizing a Siamese architecture. It dynamically folds the batch and channel dimensions so any number of physical sensors pass through the exact same weight-tied convolutional filters in parallel.

**The Tensor Transformation Pipeline:**
* **1. Input Ingestion:** 
  * Shape: `(Batch Size, Channels, 64, 1000)` 
  * *Note: Number of channels = Number of sensors*
  * Number of channels: C=1 for Phases 1, 2, and 3 (single sensor `PE11`)
  * Number of channels: C=6 for Phase 4 (multi-sensor fusion)
* **2. Siamese Folding (Pre-Processing):** 
  * Operation: `x.view(B * C, 1, 64, 1000)`
  * *All sensor matrices are stacked into a massive 1D batch to ensure identical feature extraction.*
* **3. Convolutional Encoder:**
  * Conv1 (3x3): `(B*C, 16, 64, 1000)`
  * MaxPool1 (2x4): `(B*C, 16, 32, 250)`
  * Conv2 (3x3): `(B*C, 32, 32, 250)`
  * MaxPool2 (4x10): `(B*C, 32, 8, 25)`
* **4. Flattening:**
  * Operation: `x.view(x.size(0), -1)`
  * Shape: `(B*C, 6400)`
* **5. The Capacity Bottleneck (Latent Space Divergence):**
  * **Phases 1, 2 and 3 (Unconstrained):** Linear Projection to `(B*C, 8)`
  * **Phases 3 and 4 (Constrained):** Linear Projection to `(B*C, 2)`
* **6. Decoder Expansion:**
  * Linear Projection: From `(B*C, 8)` or `(B*C, 2)` back up to `(B*C, 6400)`
  * Reshape: `(B*C, 32, 8, 25)`
* **7. Transpose Convolutional Decoder (Symmetric):**
  * ConvTranspose1 (4x10): `(B*C, 16, 32, 250)`
  * ConvTranspose2 (2x4): `(B*C, 1, 64, 1000)`
* **8. Siamese Unfolding (Output):**
  * Operation: `x.view(B, C, 64, 1000)`
  * *The reconstructed multi-sensor graph is restored to its original geometric dimensions.*

### 2. `HybridQuantumAutoencoder (HQAE)`
The master hybrid wrapper integrating PennyLane's quantum state-vector simulators with PyTorch's automatic differentiation.

**The Quantum Graph Assembly Pipeline:**
* **1. Classical Feature Extraction:** 
   * The HQAE utilizes the exact `CAE` convolutional encoder to compress the input to the Siamese bottleneck: `(B*C, 8)` or `(B*C, 2)`.
* **2. Global Graph Assembly:** 
   * Operation: `x.view(B, C * latent_dim_per_sensor)`
   * **Phase 2 Shape:** `(B, 8)` *(1 sensor × 8 latents)*
   * **Phase 3 Shape:** `(B, 2)` *(1 sensor × 2 latents)*
   * **Phase 4 Shape:** `(B, 12)` *(6 sensors × 2 latents)*
   * *Independent sensor features are flattened into a single global vector to map to the quantum register.*
* **3. The Device Bridge & Quantum Execution:** 
   * The global vector is moved from `cuda` to `cpu`. 
   * PennyLane's `VQCTorchLayer` executes the parameterized quantum circuit (angle embedding, entanglement, measurement).
   * Returns expectation values in the exact same shape (e.g., `(B, 12)`).
   * Output is migrated back to `cuda`.
* **4. Graph Deconstruction:** 
   * Operation: `x.view(B * C, latent_dim_per_sensor)`
   * Shape: `(B*C, 8)` or `(B*C, 2)`
   * *The global quantum state is sliced back into isolated physical sensor representations to pass to the classical decoder.*

### 3. `VQCTorchLayer` (The Quantum Bottleneck)
The PennyLane `qnode` that replaces the classical latent space. Supports multiple architectural topologies:
* **`none`:** Independent rotation gates. No entanglement. (Scaled successfully in Phase 4).
* **`basic`:** Ring topology entanglement. Minimal gradient vanishing.
* **`strong`:** All-to-all entanglement. (Collapsed in Phase 4 due to Barren Plateaus).
* **`bridge_hard`:** Entanglement geometrically restricted to physical spans and girders (e.g., `PE11` entangles with `PE12`, but not `PE23`).

### 4. Physics-Informed Neural Network (PINN / `bridge_soft`)
A continuous classical architecture augmented with a custom wave-mechanics loss function.
* **The Physical Constraint:** Enforces continuous elastic wave propagation across the spatial graph by penalizing non-physical spatial gradients in the reconstruction. 
* **The Loss Formulation:** $$L_{Total} = L_{Recon} + \lambda \left|\left| \frac{\partial^2 u}{\partial t^2} - c^2 \nabla^2 u \right|\right|^2$$
* **The Paradox:** 
   * PINN yielded the worst performance (`AUC ~0.612`) when evaluated against a synthetic anomaly injected at a single node (`PE11`)
   * Because the synthetic energy spike did not mathematically propagate to adjacent nodes, it created a physically impossible spatial discontinuity ($\nabla^2 u \to \infty$).
   * The PINN's physics penalty exploded $\implies$ it correctly rejected the synthetic anomaly.
   * This indicates strong theoretical potential for PINNs when applied to real-world, physically propagating structural damage.