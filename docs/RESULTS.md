# Empirical Results

## Executive Summary
The SPECTRA-CQ research is structured into progressive phases to benchmark the efficacy of Variational Quantum Circuits (VQCs) for Structural Health Monitoring (SHM). 

**Phase 1** 
* Generated 2D energy density spectrograms from raw signal data for accelerometer sensor `PE11` (100Hz sampling rate, 20 frequency bins, 8 time bins, `100*64` time steps).
* Established classical baseline (Convolutional AutoEncoder - CAE) - converged to a Validation MSE of `0.0087` to `0.0089`.

**Phase-2**
* Hybrid Quantum Autoencoder (HQAE) was developed to replace the classical bottleneck with a parameterized quantum circuit.
* Ablation study on entanglement topology and circuit depth was conducted to isolate the parameters of quantum advantage.
* Established that naive global entanglement natively triggers catastrophic gradient vanishing (Barren Plateaus) when processing continuous macroscopic structural data.

**Phase 3** 
* Optimized the HQAE by mitigating barren plateaus using:
    * gradient clipping
    * scaling classical latents to $[-\pi, \pi]$
    * topology-specific uniform initialization
* Evaluated the anomaly detection threshold of these models, empirically proving a hard physical boundary: **A single sensor architecture lacks the required spatial context to detect incipient damage, necessitating a transition to multi-sensor fusion.**

---

## Phase 1 and 2: Foundation + Entanglement Bottleneck

### The Classical Baseline (CAE)
A purely Classical Convolutional Autoencoder (CAE) was trained to establish the reconstruction benchmark.
* **Protocol:** 15-Epoch constraint across 3 deterministic seeds (42, 100, 2026).
* **Benchmark MSE:** 
    * The CAE converged rapidly $\rightarrow$ exhibited high stability invariant of seed initialization. 
    * Final Validation MSE converged tightly between `0.0087` and `0.0089`.

### Quantum Topology + Depth Ablation (HQAE)
To isolate the parameters of quantum advantage, the 8-dimensional latent space was replaced with a PennyLane VQC and subjected to an ablation matrix.

1. **Zero Entanglement (`none`):** 
    * Qubits operated as independent $R_x$ rotations.
    * Exhibited high variance but avoided barren plateaus $\rightarrow$ converged to a Validation MSE of `~0.010`.
    * The extreme parameter constraint (24 weights) acted as a strict geometric regularizer.
2. **Local & Global Entanglement (`basic`, `strong`):** 
    * Ring and All-to-all topologies at varying depths (1, 3, 5). 
    * **Result:**
        * Gradients vanished immediately. 
        * the models became permanently trapped in a Barren Plateau across all deterministic seeds and depths and flatlined at MSE `~0.0317`.
    * **Insight:** Introducing `CNOT` gates to force spatial coupling created a noise landscape that completely blinded the classical Adam optimizer.

---

## Phase 3: Synthetic Anomaly Evaluation (Inference)

To evaluate empirical resilience, the frozen architectures (Classical CAE, HQAE `none` and HQAE `strong`) were tested against mathematically injected synthetic degradation using a 3-seed ensemble protocol.

### Catastrophic Damage Detection
* **Parameters:** 30% Gaussian noise and complete erasure of 10 frequency scales.
* **Result:** All architectures successfully detected the anomaly with a perfect AUC-ROC of **1.000**.
* **Insight:** The sheer magnitude of degradation rendered the mathematical difference trivial to decode $\leftarrow$ provided no metric for early-warning capabilities.

### Incipient Damage Detection (The Single-Sensor Limit)
* **Parameters:** 2% Gaussian noise and a narrow 2-scale frequency mask (simulating early-stage micro-cracking).
* **Result:** The performance of all single-sensor models plummeted to random guessing.
    * **Classical CAE:** AUC ~0.560
    * **HQAE [Strong]:** AUC ~0.527
    * **HQAE [None]:** AUC ~0.473
* **Insight (Memorization vs. Regularization):** The over-parameterized Classical CAE slightly memorized the synthetic noise pattern, resulting in a marginal bump above 0.500. The heavily constrained HQAE `none` aggressively smoothed out the noise, treating the 2% micro-crack as standard environmental variance (e.g., wind).

---

## Conclusion + Phase 4 Roadmap
The Phase 3 empirical results prove a fundamental physical limitation of localized SHM: **a single accelerometer simply lacks the necessary signal-to-noise ratio and spatial context to distinguish subtle, early-stage structural degradation from routine environmental variance.** If a single sensor cannot reliably detect localized anomalies, the system must map the global correlations of the entire structure. 

**Next Steps (Phase 4: Multi-Sensor Fusion):** We will transition the architecture to ingest 6 synchronized sensors (PE11, PE12, PE13, PE21, PE22, PE23) simultaneously. This shift justifies the ultimate test of the Quantum Advantage: while classical networks struggle to map complex multi-dimensional spatial correlations without exploding parameter counts, Quantum Entanglement (`CNOT` gates) is theoretically designed to map these highly correlated physical states efficiently.