# Phase 4: Multi-Sensor Fusion & Physics-Informed Quantum Graphs (PIQNN)

## Overview
Phase 3 empirically proved a single-sensor limitation: a solitary telemetry node (`PE11`) lacks the spatial context required to distinguish incipient structural micro-cracking from ambient environmental variance. 

Phase 4 resolves this by transitioning from localized 1D processing to global 6D Spatio-Temporal processing.
Instead of utilizing a classical Graph Neural Network (GNN), the architecture forces the Variational Quantum Circuit (VQC) to act as a physical structural graph to evaluate if quantum entanglement can natively map macroscopic physics.

## 1. The Siamese Spatial Bottleneck
The architecture utilizes a **Weight-Tied Siamese Encoder** to process 6 sensors without causing a 6x explosion in classical convolutional parameters.

* **Input Tensor:** A stacked multi-channel matrix of shape `(Batch, 6, 64, 1000)`.
* **Folding Operation:** The Batch and Channel dimensions are folded $\rightarrow$ `(Batch * 6, 1, 64, 1000)`.
* **Shared Convolution:** All 6 sensors pass through the *exact same* convolutional filters independently.
* **The Quantum Handoff:** Each sensor's high-dimensional feature map is compressed into exactly **2 continuous dimensions**. 
* **Global Output:** The 6 independent outputs are unfolded and concatenated into a `(Batch, 12)` tensor.
   * `PE11` $\rightarrow$ Qubits $Q_0, Q_1$
   * `PE12` $\rightarrow$ Qubits $Q_2, Q_3$
   * `PE13` $\rightarrow$ Qubits $Q_4, Q_5$
   * `PE21` $\rightarrow$ Qubits $Q_6, Q_7$
   * `PE22` $\rightarrow$ Qubits $Q_8, Q_9$
   * `PE23` $\rightarrow$ Qubits $Q_{10}, Q_{11}$

## 2. Spatial Combinatorics
In `quantum_layer.py` $\rightarrow$ the parameter initialization for the `bridge_graph` topologies was strictly defined as `(n_layers, 15)`.
This represents the maximum number of unique physical connections (edges) between the 6 sensors on the bridge.

### The Math of Graph Edges
In an undirected graph with no self-loops (a sensor cannot entangle with itself), the total number of unique pairs is calculated using the standard combinations formula:

$$C(n, k) = \frac{n!}{k!(n-k)!}$$

Where:
* $n$: Total number of sensors (6)
* $k$: Size of the pairing (2 sensors per CNOT gate)

$$C(6, 2) = \frac{6 \times 5}{2} = 15 \text{ unique combinations}$$

Because the bridge is a 3x2 rectangular grid (3 girders across 2 spans), these 15 connections represent every possible path stress waves could take:

**Transverse / Intra-Span Connections (3 per span = 6 total):**
1. `PE11` $\leftrightarrow$ `PE12` (Span 1, Adjacent Girders)
2. `PE11` $\leftrightarrow$ `PE13` (Span 1, Outer Girders)
3. `PE12` $\leftrightarrow$ `PE13` (Span 1, Adjacent Girders)
4. `PE21` $\leftrightarrow$ `PE22` (Span 2, Adjacent Girders)
5. `PE21` $\leftrightarrow$ `PE23` (Span 2, Outer Girders)
6. `PE22` $\leftrightarrow$ `PE23` (Span 2, Adjacent Girders)

**Longitudinal & Diagonal / Cross-Span Connections (9 total):**
1. `PE11` $\leftrightarrow$ `PE21` (Continuous load path down Girder 1)
2. `PE11` $\leftrightarrow$ `PE22` (Diagonal stress transfer)
3. `PE11` $\leftrightarrow$ `PE23` (Extreme diagonal, max physical distance)
4. `PE12` $\leftrightarrow$ `PE21` (Diagonal stress transfer)
5. `PE12` $\leftrightarrow$ `PE22` (Continuous load path down Girder 2)
6. `PE12` $\leftrightarrow$ `PE23` (Diagonal stress transfer)
7. `PE13` $\leftrightarrow$ `PE21` (Extreme diagonal, max physical distance)
8. `PE13` $\leftrightarrow$ `PE22` (Diagonal stress transfer)
9. `PE13` $\leftrightarrow$ `PE23` (Continuous load path down Girder 3)


## 3. Physics-Informed Distance Mapping
To prevent the VQC from succumbing to the chaotic over-entanglement known as a "Barren Plateau" (observed in Phase 2 with the `strong` topology), the 15 possible connections are constrained by the actual Euclidean geometry of the OpenLAB bridge.

### Euclidean Distance Calculation
The physical distance between any two sensor nodes is computed as:

$$D_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

Where:
* $D_{ij}$: The absolute physical distance in meters between sensor $i$ and sensor $j$.
* $x$: The longitudinal coordinate (span placement). Spans are 15.0m apart.
* $y$: The transverse coordinate (girder placement). Girders are 1.5m apart.

### Topology A: Hard Boundary (`bridge_graph_hard`)
Acts as a strict geometric cut-off to isolate spatial noise.
* **Mechanism:** The Controlled-Y (`CRY`) entanglement gate is only applied if $D_{ij} \le 15.0 \text{ m}$. 
* **Effect:** Allows adjacent girder communication (1.5m) and direct longitudinal span-to-span communication (15.0m), but severs the extreme diagonal cross-talk (e.g., `PE11` to `PE23` is 15.3m).

### Topology B: Soft Attenuation (`bridge_graph_soft`)
Models the physical decay of acoustic/stress waves traveling through concrete.
* **Mechanism:** All 15 connections are permitted, but the trainable rotation angle ($\theta$) is penalized exponentially by distance.

$$\theta_{applied} = \theta_{learned} \times e^{-\gamma D_{ij}}$$

Where:
* $\theta_{applied}$: The final rotation angle passed to the quantum simulation.
* $\theta_{learned}$: The raw parameter outputted by the classical Adam optimizer.
* $\gamma$: The structural attenuation coefficient (currently set to 0.1).
* $D_{ij}$: The calculated physical distance.

**Implication $\implies$** The optimizer must output a massively larger gradient to force entanglement between distant nodes (`PE11` and `PE23`) compared to adjacent nodes (`PE11` and `PE12`), perfectly mimicking the energy required to propagate a physical wave across that distance.