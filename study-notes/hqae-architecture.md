# Hybrid Quantum Autoencoder (HQAE) Architecture and Concepts

### **Pre-Note: The Classical Foundation**
Before diving into the quantum mechanics, it is critical to understand the constraints of our ablation study.

To mathematically prove whether quantum correlation provides an advantage in SHM $\rightarrow$ the spatial input, the spatial encoder and the spatial decoder must be 100% identical to the Classical Autoencoder (CAE).

The only independent variable in this architecture is the Latent Bottleneck. 
We violently rip out the classical dense bottleneck and replace it with a Variational Quantum Circuit (VQC).

---

### **Concept Note: Why Inject Quantum Mechanics?**

In the classical model, the bottleneck forced 6,400 data points down into continuous floating-point numbers (8 per sensor in early single-sensor ablation phase, strictly 2 per sensor in later phases).

Although they are effective, classical nodes operate completely independently of one another during the forward pass.

**The Structural Engineering Problem:**
When a massive concrete bridge degrades, the vibration anomalies are rarely isolated. A micro-crack in a cross-beam might cause a 4 Hz resonance to suddenly couple with a 12 Hz resonance. These macroscopic frequencies are physically entangled.

**The Quantum Hypothesis:**
By mapping these compressed features into Quantum Bits (Qubits) and applying `CNOT` (Controlled-NOT) gates, we can artificially entangle the data streams. The hypothesis of Phase 4 is that a highly entangled quantum architecture (the "Strong" topology) can map these complex, coupled structural degradation frequencies better than independent classical nodes.

---

### **Note: The Device Bridge (The CPU/GPU Handshake)**

Before the data can enter the quantum circuit, we face a severe hardware limitation of the current Noisy Intermediate-Scale Quantum (NISQ) era simulators.

Our spatial convolutions (handling matrices of `64 × 1000`) run on the Colab GPU (CUDA).
However, PennyLane's `default.qubit` state-vector simulator requires heavy CPU memory allocations to simulate the complex probability amplitudes of the Hilbert space.
Attempting to simulate deep quantum entanglement directly on PyTorch CUDA tensors causes fatal memory collisions.

To fix this, `hqae.py` implements a **Device Bridge**:
1. The spatial GPU compresses the data down to the global latent vector (e.g., 2-dimensional for Phase 3, or 12-dimensional for Phase 4).
2. The PyTorch graph is intercepted, and the global tensor is dynamically pushed to the CPU.
3. The quantum circuit executes on the CPU.
4. The resulting output vector is pushed back to the GPU to resume spatial decoding.

---

## **The Quantum Bottleneck**
The quantum bottleneck operates in three distinct phases for every single forward pass:

### **1. State Preparation (Angle Embedding)**
We cannot just "feed" classical numbers into a qubit. A qubit is a 2D complex vector existing on a probability sphere (the Bloch sphere).
* We use **Angle Embedding** ($R_y$ rotations).
* The classical features dictate the angle at which we rotate the corresponding qubits away from the North Pole ($|0\rangle$) toward the South Pole ($|1\rangle$).
* This encodes the bridge's vibration energy directly into the quantum probability amplitudes.

### **2. The Ansatz (Trainable Entanglement)**
This is the "brain" of the quantum circuit. It consists of layers of trainable rotations and `CNOT` gates. In Phase 4, we dynamically test three topologies:
* **`none` (The Control):** Applies simple independent rotations. Proves if quantum mechanics are even necessary.
* **`basic` (Ring Topology):** Qubit 1 entangles with Qubit 2; 2 with 3; and the final qubit loops back to entangle with Qubit 1. Simulates localized frequency coupling.
* **`strong` (All-to-All Topology):** Applies full 3D Euler rotations ($R_x, R_y, R_z$) and entangles every qubit with every other qubit. Maximizes the ability to map global, structure-wide frequency shifts.

### **3. Measurement (State Collapse)**
To return the data to the classical PyTorch decoder, the quantum state must be collapsed back into standard numbers.
* We measure the **Pauli-Z Expectation Value** ($\langle \sigma_z \rangle$) of each of the qubits in the global register (e.g., 2 or 12).
* This returns an equivalently sized continuous floating-point vector. Because of the math of the Pauli-Z operator, these numbers are strictly bounded between `[-1.0, 1.0]`.


## **Technical Implementation Summary (Tensor Journey)**
The software engineering implementation for this architecture spans multiple files, specifically bridging `src/models/hybrid/hqae.py` and `src/models/hybrid/quantum_layer.py`.

Here is the exact flow of tensor dimensionality during a single forward pass:

**1. Input Phase & Spatial Encoding (GPU)**
* *Identical to CAE*
* Input `(B, 1, 64, 1000)` is spatially crushed down to `(B, 32, 8, 25)`.

**2. Pre-Quantum Compression (GPU)**
* Operation: `x.view(x.size(0), -1)` (Dynamic Flattening)
  * Shape Mutated: $\rightarrow$ `(B, 6400)`
* Layer: `nn.Linear(in_features=6400, out_features=latent_dim)`
  * Shape Mutated: $\rightarrow$ `(B, 8)` or `(B, 2)` **(The Continuous Classical Vector)**

**3. The Device Bridge (GPU $\rightarrow$ CPU)**
* Operation: `x_cpu = x.to(torch.device('cpu'))`
  * Data leaves VRAM and enters System RAM.

**4. The Variational Quantum Circuit (CPU)**
* *File:* `src/models/quantum/vqc.py`
* **Embedding:** `qml.AngleEmbedding(rotation='Y')`
  * Maps `(B, N)` floats into $N$ continuous Qubit probability states.
* **Ansatz:** `build_ansatz(topology)`
  * Rotates and entangles the states across layers based on the configuration.
* **Measurement:** `[qml.expval(qml.PauliZ(wires=i)) for i in range(N)]`
  * State collapses
  * Shape Mutated: $\rightarrow$ Returns a `(B, N)` tensor bounded `[-1, 1]`.

**5. The Device Bridge (CPU $\rightarrow$ GPU)**
* Operation: `x = x_quantum.to(original_device)`
  * The measured `[-1, 1]` tensor is pushed back into VRAM.

**6. Spatial Decoding & Output Phase (GPU)**
* *Identical to CAE.* * Layer: `nn.Linear(in_features=latent_dim, out_features=6400)` $\rightarrow$ Tensor mapped to `(B, 6400)`
* Operation: `x.view(x.size(0), 32, 8, 25)` $\rightarrow$ Reshaped to `(B, 32, 8, 25)`
* Upsampled through `ConvTranspose2d` layers back to `(B, 1, 64, 1000)`.
* `nn.Sigmoid()` activation limits the final reconstruction to `[0, 1]` for MSE loss calculation against the normalized input.