## Theoretical Justification: Paradigm Shift from 1D Sequential to 2D Spatial Modeling in SHM

### 1. **The Nature of Structural Vibration Data**
Vibration data collected from full-scale civil infrastructure (such as the openLAB bridge) possesses specific characteristics that dictate the required modeling architecture:

* **High Dimensionality**: Sampled at high frequencies (e.g., $100\text{ Hz}$ to $500\text{ Hz}$). So a mere 10-second structural event generates thousands of discrete data points.
* **Non-Stationarity**: Traffic-induced or seismic excitations cause the statistical properties of the vibration signal to change over time. The frequency content is not static.
* **Low Signal-to-Noise Ratio (SNR)**: Damage signatures (e.g., micro-cracking causes a slight loss of stiffness) manifest as minute frequency shifts buried under massive environmental and ambient noise.

### 2. **1D Sequential Modeling: The BiLSTM Approach**
Recurrent Neural Networks (RNNs) and their variants process data sequentially. The fundamental assumption is that the state of the system at time $t$ is a function of the input at time $t$ and the hidden state from time $t-1$.

#### **2.1 The LSTM Cell Formulation**
To solve the vanishing gradient problem inherent in vanilla RNNs, the LSTM introduces a memory cell $C_t$ and gating mechanisms to regulate information flow. For a given input sequence $x = (x_1, x_2, \dots, x_T)$, the mechanics of a single LSTM cell at timestep $t$ are defined by the following equations:
* **Forget Gate:** Determines what information to discard from the previous cell state

    * $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

* **Input Gate & Candidate Memory:** Determines what new information to store.
    * $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

    * $$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$

* **Cell State Update:** Computes the new internal memory.
    * $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
    
* **Output Gate & Hidden State:** Determines what the cell outputs to the next layer/timestep.
    * $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
    * $$h_t = o_t \odot \tanh(C_t)$$

where:
| Symbol | Definition |
| :--- | :--- |
| $f_t$ | Forget gate activation vector at time $t$ |
| $i_t$ | Input gate activation vector |
| $o_t$ | Output gate activation vector |
| $\tilde{C}_t$ | Candidate values (new information) to be added to the memory |
| $C_t$ | Updated cell state (long-term memory) at time $t$ |
| $C_{t-1}$ | Previous cell state |
| $h_t$ | Final hidden state (short-term memory) at time $t$ |
| $[h_{t-1}, x_t]$ | Concatenation of the previous hidden state and the current input |
| $W_f, W_i, W_C, W_o$ | Learnable weight matrices for their respective gates |
| $b_f, b_i, b_C, b_o$ | Learnable bias vectors for their respective gates |
| $\sigma$ | Sigmoid activation function mapping values to $(0, 1)$ |
| $\tanh$ | Hyperbolic tangent function mapping values to $(-1, 1)$ |
| $\odot$ | Hadamard product (element-wise multiplication) |

#### **2.2 Bidirectional Extension (BiLSTM)**
A standard LSTM only retains past context. A BiLSTM processes the sequence in both forward and backward directions. It concatenates the hidden states to capture context from both the past and the future:

$$y_t = [\vec{h}_t \oplus \overleftarrow{h}_t]$$

While it is theoretically capable of learning temporal dependencies; the BiLSTM forces the network to implicitly learn the harmonic and spectral properties of the vibration data entirely through the optimization of its weight matrices via Backpropagation Through Time (BPTT).

### 3. **The 2D Spatial Paradigm: Time-Frequency Representations (TFR)**
we must fundamentally alter the feature space to transcend the sequential bottlenecks of recurrent architectures. Signal processing theory dictates that non-stationary structural vibrations cannot be comprehensively characterized in the time domain alone. By projecting the 1D acceleration sequence into a 2D time-frequency plane, we explicitly isolate the transient resonant frequencies that serve as primary indicators of structural health.

#### **3.1 Short-Time Fourier Transform (STFT)**
STFT achieves this mathematical projection by segmenting the continuous signal into short, overlapping frames prior to applying the discrete Fourier transform. This localized approach gives a complex-valued matrix representing both signal phase and magnitude over time. For a continuous structural response $x(t)$ and a localized sliding window function $w(t)$ (such as a Hanning or Gaussian window), the STFT is defined as:

$$X(\tau, \omega) = \int_{-\infty}^{\infty} x(t) w(t-\tau) e^{-j\omega t} dt$$

where:

| Symbol | Definition |
| :--- | :--- |
| $X(\tau, \omega)$ | Complex-valued STFT output matrix |
| $S(\tau, \omega)$ | Real-valued Spectrogram (energy density matrix) |
| $x(t)$ | Continuous raw vibration signal in the time domain |
| $w(t)$ | Localized sliding window function (e.g., Hanning window) |
| $\tau$ | Time translation parameter (center of the sliding window) |
| $\omega$ | Angular frequency variable |

To generate a spatial matrix strictly suitable for visual feature extraction, we compute the energy density spectrum, commonly referred to as a spectrogram. The spectrogram $S(\tau, \omega)$ is the squared magnitude of the STFT:

$$S(\tau, \omega) = |X(\tau, \omega)|^2$$

This transformation successfully converts a chaotic temporal sequence into a highly structured spatial texture, where axes correspond directly to time and frequency.

#### **3.2 Continuous Wavelet Transform (CWT) Integration**
STFT relies upon a fixed window size (thereby fixing the time-frequency resolution). But structural impulses occasionally demand multi-resolution analysis. 

CWT offers a robust alternative by convolving the signal with a scaled and translated mother wavelet $\psi(t)$:

$$W(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^* \left( \frac{t-b}{a} \right) dt$$

where:

| Symbol | Definition |
| :--- | :--- |
| $W(a, b)$ | Wavelet coefficients (2D scalogram) |
| $a$ | Scale parameter (inversely proportional to frequency) |
| $b$ | Translation parameter (temporal position) |
| $x(t)$ | Continuous raw vibration signal |
| $\psi^*$ | Complex conjugate of the mother wavelet function |

Here, the scale parameter $a$ is inversely proportional to frequency, and $b$ controls temporal translation. Whether deploying STFT spectrograms or CWT scalograms, the underlying objective remains identical. We eliminate the need for the neural network to implicitly deduce harmonic motion, instead presenting it with a direct map of spectral energy.

### 4. **Convolutional Neural Networks (CNN) for Spatial Feature Extraction**
Unlike recurrent networks that maintain a fragile temporal state, CNNs process data spatially through localized receptive fields. When a structural anomaly alters the stiffness matrix of a bridge, it manifests on the spectrogram as a distinct visual aberration (such as a discontinued frequency band or a sudden energy attenuation).

#### **The Convolutional Operation**
A convolutional layer applies a set of learnable two-dimensional filters (or kernels) $K$ across the input spectrogram $S$. This operation detects localized energy patterns regardless of their precise temporal occurrence within the window. The resulting feature map $Y$ at spatial coordinates $(i, j)$ is computed via the discrete spatial convolution:

$$Y(i, j) = \sigma \left( \sum_{m} \sum_{n} K(m, n) S(i+m, j+n) + b \right)$$

where:

| Symbol | Definition |
| :--- | :--- |
| $Y(i, j)$ | Output feature map evaluated at spatial coordinates $(i, j)$ |
| $K(m, n)$ | Learnable 2D convolutional filter (kernel) matrix |
| $S$ | Input 2D spectrogram matrix |
| $\sigma$ | Non-linear activation function (typically ReLU) |
| $b$ | Learnable bias scalar for the filter |

Through successive layers of convolution and maximum pooling, the architecture systematically compresses the high-dimensional spectrogram. It distills complex visual data into a robust, low-dimensional latent vector $z$, which represents the global structural state ready for final classification.

### 5. **Theoretical Superiority of the 2D Spatial Framework**
The paradigm shift from 1D sequential models (BiLSTM) to 2D spatial models (CNN) is not a mere architectural preference. It represents a fundamental mathematical optimization for analyzing structural physics. This superiority is anchored in three distinct theoretical mechanisms:

#### **5.1 Physics-Informed Inductive Bias**
Neural networks function fundamentally as universal approximators. When a BiLSTM processes raw acceleration data, it must blindly deduce the physical laws of resonance, damping and harmonic motion entirely through gradient descent. The 2D approach drastically reduces this computational burden. By applying the STFT, we forcefully inject established physical principles into the preprocessing phase. The Fourier transform executes the highly complex frequency decomposition mathematically prior to inference. Consequently, the neural network dedicates its parameter capacity exclusively to pattern recognition rather than basic signal unrolling.

#### **5.2 Mitigation of Vanishing Context (Global vs. Local View)**
Recurrent models suffer from profound temporal myopia. Because they process data incrementally, comparing a resonant frequency at $t=0.1\text{ s}$ with its decayed state at $t=1.9\text{ s}$ requires the network to bridge an enormous informational gap via the hidden state $h_t$. Over thousands of sampling points, the BiLSTM forget gate inevitably dilutes the influence of early resonant peaks. This leads to catastrophic context loss.

A CNN operating on a spectrogram circumvents this sequential bottleneck entirely. The entire vibration event is presented simultaneously as a static image. The hierarchical, expanding receptive fields of the CNN can mathematically correlate an impact transient at the beginning of the window with the resulting structural decay at the very end in a single forward pass. There is no sequential forgetting.

#### **5.3 Spatial Segregation of Non-Stationary Noise**
Full-scale civil infrastructure operates in environments saturated with ambient noise. Traffic loads generate chaotic, broadband excitations that easily obscure the subtle micro-cracking signatures within a 1D sequence. In the temporal domain, this noise is mathematically superimposed directly over the structural response. A BiLSTM struggles immensely to separate the two overlapping signals.

In the 2D time-frequency domain, however, energy distributions become spatially segregated. High-frequency sensor noise manifests as uniform, scattered background pixels. Conversely, genuine structural modes appear as distinct, high-intensity geometric bands. A CNN inherently learns to ignore the scattered background static through its max-pooling operations, focusing its feature extraction strictly on the contiguous structural bands. This spatial filtering grants the 2D architecture a profound resilience to environmental noise that routinely causes 1D models to fail.

### 6. **Synthesis: Architectural Comparison of 1D vs. 2D Paradigms**

To solidify the theoretical justification for transitioning away from sequential models in Structural Health Monitoring, we must distill the fundamental operational differences. Recurrent architectures and convolutional networks solve entirely different optimization problems. An RNN attempts to carry a fragile mathematical state across time. A CNN, conversely, evaluates a static mathematical landscape in its entirety. 

The following table synthesizes the core mechanical, mathematical and computational distinctions between 1D RNNs and 2D CNNs when applied to structural vibration analysis.

| Architectural Dimension | 1D Recurrent Neural Networks (RNN/LSTM) | 2D Convolutional Neural Networks (CNN) |
| :--- | :--- | :--- |
| **Fundamental Processing Mechanism** | Ingests data chronologically. Relies upon an internal hidden state to function as a sequential temporal memory. | Evaluates data spatially. Utilizes overlapping receptive fields to extract geometric textures simultaneously. |
| **Core Mathematical Operation** | Temporal recurrence (mapping present input and past state): <br> $h_t = \sigma(W_x x_t + W_h h_{t-1} + b)$ | Discrete spatial convolution (mapping local pixel neighborhoods): <br> $Y(i,j) = \sigma(\sum_m \sum_n K_{m,n} S_{i+m, j+n} + b)$ |
| **Context Retention & Gradient Flow** | Highly susceptible to vanishing gradients. Long-term dependencies decay exponentially over the hundreds of discrete time steps required for high-frequency SHM data. | Context is maintained globally within the bounded image window. Gradient flow through spatial layers is direct, avoiding the sequential decay inherent to recurrent unrolling. |
| **Feature Space Formulation** | Operates directly on raw time-domain amplitudes. The network is forced to implicitly learn harmonic laws and spectral behavior via standard weight optimization. | Operates on pre-computed Time-Frequency Representations (e.g., STFT Spectrograms). Spectral decomposition is handled explicitly via physics-informed mathematical transformations. |
| **Handling of Non-Stationary Noise** | Poor. Ambient structural noise is mathematically superimposed directly over the damage signature within the 1D sequence, severely complicating feature extraction. | Robust. Broadband noise distributes uniformly across the spectrogram as scattered pixels. Structural modes manifest as concentrated energy bands, easily isolated via max-pooling. |
| **Computational Parallelism** | Severely limited. Execution is strictly sequential ($\mathcal{O}(T)$ operations), creating a severe processing bottleneck during both forward passes and backpropagation. | Highly optimized. Convolutional filters are applied across the spatial dimensions simultaneously, fully exploiting the parallel architecture of modern GPU accelerators. |

### 7. **The Hybrid Quantum-Classical Extension (HQCNN)**
Transitioning to a 2D convolutional framework effectively filters out sequential noise; yet the final classification stage introduces a new bottleneck. Classical models typically rely on a Multi-Layer Perceptron (MLP) to process the extracted latent vector $z$. These dense networks demand massive amounts of training data, but instances of actual structural failure are incredibly rare in real-world environments like the openLAB bridge. Training a heavily parameterized MLP on such sparse data guarantees severe overfitting. We construct a Hybrid Quantum-Classical Neural Network (HQCNN) to directly bypass this limitation. Instead of routing features through classical dense layers, this architecture feeds the CNN output into a Variational Quantum Circuit (VQC). This maps the spatial data into an exponentially larger Hilbert space where complex patterns become linearly separable.

#### **7.1 Quantum State Encoding**
Classical data cannot enter a Quantum Processing Unit directly. The system must first embed the continuous latent vector $z$ into a formal quantum state. We execute this mathematical transformation using Angle Encoding. This technique takes every scalar component $z_i$ and translates it into the physical rotational angle of a dedicated qubit, beginning from the absolute computational ground state $|0\rangle$. A sequence of $R_y$ rotation gates then generates the final encoded state $|\psi\rangle$.

$$|\psi(z)\rangle = \bigotimes_{i=1}^{N} R_y(z_i) |0\rangle$$

where:

| Symbol | Definition |
| :--- | :--- |
| $|\psi(z)\rangle$ | Quantum state vector after encoding classical data $z$ |
| $\bigotimes$ | Tensor product operator for constructing multi-qubit states |
| $N$ | Dimensionality of the latent vector $z$ (and total number of qubits) |
| $R_y(z_i)$ | Quantum rotation gate around the Y-axis by angle $z_i$ (Angle Encoding) |
| $z_i$ | The $i$-th scalar component of the classical latent vector $z$ |
| $|0\rangle$ | The computational basis ground state for a single qubit |

While mathematically concise, this operation triggers a profound structural expansion. It forces a classical vector of length $N$ into a $2^N$-dimensional complex Hilbert space. The resulting geometric environment is vast; and it allows the subsequent quantum operations to detect subtle structural anomalies that classical networks completely ignore.

#### **7.2 The Variational Quantum Circuit (Ansatz) and Measurement**
After the network encodes the structural data, a parameterized unitary matrix $U(\theta)$ processes the quantum state. Physicists refer to this specific circuit layout as the Ansatz. It weaves together highly entangling CNOT operations with rotational gates, and a vector of classical weights $\theta$ controls these rotations during training. We must eventually collapse this quantum state to retrieve a usable prediction. The system extracts the final damage probability by calculating the expectation value of the Pauli-Z observable $\hat{Z}$ across the target qubits. The following equation defines the complete forward pass of this quantum head:

$$f(z; \theta) = \langle \psi(z) | U^\dagger(\theta) \hat{Z} U(\theta) | \psi(z) \rangle$$

where:

| Symbol | Definition |
| :--- | :--- |
| $f(z; \theta)$ | Final classification output (probability of structural damage) |
| $\langle \cdot \| \cdot \| \cdot \rangle$ | Quantum expectation value of the measurement operator |
| $|\psi(z)\rangle$ | Encoded quantum state vector |
| $U^\dagger(\theta)$ | Conjugate transpose (adjoint) of the unitary matrix |
| $\hat{Z}$ | Pauli-Z observable operator measured at the end of the circuit |
| $U(\theta)$ | Parameterized unitary matrix representing the quantum Ansatz |
| $\theta$ | Vector of trainable classical parameters (rotational weights) |

#### **7.3 Parameter Efficiency and The Small Data Problem**
The HQCNN architecture fundamentally solves the issue of parameter bloat. Distinguishing a harmless thermal expansion from a critical structural micro-crack forces classical dense layers to rely on massive, data-hungry weight matrices. The Variational Quantum Circuit bypasses this brute-force approach; it resolves decision boundaries by entangling qubits directly within the Hilbert space. This mechanism requires an exponentially smaller number of trainable parameters. Because the network remains lightweight, it aggressively resists overfitting even when fed highly restricted datasets. Civil engineering diagnostics inherently suffer from a lack of labeled failure data, making this quantum-enhanced parameter efficiency mathematically invaluable.

### Nomenclature & Mathematical Glossary

#### 1. 1D Sequential Modeling (BiLSTM)

| Symbol | Definition |
| :--- | :--- |
| $t$ | Discrete time step index |
| $x_t$ | Input vibration data vector at time $t$ |
| $h_t$ | Hidden state vector at time $t$ (the network's short-term memory) |
| $C_t$ | Cell state vector at time $t$ (the network's long-term memory) |
| $\tilde{C}_t$ | Candidate cell state generated at time $t$ |
| $f_t$ | Forget gate activation vector (controls what is dropped from $C_{t-1}$) |
| $i_t$ | Input gate activation vector (controls what is added to $C_t$) |
| $o_t$ | Output gate activation vector (controls what is passed to $h_t$) |
| $W_f, W_i, W_C, W_o$ | Learnable weight matrices for their respective gates |
| $b_f, b_i, b_C, b_o$ | Learnable bias vectors for their respective gates |
| $\sigma(\cdot)$ | Sigmoid activation function, mapping values to $(0, 1)$ |
| $\tanh(\cdot)$ | Hyperbolic tangent activation function, mapping values to $(-1, 1)$ |
| $\odot$ | Hadamard product (element-wise multiplication) |
| $\oplus$ | Vector concatenation operator |
| $\overrightarrow{h}_t, \overleftarrow{h}_t$ | Forward and backward hidden states in a BiLSTM |

#### 2. Time-Frequency Transformations (STFT & CWT)

| Symbol | Definition |
| :--- | :--- |
| $x(t)$ | Continuous raw vibration signal in the time domain |
| $w(t)$ | Localized sliding window function (e.g., Hanning or Gaussian window) |
| $\tau$ | Time translation parameter (the center of the sliding window) |
| $\omega$ | Angular frequency variable |
| $X(\tau, \omega)$ | Complex-valued STFT output matrix |
| $S(\tau, \omega)$ | Real-valued Spectrogram (energy density matrix) |
| $W(a, b)$ | Wavelet coefficients (2D scalogram) |
| $a$ | Scale parameter (inversely proportional to frequency) |
| $b$ | Translation parameter (temporal position) |
| $\psi^*$ | Complex conjugate of the mother wavelet function |

#### 3. Spatial Feature Extraction (CNN Backbone)

| Symbol | Definition |
| :--- | :--- |
| $S$ | Input 2D spectrogram matrix |
| $K$ | Learnable 2D convolutional filter (kernel) matrix |
| $Y(i, j)$ | Output feature map evaluated at spatial coordinates $(i, j)$ |
| $m, n$ | Spatial dimensions of the convolutional kernel (e.g., $3 \times 3$) |
| $b$ | Learnable bias scalar for the convolutional filter |
| $z$ | Low-dimensional continuous latent feature vector extracted by the CNN |
| $N$ | Dimensionality of the latent vector $z$ (and correspondingly, the number of qubits) |

#### 4. Hybrid Quantum Architecture (HQCNN)

| Symbol | Definition |
| :--- | :--- |
| $\vert \psi \rangle$ | Quantum state vector in a complex Hilbert space |
| $\vert 0 \rangle$ | The computational basis ground state for a single qubit |
| $z_i$ | The $i$-th scalar component of the classical latent vector $z$ |
| $R_y(\phi)$ | Quantum rotation gate around the Y-axis by angle $\phi$. Used here for Angle Encoding |
| $\bigotimes$ | tensor product operator, used to build multi-qubit states |
| $\theta$ | vector of trainable classical parameters within the quantum circuit |
| $U(\theta)$ | parameterized unitary matrix (the variational quantum "Ansatz") |
| $U^\dagger(\theta)$ | the conjugate transpose (adjoint) of the unitary matrix |
| $\hat{Z}$ | the Pauli-Z observable operator, measured to determine class probabilities |
| $\langle \psi \vert \hat{Z} \vert \psi \rangle$ | the quantum expectation value of the observable $\hat{Z}$ given state $\vert\psi\rangle$ |
| $f(z; \theta)$ | the final classification output of the Hybrid Quantum Network |