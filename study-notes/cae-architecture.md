# Classical Convolutional Autoencoder (CAE) Architecture and Concepts

### **Pre-Note: What Does the Input Data Actually Look Like?**
Before understanding the neural network, we must understand exactly what we are feeding it. 
We have transitioned from 1D time-series acceleration (raw vibration) into 2D Continuous Wavelet Transform (CWT) matrices.

**Analogy**: A standard greyscale image is stored on a hard drive as a 2D matrix of numbers. 
* The position of a number in the matrix represents the physical X/Y location of a pixel
* The value of the number represents the intensity or brightness of that pixel

**Spectrograms** are stored exactly the same way mathematically, but the physical meaning of the grid is entirely different:
* **Rows (vertical axis):** Represent *Frequency* $\rightarrow$ 0.5 Hz to 100 Hz mapped across 64 discrete rows.
* **Columns (horizontal axis):** Represent *Time* $\rightarrow$ a 2-second burst sampled at 500 Hz gives us 1000 columns.
* **"Pixel" Value:** Represents the *Vibration Energy* at that specific frequency and at that exact millisecond. 

**The Mathematical Reality:**
On the hard drive (and in the GPU's memory) $\rightarrow$ our spectrogram is a matrix of size `64 × 1000`.
Through Min-Max normalization, every single energy value is bounded between `0.0` and `1.0`. 
* A value of `0.0` (black) means the bridge is completely still at that frequency.
* A value of `1.0` (white) means the bridge is experiencing maximum resonant vibration at that frequency.

---

### **Pre-Note: What is an Autoencoder?**
Autoencoder is an unsupervised neural network.

**Intuitive Walkie-Talkie Analogy:**
Imagine you have a highly detailed CAD blueprint of a bridge and you need to send it to your colleague. But you only have a walkie-talkie. You cannot transmit the entire drawing.

You are forced to summarize it into a few critical dimensions (length, main girder thickness, pillar spacing). If your colleague can redraw a nearly perfect blueprint based *only* on those few numbers $\implies$ your summary was perfect.

In SHM, we only train the autoencoder on **healthy** bridge data. 
1. It learns to summarize the "normal" vibration patterns.
2. It learns to reconstruct the spectrogram from that summary.
3. **The Anomaly Detection:** If a structural crack occurs, the frequencies shift. When we feed this "damaged" spectrogram into the model $\rightarrow$ it tries to summarize it using the rules it learned for a healthy bridge. The reconstruction will fail drastically. We can quantify the structural damage by measuring the error (MSE) between the original input and the failed reconstruction.

---

### **Note: Dimensionality Crisis in CWT Spectrograms**
* A standard Continuous Wavelet Transform (CWT) output in our pipeline represents a 2-second vibration burst.
* The shape is $(64, 1000)$ $\rightarrow$ $64$ log-spaced frequency scales (0.5 Hz to 100 Hz) and $1,000$ time steps (500 Hz sampling rate).
* Total features per sample $\rightarrow$ $64 \times 1000 = 64,000$ classical floating-point values.
* **The Goal**: To aggressively compress these $64,000$ features down to exactly $8$ continuous values. 
* **Why 8?** Because current NISQ-era quantum simulators (for HQAE) cannot handle massive qubit counts without simulation times exploding. An 8-qubit bottleneck requires exactly 8 classical input features.

## **Architecture of CAE**
### **1. The Spatial Encoder (Compression)**

1. `Conv2d`:
    * Slides "kernel" across spectrogram to detect patterns
    * Example:
        * The first layer might detect simple horizontal lines (constant hums at a specific frequency).
        * Deeper layers combine these to detect complex, macroscopic frequency couplings (e.g., how the main span's vibration interacts with a specific cross-beam). 

2. `MaxPool2d`: 
    * Severe downsampling mechanism
    * Discards the high-frequency, low-energy stochastic environmental noise (e.g., noise from wind or a small car passing)
    * Strictly preserves the most dominant, loudest structural resonant frequencies.

### **2. The Classical Bottleneck (Latent Space)**
After passing through the Convolutional and Pooling layers, our massive `(1, 64, 1000)` matrix has been spatially crushed into 32 tiny feature maps of size `8 × 25`. 

If we unroll these maps into a single flat line $\rightarrow$ we get 6,400 numbers. 

We then pass these 6,400 numbers through a `Linear` layer (a standard, fully connected neural network layer) that forces them down into just 8 continuous numbers.

**Why extreme compression?**
This is the "walkie-talkie" restriction. By forcing 64,000 original data points ($64 \times 1000$) down to just 8 numbers $\rightarrow$ we force the network to discard everything except the absolute most mathematically critical components of the bridge's structural behavior.

These 8 numbers are the "Latent Vector" $\leftarrow$ the fundamental mathematical DNA of the concrete's integrity.

*Note: In Phase 4, this exact classical bottleneck is violently ripped out and replaced by the 8-Qubit Variational Quantum Circuit.*

### **3. The Spatial Decoder (Reconstruction)**
* Transpose Convolution (`ConvTranspose2d`)
    * Also called "deconvolution" 
    * This is the mathematical reverse of the spatial encoder.
    * It takes the dense, compressed features and projects them back outward into a larger spatial grid. 
    * Learns how to "paint" the specific frequency bands back into their correct time slots based on the 8-number summary.

* Terminal Sigmoid Activation
    * The final layer of the network applies a mathematical `Sigmoid` function.
    * This guarantees all output values are squashed strictly between `0.0` and `1.0`. 
    * Input energy values were also normalized to this exact same range $\implies$ we can mathematically compare Input andOutput pixel-by-pixel to calculate the reconstruction error.

### **4. Loss Function & Optimization Landscape**
* **Objective**: Mean Squared Error (MSE).
    $$MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$$
    * Where $Y$ is the original CWT matrix, and $\hat{Y}$ is the reconstructed matrix.
* **Why not Cross-Entropy?** We are not doing discrete classification (e.g., Cat vs. Dog). We are reconstructing continuous signal amplitudes.
* **The Optimization Strategy (`Adam`)**: 
    * The optimization landscape of an 8-dimensional bottleneck is extremely sharp.
    * Standard Stochastic Gradient Descent (SGD) would likely get stuck.
    * `Adam` dynamically adapts the learning rate for each individual parameter based on the first and second moments of the gradients, allowing the network to navigate narrow ravines in the loss landscape.


## **Technical Implementation Summary (Tensor Journey)**

The software engineering implementation for this specific architecture is strictly isolated within the `ClassicalAutoencoder` class located at:
- **File:** `src/models/classical/cae.py`
- **Method:** `def forward(self, x: torch.Tensor) -> torch.Tensor:`

Below is the exact flow of tensor dimensionality and the explicit PyTorch layer configurations executed during a single forward pass:

**1. Input Phase**
* Input Tensor (`x`): `torch.Tensor`
* Shape: `(Batch, 1 Channel, 64 Height, 1000 Width)`
* Data Type: `torch.float32` (Bounded `[0.0, 1.0]`)

**2. Spatial Encoding (Compression)**
* Layer: `nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)`
  * Shape Mutated: $\rightarrow$ `(B, 16, 64, 1000)`
* Layer: `nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))`
  * Shape Mutated: $\rightarrow$ `(B, 16, 32, 250)`
  * Downsampled frequency by a factor of 2 (64 $\rightarrow$ 32)
  * Downsampled time by a factor of 4 (1000 $\rightarrow$ 250)
* Layer: `nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)`
  * Shape Mutated: $\rightarrow$ `(B, 32, 32, 250)`
* Layer: `nn.MaxPool2d(kernel_size=(4, 10), stride=(4, 10))`
  * Shape Mutated: $\rightarrow$ `(B, 32, 8, 25)`
  * Downsampled frequency by a factor of 4 (32 $\rightarrow$ 8)
  * Downsampled time by a factor of 10 (250 $\rightarrow$ 25)

**3. The Classical Bottleneck (Latent Space)**
* Operation: `x.view(x.size(0), -1)` (Dynamic Flattening)
  * Shape Mutated: $\rightarrow$ `(B, 6400)`
* Layer: `nn.Linear(in_features=6400, out_features=8)`
  * Shape Mutated: $\rightarrow$ `(B, 8)` (This is the 8-dimensional Continuous Latent Vector)

**4. Spatial Decoding (Reconstruction)**
* Layer: `nn.Linear(in_features=8, out_features=6400)`
  * Shape Mutated: $\rightarrow$ `(B, 6400)`
* Operation: `x.view(x.size(0), 32, 8, 25)` (Dynamic Reshaping)
  * Shape Mutated: $\rightarrow$ `(B, 32, 8, 25)`
* Layer: `nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(4, 10), stride=(4, 10))`
  * Shape Mutated: $\rightarrow$ `(B, 16, 32, 250)`
* Layer: `nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(2, 4), stride=(2, 4))`
  * Shape Mutated: $\rightarrow$ `(B, 1, 64, 1000)`

**5. Output Phase**
* Layer: `nn.Sigmoid()`
* Return Type: `torch.Tensor` of shape `(B, 1, 64, 1000)`
* **Loss Computation:** This predicted tensor is compared against the original input tensor using `nn.MSELoss()` within the `trainer.py` loop.