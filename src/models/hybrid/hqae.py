import torch
import torch.nn as nn
from src.models.classical.cae import ClassicalAutoencoder
from src.models.hybrid.quantum_layer import VQCTorchLayer

class HybridQuantumAutoencoder(nn.Module):
    """
    The master HQAE architecture.
    Universal design: Imports the identical convolutional components from the CAE, 
    and dynamically routes either 1 sensor (Phase 3) or 6 sensors (Phase 4) 
    through the Variational Quantum Circuit (VQC) wrapper.
    """
    def __init__(self, latent_dim_per_sensor=2, num_sensors=6, n_quantum_layers=3, topology="basic"):
        super(HybridQuantumAutoencoder, self).__init__()
        
        self.num_sensors = num_sensors
        self.latent_dim_per_sensor = latent_dim_per_sensor
        
        # Calculate qubits dynamically to support both Phase 3 (e.g., 2 or 8) and Phase 4 (12)
        self.n_qubits = self.num_sensors * self.latent_dim_per_sensor
        
        # Instantiate the baseline model to rip its classical components
        cae = ClassicalAutoencoder(latent_dim_per_sensor=latent_dim_per_sensor, num_sensors=num_sensors)
        
        # 1. Classical Encoder (Identical to Baseline)
        self.encoder_conv = cae.encoder_conv
        
        # 2. Pre-Quantum Compression (Down-projection)
        self.encoder_linear = cae.encoder_linear 
        
        # 3. The Quantum Bottleneck (Replaces the classical latent space)
        # We pass the dynamic n_qubits down to avoid hardcoded Phase 4 collisions
        self.quantum_bottleneck = VQCTorchLayer(
            n_qubits=self.n_qubits,
            n_layers=n_quantum_layers,
            topology=topology
        )
        
        # 4. Post-Quantum Expansion (Up-projection)
        self.decoder_linear = cae.decoder_linear
        
        # 5. Classical Decoder
        self.decoder_conv = cae.decoder_conv

    def to(self, *args, **kwargs):
        """
        Overrides the default .to() method. 
        Ensures the classical layers move to GPU, but forces the VQC to stay on CPU 
        to prevent state-vector crashes in PennyLane.
        """
        super(HybridQuantumAutoencoder, self).to(*args, **kwargs)
        self.quantum_bottleneck.to(torch.device('cpu'))
        return self

    def forward(self, x):
        # CRITICAL FIX: Auto-inject missing channel dimension for single-sensor Phase 3 data
        if x.dim() == 3:
            x = x.unsqueeze(1) # Converts (B, H, W) -> (B, 1, H, W)
        
        B, C, H, W = x.size()
        
        # Siamese Spatial Compression: (B*C, 1, H, W)
        x = x.view(B * C, 1, H, W)
        
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically (B*C, Flatten_Size)
        
        # Compress down to latent dimensions per sensor
        x = self.encoder_linear(x)
        
        # Graph Assembly: Flatten the independent sensor features into the global quantum space
        # Shape becomes (B, C * latent_dim) -> e.g. Phase 3: (B, 2), Phase 4: (B, 12)
        x = x.view(B, C * self.latent_dim_per_sensor)
        
        # The Device Bridge
        # 1. Store original GPU device
        original_device = x.device
        
        # 2. Move data to CPU for PennyLane
        x_cpu = x.to(torch.device('cpu'))
        
        # Pass through the state-vector simulator (Measurement outputs [-1, 1])
        x_quantum = self.quantum_bottleneck(x_cpu)
        
        # 3. Move back to GPU for classical decoding
        x = x_quantum.to(original_device)
        
        # Deconstruct the graph back to isolated sensor features for the decoder
        x = x.view(B * C, self.latent_dim_per_sensor)
        
        # Reconstruct the spatial blocks
        x = self.decoder_linear(x)
        
        # Reshape for transpose convolutions using dynamic batch sizing
        x = x.view(x.size(0), 32, 8, 25) 
        
        x = self.decoder_conv(x)
        
        # Unfold back to multi-sensor or single-sensor shape
        x = x.view(B, C, H, W)
        
        return x