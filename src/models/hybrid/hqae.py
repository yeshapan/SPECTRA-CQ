import torch
import torch.nn as nn
from src.models.classical.cae import ClassicalAutoencoder
from src.models.hybrid.quantum_layer import VQCTorchLayer

class HybridQuantumAutoencoder(nn.Module):
    """
    The master HQAE architecture.
    Imports the identical convolutional components from the baseline, but replaces 
    the dense bottleneck with the Variational Quantum Circuit (VQC) wrapper.
    """
    def __init__(self, latent_dim=8, n_quantum_layers=3):
        super(HybridQuantumAutoencoder, self).__init__()
        
        # Instantiate the baseline model to rip its classical components
        cae = ClassicalAutoencoder(latent_dim=latent_dim)
        
        # 1. Classical Encoder (Identical to Baseline)
        self.encoder_conv = cae.encoder_conv
        
        # 2. Pre-Quantum Compression (Down-projection)
        self.encoder_linear = cae.encoder_linear 
        
        # 3. The Quantum Bottleneck (Replaces the classical latent space)
        self.quantum_bottleneck = VQCTorchLayer(n_layers=n_quantum_layers)
        
        # 4. Post-Quantum Decompression (Up-projection)
        self.decoder_linear = cae.decoder_linear
        
        # 5. Classical Decoder (Identical to Baseline)
        self.decoder_conv = cae.decoder_conv

    def to(self, *args, **kwargs):
        """
        Intercept PyTorch's .to() device mapping.
        Forces the classical spatial blocks to the GPU, but physically pins
        the PennyLane VQC and its trainable weights to the CPU to prevent state-vector crashes.
        """
        super(HybridQuantumAutoencoder, self).to(*args, **kwargs)
        self.quantum_bottleneck.to(torch.device('cpu'))
        return self

    def forward(self, x):
        # Squeeze through the spatial hierarchy
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically (B, C*H*W)
        
        # Compress to 8-dimensions for the qubits
        x = self.encoder_linear(x)
        
        # The Device Bridge
        # 1. Store original GPU device
        original_device = x.device
        
        # 2. Move data to CPU for PennyLane
        x_cpu = x.to(torch.device('cpu'))
        
        # Pass through the state-vector simulator (Measurement outputs [-1, 1])
        x_quantum = self.quantum_bottleneck(x_cpu)
        
        # 3. Move back to GPU for classical decoding
        x = x_quantum.to(original_device)
        
        # Reconstruct the spatial tensors
        x = self.decoder_linear(x)
        x = x.view(x.size(0), 32, 8, 25)  # Unflatten dynamically
        x = self.decoder_conv(x)
        
        return x