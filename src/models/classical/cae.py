import torch
import torch.nn as nn

class ClassicalAutoencoder(nn.Module):
    """
    Baseline Convolutional Autoencoder for CWT spectrogram reconstruction
    
    Input shape: (B, 1, 64, 1000) -> (Batch, Channels, Scales, Time)
    The architecture utilizes asymmetric pooling to aggressively downsample the time 
    domain while preserving the narrower frequency domain
    """
    def __init__(self, latent_dim: int = 8):
        super(ClassicalAutoencoder, self).__init__()
        
        # ENCODER: Compress (1, 64, 1000) -> (latent_dim)
        self.encoder_conv = nn.Sequential(
            # Conv1: (B, 1, 64, 1000) -> (B, 16, 64, 1000)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Pool1: Reduces time/freq -> (B, 16, 32, 250) (kernel 2x4)
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            
            # Conv2: (B, 16, 32, 250) -> (B, 32, 32, 250)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Pool2: Reduces time/freq -> (B, 32, 8, 25) (kernel 4x10)
            nn.MaxPool2d(kernel_size=(4, 10), stride=(4, 10)),
        )
        
        # Flattened size calculation: 32 channels * 8 height * 25 width = 6400
        self.flatten_size = 32 * 8 * 25
        
        # The Latent Bottleneck. Simulates the information bottleneck of a quantum circuit.
        self.encoder_linear = nn.Linear(self.flatten_size, latent_dim)

        # DECODER: Reconstruct (latent_dim) -> (1, 64, 1000)
        self.decoder_linear = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder_conv = nn.Sequential(
            # Transpose1: Upsample -> (B, 16, 32, 250)
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 10), stride=(4, 10)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Transpose2: Upsample -> (B, 1, 64, 1000)
            nn.ConvTranspose2d(16, 1, kernel_size=(2, 4), stride=(2, 4)),
            
            # Sigmoid activation since target inputs are normalized to [0, 1]
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass defining the computational graph
        """
        # Encode
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten (B, C*H*W)
        latent_vector = self.encoder_linear(x)
        
        # Decode
        x = self.decoder_linear(latent_vector)
        x = x.view(x.size(0), 32, 8, 25)  # Unflatten back to spatial dimensions
        reconstruction = self.decoder_conv(x)
        
        return reconstruction