import torch
import torch.nn as nn

class ClassicalAutoencoder(nn.Module):
    """
    Universal Convolutional Autoencoder for CWT spectrogram reconstruction.
    
    Supports both Phase 3 (Single-Sensor) and Phase 4 (Multi-Sensor Siamese) natively.
    The channel and batch dimensions are dynamically folded together so any number 
    of sensors pass through the exact same convolutional filters (weight-tied parallel encoding). 
    """
    def __init__(self, latent_dim_per_sensor: int = 2, num_sensors: int = 6):
        super(ClassicalAutoencoder, self).__init__()
        
        self.latent_dim_per_sensor = latent_dim_per_sensor
        self.num_sensors = num_sensors
        
        # ENCODER: Compress (1, 64, 1000) -> Flattened Vector
        self.encoder_conv = nn.Sequential(
            # Conv1: (B*C, 1, 64, 1000) -> (B*C, 16, 64, 1000)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Pool1: Reduces time/freq -> (B*C, 16, 32, 250) (kernel 2x4)
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            
            # Conv2: (B*C, 16, 32, 250) -> (B*C, 32, 32, 250)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Pool2: Reduces time/freq -> (B*C, 32, 8, 25) (kernel 4x10)
            nn.MaxPool2d(kernel_size=(4, 10), stride=(4, 10))
        )
        
        # 32 channels * 8 height * 25 width
        self.flatten_size = 32 * 8 * 25 
        
        self.encoder_linear = nn.Linear(self.flatten_size, self.latent_dim_per_sensor)
        self.decoder_linear = nn.Linear(self.latent_dim_per_sensor, self.flatten_size)
        
        self.decoder_conv = nn.Sequential(
            # Transpose1: Upsample -> (B*C, 16, 32, 250)
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 10), stride=(4, 10)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Transpose2: Upsample -> (B*C, 1, 64, 1000)
            nn.ConvTranspose2d(16, 1, kernel_size=(2, 4), stride=(2, 4)),
            
            # Sigmoid activation since target inputs are normalized to [0, 1]
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass defining the computational graph
        """
        # CRITICAL FIX: Auto-inject missing channel dimension for single-sensor Phase 3 data
        if x.dim() == 3:
            x = x.unsqueeze(1) # Converts (B, H, W) -> (B, 1, H, W)
            
        B, C, H, W = x.size()
        
        # Fold the batch and channel dimensions to process identically via Siamese weights
        # Shape transforms dynamically based on C (1 for Phase 3, 6 for Phase 4)
        x = x.view(B * C, 1, H, W)
        
        # Encode
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically using x.size(0) which is B*C
        
        # Compress down to latent space
        x = self.encoder_linear(x)
        
        # Decode
        x = self.decoder_linear(x)
        
        # Reshape for transpose convolutions using dynamic B*C batch sizing
        x = x.view(x.size(0), 32, 8, 25) 
        
        x = self.decoder_conv(x)
        
        # Unfold back to multi-sensor or single-sensor shape
        x = x.view(B, C, H, W)
        
        return x