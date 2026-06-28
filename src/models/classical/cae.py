import torch
import torch.nn as nn

class ClassicalAutoencoder(nn.Module):
    """
    Baseline Convolutional Autoencoder for CWT spectrogram reconstruction
    
    Phase-4 Siamese Update: 
    The network ingests a (B, 6, 64, 1000) tensor instead of processing a single sensor.
    The channel and batch dimensions are temporarily folded together so all 6 sensors 
    pass through the exact same convolutional filters (weight-tied parallel encoding). 
    This extracts sensor-independent physical features without expanding the parameter count.
    """
    def __init__(self, latent_dim_per_sensor: int = 2, num_sensors: int = 6):
        super(ClassicalAutoencoder, self).__init__()
        
        self.latent_dim_per_sensor = latent_dim_per_sensor
        self.num_sensors = num_sensors
        
        # ENCODER: Compress (1, 64, 1000) -> Flattened Vector
        self.encoder_conv = nn.Sequential(
            # Conv1: (B*6, 1, 64, 1000) -> (B*6, 16, 64, 1000)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Pool1: Reduces time/freq -> (B*6, 16, 32, 250) (kernel 2x4)
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            
            # Conv2: (B*6, 16, 32, 250) -> (B*6, 32, 32, 250)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Pool2: Reduces time/freq -> (B*6, 32, 8, 25) (kernel 4x10)
            nn.MaxPool2d(kernel_size=(4, 10), stride=(4, 10)),
        )
        
        # Flattened size calculation: 32 channels * 8 height * 25 width = 6400
        self.flatten_size = 32 * 8 * 25
        
        # The Latent Bottleneck
        # Compresses the 6400 features of EACH sensor down to just 2 continuous dimensions
        self.encoder_linear = nn.Linear(self.flatten_size, self.latent_dim_per_sensor)

        # DECODER: Reconstruct (latent_dim_per_sensor) -> (1, 64, 1000)
        self.decoder_linear = nn.Linear(self.latent_dim_per_sensor, self.flatten_size)
        
        self.decoder_conv = nn.Sequential(
            # Transpose1: Upsample -> (B*6, 16, 32, 250)
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 10), stride=(4, 10)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Transpose2: Upsample -> (B*6, 1, 64, 1000)
            nn.ConvTranspose2d(16, 1, kernel_size=(2, 4), stride=(2, 4)),
            
            # Sigmoid activation since target inputs are normalized to [0, 1]
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass defining the computational graph
        """
        B, C, H, W = x.size()
        
        # Fold the batch and channel dimensions to process identically via Siamese weights
        # Shape transforms from (B, 6, 64, 1000) to (B * 6, 1, 64, 1000)
        x = x.view(B * C, 1, H, W)
        
        # Encode
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically (B*C, C*H*W)
        latent_vector = self.encoder_linear(x) # (B*C, 2)
        
        # Decode
        x = self.decoder_linear(latent_vector)
        x = x.view(x.size(0), 32, 8, 25)  # Unflatten back to spatial dimensions
        x = self.decoder_conv(x)
        
        # Unfold back to the global multi-channel shape (B, 6, 64, 1000)
        reconstruction = x.view(B, C, H, W)
        
        return reconstruction