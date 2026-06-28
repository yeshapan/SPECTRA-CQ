import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset for SHM Continuous Wavelet Transform matrices.
    
    Operates in an unsupervised context: __getitem__ returns the input 
    matrix as both the feature (x) and the target (y) for reconstruction loss.
    """
    def __init__(self, data_dir: str):
        self.file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
        if not self.file_paths:
            raise RuntimeError(f"No .npy files found in {data_dir}. Run preprocess.py first.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Disk IO bottleneck: mmap_mode='r' reads directly from disk without loading full array
        # This is critical if the dataset grows beyond system RAM
        # Phase-4 Update: spec is now shaped (6, 64, 1000)
        spec = np.load(self.file_paths[idx])
        
        # Min-Max Normalization
        # Phase-4 Update: Computed the min/max across the ENTIRE 6-channel array globally.
        # CRITICAL: If we normalize each channel independently, we destroy the relative energy 
        # differences between sensors (for example: erasing the physical fact that PE11 is closer to the load).
        spec_min, spec_max = spec.min(), spec.max()
        if spec_max > spec_min:
            spec = (spec - spec_min) / (spec_max - spec_min)
            
        # Convert to Tensor
        # Input shape expected by PyTorch Conv2d: (Channels, Height, Width)
        # Phase-4 Update: Removed the .unsqueeze(0) call (done previously to add a channel dimension) because now the array is natively (6, 64, 1000).
        tensor_spec = torch.tensor(spec, dtype=torch.float32)
        
        return tensor_spec, tensor_spec

def get_dataloaders(data_dir: str, batch_size: int = 32, train_split: float = 0.8):
    """
    Generates training and validation DataLoader iterables.
    
    Args:
        data_dir: Directory containing .npy files
        batch_size: Defines (B) dimension in (B, C, H, W)
    """
    dataset = SpectrogramDataset(data_dir)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Generator seed fixed for deterministic splits (adhering to our 3-seed protocol)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    # pin_memory=True speeds up CPU-to-GPU memory transfer
    # num_workers=2 parallelizes disk I/O
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    
    return train_loader, val_loader