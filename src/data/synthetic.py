import os
import numpy as np
import torch
from typing import List, Tuple

def inject_synthetic_damage(spectrogram: np.ndarray, noise_factor: float = 0.3, mask_band: tuple = (40, 50)) -> np.ndarray:
    """
    Simulates structural degradation on a 2D CWT spectrogram.
    
    1. Adds structural noise → to simulate micro-cracking / sensor degradation.
    2. Masks a specific frequency band → to simulate stiffness loss / mode shift.
    
    Args:
        spectrogram (np.ndarray): Original healthy spectrogram.
        noise_factor (float): Intensity of the Gaussian noise.
        mask_band (tuple): (start, end) index of the frequency scales to wipe out.
        
    Returns:
        np.ndarray: The degraded, normalized spectrogram.
    """
    damaged_spec = spectrogram.copy()
    
    # 1. Inject Gaussian Noise
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=damaged_spec.shape)
    damaged_spec = damaged_spec + noise
    
    # 2. Frequency Band Masking
    damaged_spec[mask_band[0]:mask_band[1], :] = 0.0
    
    # 3. Re-normalize to [0, 1] to match the Autoencoder input bounds
    spec_min, spec_max = damaged_spec.min(), damaged_spec.max()
    if spec_max > spec_min:
        damaged_spec = (damaged_spec - spec_min) / (spec_max - spec_min)
        
    return damaged_spec

def prepare_evaluation_tensors(file_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads raw spectrograms, normalizes them, generates synthetic damage equivalents,
    and stacks them into PyTorch tensors ready for inference.
    
    Args:
        file_paths: List of paths to the .npy spectrogram files.
        
    Returns:
        Tuple containing (healthy_tensor, damaged_tensor) of shape (B, 1, 64, 1000)
    """
    healthy_tensors = []
    damaged_tensors = []

    for file_path in file_paths:
        # Load and normalize healthy spectrogram
        raw_spec = np.load(file_path)
        spec_min, spec_max = raw_spec.min(), raw_spec.max()
        healthy_spec = (raw_spec - spec_min) / (spec_max - spec_min) if spec_max > spec_min else raw_spec
        
        # Generate damaged version
        damaged_spec = inject_synthetic_damage(healthy_spec)
        
        healthy_tensors.append(healthy_spec)
        damaged_tensors.append(damaged_spec)

    # Stack into Batches: Shape (Batch, Channel, Height, Width) -> (B, 1, 64, 1000)
    t_healthy = torch.tensor(np.array(healthy_tensors), dtype=torch.float32).unsqueeze(1)
    t_damaged = torch.tensor(np.array(damaged_tensors), dtype=torch.float32).unsqueeze(1)
    
    return t_healthy, t_damaged