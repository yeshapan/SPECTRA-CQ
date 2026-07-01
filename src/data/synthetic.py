import os
import numpy as np
import torch
from typing import List, Tuple

def inject_synthetic_damage(spectrogram: np.ndarray, noise_factor: float = 0.02, mask_band: tuple = (45, 47), target_sensor_idx: int = 0) -> np.ndarray:
    """
    Simulates early-stage (incipient) structural degradation on a multi-channel CWT spectrogram.
    
    1. Adds subtle structural noise globally → to simulate baseline environmental/wind variance.
    2. Masks a very narrow frequency band locally → to simulate slight localized stiffness loss.
    
    Multi-sensor Spatial Update:
    To evaluate the multi-sensor spatial fusion:
    - The micro-cracking signature (frequency mask) is strictly isolated to a single physical node (default: index 0 / PE11)
    - Surrounding sensors remain physically intact.
    - The model must rely on the geometric entanglement of the bridge_graph to detect this spatial anomaly.
    
    Args:
        spectrogram (np.ndarray): Original healthy multi-channel spectrogram. Shape: (6, 64, 1000)
        noise_factor (float): Intensity of the Gaussian noise (Default: 0.02).
        mask_band (tuple): The (start, end) index of the frequency scales to wipe out (Default: 45 to 47).
        target_sensor_idx (int): The specific channel index to inject the structural failure.
        
    Returns:
        np.ndarray: The degraded, normalized multi-channel spectrogram.
    
    Synthetic damage must mimic realistic physical wave propagation and structural failure to test efficiency of the model:
    
    1. Global Noise: Gaussian noise is applied across all 6 sensor channels simultaneously. 
       This mathematically simulates routine, global environmental variance (eg: wind load, 
       traffic vibrations) that affects the entire macroscopic structure.
       
    2. Localized Masking: The 2-scale frequency mask (simulating early-stage stiffness loss 
       from micro-cracking) is strictly isolated to a single physical target node (eg: PE11).
       
    The Phase 4 multi-sensor Siamese model must leverage its spatial graph (inter-sensor CNOT entanglement)
    to detect the localized anomaly by cross-referencing it against the healthy baseline of the adjacent physical nodes.
    """
    damaged_spec = spectrogram.copy()
    
    # 1. Inject Gaussian Noise globally across all physical sensors
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=damaged_spec.shape)
    damaged_spec = damaged_spec + noise
    
    # 2. Frequency Band Masking (Strictly Localized)
    # The anomaly is mathematically confined to the target physical node
    damaged_spec[target_sensor_idx, mask_band[0]:mask_band[1], :] = 0.0
    
    # 3. Re-normalize globally to maintain cross-sensor energy deltas
    spec_min, spec_max = damaged_spec.min(), damaged_spec.max()
    if spec_max > spec_min:
        damaged_spec = (damaged_spec - spec_min) / (spec_max - spec_min)
        
    return damaged_spec

def prepare_evaluation_tensors(file_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads raw multi-channel spectrograms, normalizes them, generates synthetic damage equivalents
    and stacks them into PyTorch tensors ready for spatial inference.
    
    Args:
        file_paths: List of paths to the .npy spectrogram files.
        
    Returns:
        Tuple containing (healthy_tensor, damaged_tensor) of shape (B, 6, 64, 1000)
    """
    healthy_tensors = []
    damaged_tensors = []

    for file_path in file_paths: # Loops through the unseen validation spectrograms
        # Load and normalize healthy spectrogram
        raw_spec = np.load(file_path) # Shape is now (6, 64, 1000)
        
        # Global Min-Max normalization preserves spatial relative magnitudes
        spec_min, spec_max = raw_spec.min(), raw_spec.max() 
        healthy_spec = (raw_spec - spec_min) / (spec_max - spec_min) if spec_max > spec_min else raw_spec
        
        # Generate damaged version (localized to PE11)
        damaged_spec = inject_synthetic_damage(healthy_spec)
        
        healthy_tensors.append(healthy_spec) 
        damaged_tensors.append(damaged_spec) 

    # Stack into Batches: (B, 6, 64, 1000)
    # Removed the .unsqueeze(1) call since the 6-channel dimension natively satisfies the PyTorch C dimension
    t_healthy = torch.tensor(np.array(healthy_tensors), dtype=torch.float32)
    t_damaged = torch.tensor(np.array(damaged_tensors), dtype=torch.float32)
    
    return t_healthy, t_damaged