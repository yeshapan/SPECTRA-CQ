import os
import numpy as np
import torch
from typing import List, Tuple

def inject_ambient_noise(spectrogram: np.ndarray, noise_factor: float = 0.02) -> np.ndarray:
    """
    Applies global environmental baseline variance.
    
    Gaussian noise is injected across all sensor channels simultaneously. 
    This mathematically simulates routine, global environmental variance (eg: wind load, 
    traffic vibrations) that affects the entire macroscopic structure.
    """
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=spectrogram.shape)
    noisy_spec = spectrogram + noise
    
    return noisy_spec

def inject_structural_damage(spectrogram: np.ndarray, mask_width: int = 2, target_sensor_idx: int = 0) -> np.ndarray:
    """
    Simulates localized structural degradation by masking a dynamic frequency band.
    Always starts at scale index 45 to represent high-frequency stiffness loss.
    
    Multi-sensor Spatial Update:
    - The micro-cracking signature (frequency mask) is strictly isolated to a single physical node (default: index 0 / PE11).
    - Surrounding sensors remain physically intact.
    - The model must rely on the geometric entanglement of the bridge_graph to detect this spatial anomaly.
    """
    damaged_spec = spectrogram.copy()
    start_idx = 45
    end_idx = min(start_idx + mask_width, spectrogram.shape[-2]) # Ensure we don't go out of bounds
    
    # Frequency Band Masking (Strictly Localized)
    # The anomaly is mathematically confined to the target physical node `PE11` if multi-channel (Phase 4) or applied globally if single-channel (Phase 3)
    if damaged_spec.shape[0] == 6:
        damaged_spec[target_sensor_idx, start_idx:end_idx, :] = 0.0
    else:
        # Fallback for Phase 3 (1, 64, 1000) or (64, 1000)
        if damaged_spec.ndim == 3:
            damaged_spec[0, start_idx:end_idx, :] = 0.0
        else:
            damaged_spec[start_idx:end_idx, :] = 0.0
            
    return damaged_spec

def prepare_evaluation_tensors(file_paths: List[str], ambient_noise: float = 0.02, damage_width: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads raw multi-channel spectrograms, applies equal environmental baselines, 
    generates synthetic damage equivalents, and stacks them into PyTorch tensors ready for spatial inference.
    
    CRITICAL FIX: 
    Both healthy and damaged tensors now receive the exact same ambient noise floor.
    This forces the model to ignore global weather variance and isolate the geometric discontinuity.
    
    Args:
        file_paths: List of paths to the .npy spectrogram files.
        ambient_noise: Fixed global noise factor (weather baseline).
        damage_width: Dynamic localized frequency mask width (damage severity).
        
    Returns:
        Tuple containing (healthy_tensor, damaged_tensor) of shape (B, C, 64, 1000)
    """
    healthy_tensors = []
    damaged_tensors = []

    for file_path in file_paths: # Loops through the unseen validation spectrograms
        raw_spec = np.load(file_path) # Shape could be (6, 64, 1000) or (64, 1000)
        
        # Ensure channel dimension exists for Phase 3 arrays
        if raw_spec.ndim == 2:
            raw_spec = np.expand_dims(raw_spec, axis=0)
            
        # 1. Apply baseline environmental noise to the raw state
        noisy_base = inject_ambient_noise(raw_spec, noise_factor=ambient_noise)
        
        # 2. Normalize the noisy baseline -> This is our True Healthy State
        # Global Min-Max normalization preserves spatial relative magnitudes
        spec_min, spec_max = noisy_base.min(), noisy_base.max() 
        healthy_spec = (noisy_base - spec_min) / (spec_max - spec_min) if spec_max > spec_min else noisy_base
        
        # 3. Apply localized damage to the noisy baseline
        damaged_unnorm = inject_structural_damage(noisy_base, mask_width=damage_width)
        
        # 4. Normalize the damaged state -> This is our True Damaged State
        d_min, d_max = damaged_unnorm.min(), damaged_unnorm.max()
        damaged_spec = (damaged_unnorm - d_min) / (d_max - d_min) if d_max > d_min else damaged_unnorm
        
        healthy_tensors.append(healthy_spec) 
        damaged_tensors.append(damaged_spec) 

    # Stack into Batches
    t_healthy = torch.tensor(np.array(healthy_tensors), dtype=torch.float32)
    t_damaged = torch.tensor(np.array(damaged_tensors), dtype=torch.float32)
    
    return t_healthy, t_damaged