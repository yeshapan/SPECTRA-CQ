import os
import numpy as np
import torch
from typing import List, Tuple

def inject_structural_damage(spectrogram: np.ndarray, mask_width: int = 2, target_sensor_idx: int = 0) -> np.ndarray:
    """
    Simulates localized structural degradation by injecting a high-frequency Acoustic Emission (AE) burst.
    Always starts at scale index 45 to represent high-frequency energy release from micro-cracking.
    
    Multi-sensor Spatial Update:
    - The micro-cracking signature (AE burst) is strictly isolated to a single physical node (default: index 0 / PE11).
    - Surrounding sensors remain physically intact.
    - The model must rely on the geometric entanglement of the bridge_graph to detect this spatial anomaly.
    """
    damaged_spec = spectrogram.copy()
    start_idx = 45
    end_idx = min(start_idx + mask_width, spectrogram.shape[-2]) # Ensure we don't go out of bounds
    
    # Frequency Band Masking (Acoustic Emission Burst)
    # We inject a 1.0 (max normalized energy) to simulate the physical snap of a crack.
    # The anomaly is mathematically confined to the target physical node `PE11` if multi-channel (Phase 4) or applied globally if single-channel (Phase 3)
    if damaged_spec.shape[0] == 6:
        damaged_spec[target_sensor_idx, start_idx:end_idx, :] = 1.0
    else:
        # Fallback for Phase 3 (1, 64, 1000) or (64, 1000)
        if damaged_spec.ndim == 3:
            damaged_spec[0, start_idx:end_idx, :] = 1.0
        else:
            damaged_spec[start_idx:end_idx, :] = 1.0
            
    return damaged_spec

def prepare_evaluation_tensors(file_paths: List[str], damage_width: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads raw multi-channel spectrograms, normalizes them, 
    generates synthetic damage equivalents, and stacks them into PyTorch tensors ready for spatial inference.
    
    CRITICAL FIX: 
    The artificial Gaussian noise injection has been removed. The models are evaluated on the 
    natural environmental noise floor present in the raw OpenLAB dataset to prevent domain shift.
    
    Args:
        file_paths: List of paths to the .npy spectrogram files.
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
            
        # 1. Normalize the raw baseline -> This is our True Healthy State
        # Global Min-Max normalization preserves spatial relative magnitudes
        spec_min, spec_max = raw_spec.min(), raw_spec.max() 
        healthy_spec = (raw_spec - spec_min) / (spec_max - spec_min) if spec_max > spec_min else raw_spec
        
        # 2. Apply localized damage to the healthy state (AE Burst injection)
        damaged_unnorm = inject_structural_damage(healthy_spec, mask_width=damage_width)
        
        # 3. Normalize the damaged state -> This is our True Damaged State
        d_min, d_max = damaged_unnorm.min(), damaged_unnorm.max()
        damaged_spec = (damaged_unnorm - d_min) / (d_max - d_min) if d_max > d_min else damaged_unnorm
        
        healthy_tensors.append(healthy_spec) 
        damaged_tensors.append(damaged_spec) 

    # Stack into Batches
    t_healthy = torch.tensor(np.array(healthy_tensors), dtype=torch.float32)
    t_damaged = torch.tensor(np.array(damaged_tensors), dtype=torch.float32)
    
    return t_healthy, t_damaged