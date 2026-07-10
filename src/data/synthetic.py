import os
import numpy as np
import torch
from typing import List, Tuple

def inject_structural_damage(spectrogram: np.ndarray, mask_width: int = 2, target_sensor_idx: int = 0, ae_amplitude: float = 0.35) -> np.ndarray:
    """
    Simulates localized structural degradation by injecting a high-frequency Acoustic Emission (AE) burst.
    
    Multi-sensor Spatial Update (Goldilocks Calibration):
    - The micro-cracking signature (AE burst) is strictly isolated to a single physical node (default: index 0 / PE11).
    - Surrounding sensors remain physically intact.
    - ae_amplitude (0.35) is calibrated to prevent the "Sonic Boom" paradox.
    It provides a signal strong enough to survive normalization, but weak enough to require spatial fusion for reliable detection.
    """
    damaged_spec = spectrogram.copy()
    start_idx = 45
    end_idx = min(start_idx + mask_width, spectrogram.shape[-2]) 
    
    # Frequency Band Masking (Calibrated Acoustic Emission Burst)
    if damaged_spec.shape[0] == 6:
        damaged_spec[target_sensor_idx, start_idx:end_idx, :] = ae_amplitude
    else:
        # Fallback for Phase 3 (1, 64, 1000) or (64, 1000)
        if damaged_spec.ndim == 3:
            damaged_spec[0, start_idx:end_idx, :] = ae_amplitude
        else:
            damaged_spec[start_idx:end_idx, :] = ae_amplitude
            
    return damaged_spec

def prepare_evaluation_tensors(file_paths: List[str], damage_width: int = 2, ae_amplitude: float = 0.35) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads raw spectrograms, normalizes them, injects damage, and stacks them into PyTorch tensors.
    Evaluated strictly on natural environmental noise to prevent domain shift.
    """
    healthy_tensors = []
    damaged_tensors = []

    for file_path in file_paths:
        raw_spec = np.load(file_path) 
        
        if raw_spec.ndim == 2:
            raw_spec = np.expand_dims(raw_spec, axis=0)
            
        # 1. Normalize True Healthy State
        spec_min, spec_max = raw_spec.min(), raw_spec.max() 
        healthy_spec = (raw_spec - spec_min) / (spec_max - spec_min) if spec_max > spec_min else raw_spec
        
        # 2. Apply localized calibrated damage (ae_amplitude=0.35)
        damaged_unnorm = inject_structural_damage(healthy_spec, mask_width=damage_width, ae_amplitude=ae_amplitude)
        
        # 3. Normalize True Damaged State
        d_min, d_max = damaged_unnorm.min(), damaged_unnorm.max()
        damaged_spec = (damaged_unnorm - d_min) / (d_max - d_min) if d_max > d_min else damaged_unnorm
        
        healthy_tensors.append(healthy_spec) 
        damaged_tensors.append(damaged_spec) 

    t_healthy = torch.tensor(np.array(healthy_tensors), dtype=torch.float32)
    t_damaged = torch.tensor(np.array(damaged_tensors), dtype=torch.float32)
    
    return t_healthy, t_damaged