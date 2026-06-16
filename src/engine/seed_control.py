import random
import numpy as np
import torch
import logging

def set_deterministic_state(seed: int = 42):
    """
    Enforces absolute determinism across all pseudo-random number generators.
    
    This is critical for the 3-seed experimental protocol to ensure that 
    variance in MSE is strictly due to the architecture not the initialization.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Forces cuDNN to use deterministic algorithms
        # This sacrifices a tiny bit of speed for perfect reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    logging.info(f"System locked to deterministic seed: {seed}")