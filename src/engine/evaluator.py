import torch
import torch.nn as nn
from typing import List, Tuple
from tqdm import tqdm
import numpy as np

class AnomalyEvaluator:
    """
    Handles inference and scoring. Modified for Target Node Isolation.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        # reduction='none' is critical to isolate specific spatial channels
        self.criterion = nn.MSELoss(reduction='none')

    def get_anomaly_scores(self, data_tensor: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """
        Calculates the Reconstruction Error strictly at the Target Node (PE11).
        Provides a mathematically identical baseline for 1-sensor vs 6-sensor architectures.
        """
        dataset = torch.utils.data.TensorDataset(data_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                
                # loss shape: (B, C, H, W)
                loss = self.criterion(outputs, inputs)
                
                # CRITICAL FIX: Target Node Isolation
                # Extract ONLY channel 0 (PE11). 
                # In Phase 3, this is the only channel. In Phase 4, this ignores the 5 healthy channels.
                target_node_loss = loss[:, 0, :, :] # Shape: (B, H, W)
                
                # Average the spatial MSE just for PE11
                sample_mse = target_node_loss.reshape(inputs.size(0), -1).mean(dim=1) # Shape: (B,)
                
                all_scores.extend(sample_mse.cpu().numpy())
                
        return np.array(all_scores)