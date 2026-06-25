# Module to handle the inference loop (after synthetic damaged data injection)
# We separate this from trainer.py because evaluation does not require optimizers, schedulers or backpropagation.

import torch
import torch.nn as nn
import numpy as np

class AnomalyEvaluator:
    """
    Handles inference for trained Autoencoders to calculate Anomaly Scores (MSE).
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        # reduction='none' allows us to get the MSE per individual sample, not the batch average
        self.criterion = nn.MSELoss(reduction='none')

    def get_anomaly_scores(self, data_tensor: torch.Tensor) -> np.ndarray:
        """
        Passes a tensor through the frozen model and calculates the reconstruction error.
        
        Args:
            data_tensor (torch.Tensor): Shape (B, C, H, W)
            
        Returns:
            np.ndarray: A 1D array of MSE scores for each sample in the batch.
        """
        with torch.no_grad():
            inputs = data_tensor.to(self.device)
            outputs = self.model(inputs)
            
            # Calculate MSE and average across the spatial dimensions (C, H, W)
            loss = self.criterion(outputs, inputs)
            sample_mse = loss.view(inputs.size(0), -1).mean(dim=1)
            
        return sample_mse.cpu().numpy()