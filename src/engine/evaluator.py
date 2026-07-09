import torch
import torch.nn as nn
from typing import List, Tuple
from tqdm import tqdm
import numpy as np

class AnomalyEvaluator:
    """
    Handles the inference and scoring for the frozen autoencoder models.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        # reduction='none' is critical here so we can isolate the exact spatial reconstruction errors 
        # instead of PyTorch automatically collapsing them into a single scalar scalar.
        self.criterion = nn.MSELoss(reduction='none')

    def get_anomaly_scores(self, data_tensor: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """
        Passes the dataset through the frozen architecture and returns the anomaly scores.
        
        The score is defined as the Max Spatial Reconstruction Error (MSE).
        """
        dataset = torch.utils.data.TensorDataset(data_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                
                outputs = self.model(inputs)
                
                # Tensor shape: (Batch, Channels, Height, Width) -> e.g., (32, 6, 64, 1000)
                loss = self.criterion(outputs, inputs)
                
                # CRITICAL FIX: The Spatial Dilution Problem
                # Calculate mean MSE *per sensor* (dim 2 and 3), then take the MAX across sensors (dim 1)
                # This prevents healthy sensors from mathematically diluting the anomaly of a broken sensor.
                sensor_mse = loss.mean(dim=(2, 3))     # Shape: (B, C)
                sample_mse, _ = sensor_mse.max(dim=1)  # Shape: (B,)
                
                # Move back to CPU for Scikit-Learn AUC calculations
                all_scores.extend(sample_mse.cpu().numpy())
                
        return np.array(all_scores)