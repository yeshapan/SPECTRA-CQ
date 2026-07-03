import os
import logging
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AutoencoderTrainer:
    """
    Handles the execution of the training loop for reconstruction models.
    
    Optimized for anomaly detection: Trains exclusively on the healthy baseline.
    The objective is to minimize Mean Squared Error (MSE) so the model 
    learns an extremely tight boundary around normal structural frequencies.
    """
    def __init__(self, model: nn.Module, device: torch.device, learning_rate: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        
        # MSE is the standard loss for continuous signal reconstruction
        self.criterion = nn.MSELoss()
        
        # Adam is preferred here over SGD as it adapts individual learning rates,
        # which is crucial for navigating the sharp loss landscapes of highly compressed latent spaces.
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Reduces learning rate if validation loss plateaus, preventing oscillation around the minima
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Executes a single pass over the training data."""
        self.model.train()
        running_loss = 0.0
        
        for batch in dataloader:
            # Handle both [data, labels] tuples and pure data tensors
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(self.device)
            
            # Zero gradients before forward pass
            self.optimizer.zero_grad()
            
            # Forward Pass
            outputs = self.model(inputs)
            
            # Reconstruction Loss (Difference between healthy input and output)
            mse_loss = self.criterion(outputs, inputs)
            
            # Physics-informed Neural Network (Pivot) Loss:
            total_loss = mse_loss
            
            # We dynamically check if the active model is using the soft physics topology
            if getattr(self.model, 'topology', None) == 'bridge_graph_soft':
                # Define bridge geometry locally to avoid circular imports
                SENSOR_COORDS = {
                    0: (0.0, 0.0), 1: (0.0, 1.5), 2: (0.0, 3.0),
                    3: (15.0, 0.0), 4: (15.0, 1.5), 5: (15.0, 3.0)
                }
                
                # Calculate distance-based penalties
                gamma = 0.1
                distances = []
                for s1 in range(6):
                    for s2 in range(s1 + 1, 6):
                        dist = math.sqrt((SENSOR_COORDS[s1][0] - SENSOR_COORDS[s2][0])**2 + 
                                         (SENSOR_COORDS[s1][1] - SENSOR_COORDS[s2][1])**2)
                        distances.append(dist)
                
                # e^(gamma * dist) creates a massive penalty for 15.0m, tiny for 1.5m
                physics_penalties = torch.tensor([math.exp(gamma * d) for d in distances], device=self.device)
                
                # Extract the quantum weights safely from the model
                q_weights = None
                for name, param in self.model.named_parameters():
                    # We look for the weights belonging to the quantum bottleneck
                    if 'quantum' in name.lower() or 'weight' in name.lower():
                        q_weights = param
                        break
                
                if q_weights is not None:
                    # Apply L1 penalty weighted by the physical distance
                    l1_physics_penalty = torch.sum(torch.abs(q_weights) * physics_penalties)
                    
                    # Combine (lambda_phys controls the strictness of the physics)
                    lambda_phys = 1e-4
                    total_loss = mse_loss + lambda_phys * l1_physics_penalty
            
            # Backpropagation
            total_loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += total_loss.item() * inputs.size(0)
            
        return running_loss / len(dataloader.dataset)

    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Executes a single pass over the validation data (no optimization)."""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                # Same unpacking logic for validation
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                
                # Validation always evaluates pure reconstruction MSE, regardless of PINN penalty
                loss = self.criterion(outputs, inputs)
                
                running_loss += loss.item() * inputs.size(0)
                
        return running_loss / len(dataloader.dataset)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, save_dir: str):
        """
        Executes the full training lifecycle and saves the optimal model weights.
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        logging.info(f"Initiating training on {self.device} for {epochs} epochs.")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # ALIGNED LOGGING: Format matches Regex
            logging.info(f"Epoch {epoch}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {current_lr:.2e}")
            
            # Checkpoint mechanism: Only save weights if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                logging.info(f"New optimal weights saved. Validation Loss: {best_val_loss:.6f}")
                
        return best_val_loss