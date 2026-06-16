import os
import logging
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
        """Executes a single pass over the training dataset."""
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimization
            self.optimizer.zero_grad(set_to_none=True) # Slightly faster than standard zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        return running_loss / len(dataloader.dataset)

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Evaluates model on validation set without tracking gradients."""
        self.model.eval()
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
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
            
            logging.info(f"Epoch {epoch:03d}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {current_lr:.2e}")
            
            # Checkpoint mechanism: Only save weights if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(save_dir, "best_cae_model.pth")
                torch.save(self.model.state_dict(), save_path)
                logging.info(f"New optimal weights saved to {save_path}")
                
        logging.info("Training complete.")