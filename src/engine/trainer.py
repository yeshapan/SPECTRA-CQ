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
        """Executes a single pass over the training set."""
        self.model.train()
        running_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            # FIX: Safely unpack the batch if dataloader returns a list/tuple like [data, labels]
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
                
            inputs = inputs.to(self.device)
            # For autoencoders, input is also the target
            targets = inputs
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            
            # V2: Gradient Clipping (Instantiation Roulette Mitigation)
            # Clips gradients to a maximum norm of 1.0. 
            # This prevents gradient explosion which severely destabilizes the purely localized ('none') topology models 
            # and prevents wild optimization jumps during early epochs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        return running_loss / len(dataloader.dataset)

    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Evaluates the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                # FIX: Safely unpack the batch in validation loop as well
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                    
                inputs = inputs.to(self.device)
                targets = inputs
                
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
            
            # ALIGNED LOGGING: Format matches Regex: Epoch (\d+)/\d+ | Train MSE: ([\d.]+) | Val MSE: ([\d.]+)
            logging.info(f"Epoch {epoch}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {current_lr:.2e}")
            
            # Checkpoint mechanism: Only save weights if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                logging.info(f"New optimal weights saved. Validation Loss: {best_val_loss:.6f}")
                
        return best_val_loss