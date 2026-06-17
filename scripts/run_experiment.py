import sys
import os
import yaml
import torch
import argparse
import logging
from typing import Dict, Any

# Bulletproof path injection: Append the parent directory (SPECTRA-CQ root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataloader import get_dataloaders
from src.models.classical.cae import ClassicalAutoencoder
from src.engine.trainer import AutoencoderTrainer
from src.engine.seed_control import set_deterministic_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: str) -> Dict[str, Any]:
    """Parses the YAML configuration file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config_path: str):
    """Master pipeline execution function"""
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Compute device allocated: {device}")
    
    seeds = config.get("seeds", [42])
    epochs = config.get("epochs", 15)
    save_dir_base = config.get("save_dir", "./checkpoints")
    
    for seed in seeds:
        logging.info(f"Starting Protocol Run | Seed: {seed}")
        
        set_deterministic_state(seed)
        
        seed_save_dir = os.path.join(save_dir_base, f"seed_{seed}")
        os.makedirs(seed_save_dir, exist_ok=True)
        
        train_loader, val_loader = get_dataloaders(
            data_dir=config["data_dir"],
            batch_size=config["batch_size"],
            train_split=config["train_split"]
        )
        
        model = ClassicalAutoencoder(latent_dim=config["latent_dim"])
        
        trainer = AutoencoderTrainer(
            model=model,
            device=device,
            learning_rate=config["learning_rate"]
        )
        
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            save_dir=seed_save_dir
        )
        
    logging.info("All seed protocols completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPECTRA-CQ Master Execution Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    main(args.config)