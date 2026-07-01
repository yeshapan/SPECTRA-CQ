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
from src.models.hybrid.hqae import HybridQuantumAutoencoder
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
        
        # Multi-sensor fusion spatial update: 
        # Parse Siamese parameters to configure the multi-sensor graph layout. 
        # Fallbacks (1 sensor, 8 latents) ensure strict backwards compatibility with Phase 3 baselines.
        num_sensors = config.get("num_sensors", 1)
        latent_dim = config.get("latent_dim", 8)
        latent_dim_per_sensor = config.get("latent_dim_per_sensor", latent_dim)
        
        # Dynamically select the architecture based on the config file name
        if "hqae" in config_path.lower():
            logging.info("Instantiating Hybrid Quantum Autoencoder (HQAE)..")
            model = HybridQuantumAutoencoder(
                latent_dim_per_sensor=latent_dim_per_sensor,
                num_sensors=num_sensors, 
                n_quantum_layers=config.get("n_quantum_layers", 3),
                topology=config.get("topology", "basic")
            )
        else:
            logging.info("Instantiating Classical Autoencoder (CAE)..")
            model = ClassicalAutoencoder(
                latent_dim_per_sensor=latent_dim_per_sensor,
                num_sensors=num_sensors
            )
        
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