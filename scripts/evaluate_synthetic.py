import os
import torch
import numpy as np
import sys

# To ensure the src module can be found when running from terminal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.classical.cae import ClassicalAutoencoder
from src.models.hybrid.hqae import HybridQuantumAutoencoder
from src.data.synthetic import prepare_evaluation_tensors
from src.engine.evaluator import AnomalyEvaluator
from src.utils.viz import plot_roc_curve

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Ensemble Inference on: {device}")

    # 1. Data Ingestion (Pointing to Local NVMe)
    DATA_DIR = "/content/local_spectra_data"
    print("Fetching file list from local NVMe...")
    
    try:
        all_files = [entry.path for entry in os.scandir(DATA_DIR) if entry.name.endswith('.npy')]
    except FileNotFoundError:
        print(f"Error: Could not find directory {DATA_DIR}. Please ensure you have extracted the zip file to the local Colab storage.")
        return

    test_files = sorted(all_files)[-200:]
    print(f"Generating incipient synthetic anomalies for {len(test_files)} samples..")
    t_healthy, t_damaged = prepare_evaluation_tensors(test_files)

    # 2. Setup Architectures and Accumulators
    SEEDS = [42, 100, 2026]
    BASE_DIR = "/content/drive/MyDrive/spectra-cq-data/checkpoints"

    cae_model = ClassicalAutoencoder(latent_dim=8)
    hqae_none = HybridQuantumAutoencoder(latent_dim=8, n_quantum_layers=3, topology="none")
    hqae_strong = HybridQuantumAutoencoder(latent_dim=8, n_quantum_layers=3, topology="strong")

    eval_cae = AnomalyEvaluator(cae_model, device)
    eval_none = AnomalyEvaluator(hqae_none, device)
    eval_strong = AnomalyEvaluator(hqae_strong, device)

    cae_h_accum, cae_d_accum = np.zeros(len(t_healthy)), np.zeros(len(t_damaged))
    none_h_accum, none_d_accum = np.zeros(len(t_healthy)), np.zeros(len(t_damaged))
    strong_h_accum, strong_d_accum = np.zeros(len(t_healthy)), np.zeros(len(t_damaged))

    # 3. Multi-Seed Inference Loop
    print("\nExecuting Ensemble Inference across all seeds:")
    for seed in SEEDS:
        print(f" → Loading and evaluating Seed {seed}")
        
        cae_model.load_state_dict(torch.load(f"{BASE_DIR}/classical/seed_{seed}/best_cae_model.pth", map_location=device))
        hqae_none.load_state_dict(torch.load(f"{BASE_DIR}/quantum/none_d3/seed_{seed}/best_model.pth", map_location=device))
        hqae_strong.load_state_dict(torch.load(f"{BASE_DIR}/quantum/strong_d3/seed_{seed}/best_model.pth", map_location=device))
        
        cae_h_accum += eval_cae.get_anomaly_scores(t_healthy)
        cae_d_accum += eval_cae.get_anomaly_scores(t_damaged)
        
        none_h_accum += eval_none.get_anomaly_scores(t_healthy)
        none_d_accum += eval_none.get_anomaly_scores(t_damaged)
        
        strong_h_accum += eval_strong.get_anomaly_scores(t_healthy)
        strong_d_accum += eval_strong.get_anomaly_scores(t_damaged)

    # Average the scores
    cae_scores = np.concatenate([cae_h_accum / len(SEEDS), cae_d_accum / len(SEEDS)])
    none_scores = np.concatenate([none_h_accum / len(SEEDS), none_d_accum / len(SEEDS)])
    strong_scores = np.concatenate([strong_h_accum / len(SEEDS), strong_d_accum / len(SEEDS)])

    # 4. Visualization
    print("\nInference complete. Generating ROC Curve...")
    y_true = np.concatenate([np.zeros(len(t_healthy)), np.ones(len(t_damaged))])
    
    results_dict = {
        "Classical CAE (Ensemble)": cae_scores,
        "HQAE [None] (Ensemble)": none_scores,
        "HQAE [Strong] (Ensemble)": strong_scores
    }

    plot_roc_curve(
        scores_dict=results_dict,
        y_true=y_true,
        title="Phase 5: Ensemble Synthetic Damage Detection (Incipient)"
    )

if __name__ == "__main__":
    main()