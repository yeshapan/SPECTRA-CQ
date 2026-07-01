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
    DATA_DIR = "/content/local_spectra_data/processed_spectrograms_6ch"
    print("Fetching file list from local NVMe..")
    
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
    
    # Multi-sensor fusion spatial update:
    # Architectures are re-dimensioned to process the 6-channel structural grid.
    # Each physical node is granted 2 latent dimensions (12-qubit circuit total).
    NUM_SENSORS = 6
    LATENT_DIM_PER_SENSOR = 2
    
    cae = ClassicalAutoencoder(latent_dim_per_sensor=LATENT_DIM_PER_SENSOR, num_sensors=NUM_SENSORS)
    hqae_none = HybridQuantumAutoencoder(latent_dim_per_sensor=LATENT_DIM_PER_SENSOR, num_sensors=NUM_SENSORS, n_quantum_layers=3, topology="none")
    hqae_strong = HybridQuantumAutoencoder(latent_dim_per_sensor=LATENT_DIM_PER_SENSOR, num_sensors=NUM_SENSORS, n_quantum_layers=3, topology="strong")
    
    # Phase 4 Physics-Informed Quantum Topologies
    hqae_bridge_hard = HybridQuantumAutoencoder(latent_dim_per_sensor=LATENT_DIM_PER_SENSOR, num_sensors=NUM_SENSORS, n_quantum_layers=3, topology="bridge_graph_hard")
    hqae_bridge_soft = HybridQuantumAutoencoder(latent_dim_per_sensor=LATENT_DIM_PER_SENSOR, num_sensors=NUM_SENSORS, n_quantum_layers=3, topology="bridge_graph_soft")

    eval_cae = AnomalyEvaluator(cae, device)
    eval_none = AnomalyEvaluator(hqae_none, device)
    eval_strong = AnomalyEvaluator(hqae_strong, device)
    eval_bridge_hard = AnomalyEvaluator(hqae_bridge_hard, device)
    eval_bridge_soft = AnomalyEvaluator(hqae_bridge_soft, device)

    # Accumulators for cross-seed averaging
    cae_h_accum, cae_d_accum = np.zeros(len(t_healthy)), np.zeros(len(t_damaged))
    none_h_accum, none_d_accum = np.zeros(len(t_healthy)), np.zeros(len(t_damaged))
    strong_h_accum, strong_d_accum = np.zeros(len(t_healthy)), np.zeros(len(t_damaged))
    hard_h_accum, hard_d_accum = np.zeros(len(t_healthy)), np.zeros(len(t_damaged))
    soft_h_accum, soft_d_accum = np.zeros(len(t_healthy)), np.zeros(len(t_damaged))

    # 3. Ensemble Evaluation Loop
    BASE_DIR = "/content/drive/MyDrive/spectra-cq-data/checkpoints"
    
    print("Evaluating Frozen Architectures..")
    for seed in SEEDS:
        print(f"\n → Loading checkpoints for Seed {seed}")
        # Load weights (Ensure the target directories map correctly to your training YAML configurations)
        cae.load_state_dict(torch.load(f"{BASE_DIR}/classical/seed_{seed}/best_model.pth", map_location=device))
        hqae_none.load_state_dict(torch.load(f"{BASE_DIR}/quantum/none_d3/seed_{seed}/best_model.pth", map_location=device))
        hqae_strong.load_state_dict(torch.load(f"{BASE_DIR}/quantum/strong_d3/seed_{seed}/best_model.pth", map_location=device))
        
        try:
            hqae_bridge_hard.load_state_dict(torch.load(f"{BASE_DIR}/quantum/bridge_hard/seed_{seed}/best_model.pth", map_location=device))
            hqae_bridge_soft.load_state_dict(torch.load(f"{BASE_DIR}/quantum/bridge_soft/seed_{seed}/best_model.pth", map_location=device))
        except FileNotFoundError:
            print(f"Warning!! Bridge Graph checkpoints for Seed {seed} not found. Skipping in accumulator.")
            continue
        
        cae_h_accum += eval_cae.get_anomaly_scores(t_healthy)
        cae_d_accum += eval_cae.get_anomaly_scores(t_damaged)
        
        none_h_accum += eval_none.get_anomaly_scores(t_healthy)
        none_d_accum += eval_none.get_anomaly_scores(t_damaged)
        
        strong_h_accum += eval_strong.get_anomaly_scores(t_healthy)
        strong_d_accum += eval_strong.get_anomaly_scores(t_damaged)

        hard_h_accum += eval_bridge_hard.get_anomaly_scores(t_healthy)
        hard_d_accum += eval_bridge_hard.get_anomaly_scores(t_damaged)
        
        soft_h_accum += eval_bridge_soft.get_anomaly_scores(t_healthy)
        soft_d_accum += eval_bridge_soft.get_anomaly_scores(t_damaged)

    # Average the scores across the deterministic seeds
    cae_scores = np.concatenate([cae_h_accum / len(SEEDS), cae_d_accum / len(SEEDS)])
    none_scores = np.concatenate([none_h_accum / len(SEEDS), none_d_accum / len(SEEDS)])
    strong_scores = np.concatenate([strong_h_accum / len(SEEDS), strong_d_accum / len(SEEDS)])
    hard_scores = np.concatenate([hard_h_accum / len(SEEDS), hard_d_accum / len(SEEDS)])
    soft_scores = np.concatenate([soft_h_accum / len(SEEDS), soft_d_accum / len(SEEDS)])

    # 4. Visualization
    print("\nInference complete. Generating ROC Curve..")
    y_true = np.concatenate([np.zeros(len(t_healthy)), np.ones(len(t_damaged))])
    
    # Dictionary maps model names to their respective MSE arrays for plotting
    scores_dict = {
        "CAE (Multi-Sensor)": cae_scores,
        "HQAE [Topology: None]": none_scores,
        "HQAE [Topology: Strong]": strong_scores,
        "HQAE [Bridge Graph - Hard]": hard_scores,
        "HQAE [Bridge Graph - Soft]": soft_scores
    }
    
    plot_roc_curve(scores_dict, y_true, title="Multi-Sensor Spatial Damage Detection")

if __name__ == "__main__":
    main()