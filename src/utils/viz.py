import re   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from sklearn.metrics import roc_curve, auc

COLOR_PALETTE = ['firebrick', 'steelblue', 'olivedrab', 'darkgoldenrod', 'indigo']

def plot_training_logs(log_file_path: str, title: str = "Training Convergence: MSE Across Seeds", y_lim=(0.005, 0.055)):
    """
    Parses the terminal output log and plots Train vs Validation MSE across all seeds.
    Y-axis limits are fixed to ensure accurate visual comparison across different topologies.
    """
    sns.set_theme(style="darkgrid")
    
    epochs = []
    train_losses = []
    val_losses = []
    seeds = []
    current_seed = None
    
    seed_pattern = re.compile(r"Starting Protocol Run \| Seed: (\d+)")
    epoch_pattern = re.compile(r"Epoch (\d+)/\d+ \| Train MSE: ([\d.]+) \| Val MSE: ([\d.]+)")

    with open(log_file_path, "r") as f:
        for line in f:
            seed_match = seed_pattern.search(line)
            if seed_match:
                current_seed = int(seed_match.group(1))
                
            epoch_match = epoch_pattern.search(line)
            if epoch_match and current_seed is not None:
                epochs.append(int(epoch_match.group(1)))
                train_losses.append(float(epoch_match.group(2)))
                val_losses.append(float(epoch_match.group(3)))
                seeds.append(current_seed)

    if not epochs:
        print("No training data found in log file.")
        return

    plt.figure(figsize=(10, 6))
    unique_seeds = list(set(seeds))
    ax = plt.gca()
    
    for idx, seed in enumerate(unique_seeds):
        seed_epochs = [e for e, s in zip(epochs, seeds) if s == seed]
        seed_train = [t for t, s in zip(train_losses, seeds) if s == seed]
        seed_val = [v for v, s in zip(val_losses, seeds) if s == seed]
        
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        
        plt.plot(seed_epochs, seed_train, linestyle='--', color=color, alpha=0.5, label=f'Seed {seed} (Train)')
        plt.plot(seed_epochs, seed_val, marker='o', color=color, linewidth=2, label=f'Seed {seed} (Val)')

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    
    # LOCK AXES SCALING
    plt.xlim(1, 15)
    plt.ylim(y_lim)
    
    # Increase axis tick density
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_ablation_results(results_dict, title="Phase 4a: Topology Ablation Validation MSE"):
    """
    Plots a bar chart comparing the final validation MSE across different architectures/topologies.
    """
    sns.set_theme(style="darkgrid")
    
    architectures = list(results_dict.keys())
    mses = list(results_dict.values())
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    bars = plt.bar(architectures, mses, color=COLOR_PALETTE[:len(architectures)])
    
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Validation MSE (Log Scale)", fontsize=12)
    plt.yscale("log")
    
    # Increase Y-axis tick density for log scale
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    # Add data labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.show()

def plot_depth_ablation(results_dict, title="Phase 4b: Circuit Depth Ablation (Fixed Topology)", y_lim=(0.005, 0.025)):
    """
    Plots a trend line comparing the validation MSE across different quantum layer depths.
    Fixed Y-axis limits prevent exaggerated scaling on minute differences.
    """
    sns.set_theme(style="darkgrid")
    
    depths = sorted(list(results_dict.keys()))
    mses = [results_dict[d] for d in depths]
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    plt.plot(depths, mses, marker='o', markersize=10, linestyle='-', linewidth=3, color=COLOR_PALETTE[3])
    
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Quantum Layers (Depth)", fontsize=12)
    plt.ylabel("Validation MSE", fontsize=12)
    
    # LOCK Y-AXIS
    plt.ylim(y_lim)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=len(depths)*2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(scores_dict: dict, y_true: np.ndarray, title: str = "ROC Curve: Synthetic Damage Detection"):
    """
    Plots the AUC-ROC curve for an arbitrary number of models.
    
    Args:
        scores_dict: Dictionary mapping model names to their concatenated MSE score arrays.
        y_true: Ground truth binary labels (0 = Healthy, 1 = Damaged).
    """
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(9, 7))

    for idx, (model_name, scores) in enumerate(scores_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        
        # Cycle through the Leg-2 palette
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')

    # LOCK AXES SCALING
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate (False Alarms)', fontsize=12)
    plt.ylabel('True Positive Rate (Correct Detections)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_multi_sensor_spectrograms(tensor: np.ndarray, title: str = "6-Sensor Spatial Spectrogram Synchronization"):
    """
    Visualizes the (6, 64, 1000) multi-channel tensor across a 3x2 grid.
    Maps the tensor indices geometrically to the physical bridge layout to verify spatial integrity.
    
    Args:
        tensor (np.ndarray): The processed CWT tensor of shape (6, 64, 1000).
    """
    sns.set_theme(style="darkgrid")
    
    # Bridge geometry mapping (Index -> Physical Sensor)
    sensor_labels = [
        "PE11 (Span 1, Girder 1)", "PE12 (Span 1, Girder 2)", "PE13 (Span 1, Girder 3)",
        "PE21 (Span 2, Girder 1)", "PE22 (Span 2, Girder 2)", "PE23 (Span 2, Girder 3)"
    ]
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    
    # Define a consistent color mapping limit across all sensors to preserve relative energy deltas
    vmin, vmax = tensor.min(), tensor.max()
    
    for idx, ax in enumerate(axes.flatten()):
        # Plot the 2D energy matrix for the specific sensor channel
        im = ax.imshow(tensor[idx], aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
        
        ax.set_title(sensor_labels[idx], fontsize=12, fontweight="bold", color=COLOR_PALETTE[1])
        ax.set_ylabel("Frequency Scale (Log)", fontsize=10)
        
        # Format axes
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        
        if idx >= 4:  # Only add X-axis labels to the bottom row
            ax.set_xlabel("Time (Samples)", fontsize=10)
            
    # Add a unified colorbar for the entire physical structure
    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Normalized Energy Magnitude")
    
    plt.tight_layout()
    plt.show()