import re   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_logs(log_file_path: str, title: str = "Training Convergence: MSE Across Seeds"):
    """
    Parses the terminal output log and plots Train vs Validation MSE across all seeds.
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
    
    # Custom color palette explicitly mapping to the requested research tones
    # Teal, Purple, Grey, Pink, Blue, Green
    custom_colors = ['#008080', '#800080', '#808080', '#FF1493', '#0000FF', '#008000']

    for idx, seed in enumerate(unique_seeds):
        seed_epochs = [e for e, s in zip(epochs, seeds) if s == seed]
        seed_train = [t for t, s in zip(train_losses, seeds) if s == seed]
        seed_val = [v for v, s in zip(val_losses, seeds) if s == seed]

        color = custom_colors[idx % len(custom_colors)]

        plt.plot(seed_epochs, seed_train, linestyle="--", alpha=0.6, color=color, label=f"Seed {seed} (Train)")
        plt.plot(seed_epochs, seed_val, linestyle="-", linewidth=2.5, color=color, label=f"Seed {seed} (Val)")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_topology_ablation(results_dict, title="Phase 4a: Topology Ablation (Fixed Depth)"):
    """
    Plots a bar chart comparing the validation MSE of different entanglement topologies.
    
    Args:
        results_dict (dict): Format {"none": 0.031, "basic": 0.025, "strong": 0.008}
    """
    sns.set_theme(style="whitegrid")
    
    # Ensure strict logical ordering
    topologies = ["none", "basic", "strong"]
    mses = [results_dict.get(t, 0) for t in topologies]
    
    plt.figure(figsize=(8, 6))
    
    # Map colors: Grey for none, Purple for basic, Teal for strong
    bar_colors = ['#808080', '#800080', '#008080']
    
    bars = plt.bar(topologies, mses, color=bar_colors, width=0.6)
    
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Entanglement Topology", fontsize=12)
    plt.ylabel("Validation MSE (Log Scale)", fontsize=12)
    plt.yscale("log")
    
    # Add data labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.show()


def plot_depth_ablation(results_dict, title="Phase 4b: Circuit Depth Ablation (Fixed Topology)"):
    """
    Plots a trend line comparing the validation MSE across different quantum layer depths.
    
    Args:
        results_dict (dict): Format {1: 0.021, 3: 0.008, 5: 0.009}
    """
    sns.set_theme(style="darkgrid")
    
    depths = sorted(list(results_dict.keys()))
    mses = [results_dict[d] for d in depths]
    
    plt.figure(figsize=(8, 6))
    
    # Use Pink for the depth trendline
    plt.plot(depths, mses, marker='o', markersize=10, linestyle='-', linewidth=3, color='#FF1493')
    
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Quantum Layers (Depth)", fontsize=12)
    plt.ylabel("Validation MSE (Log Scale)", fontsize=12)
    plt.yscale("log")
    
    # Set explicit x-ticks to avoid decimal layer numbers (e.g., 1.5, 2.5)
    plt.xticks(depths)
    
    # Add data labels
    for i, txt in enumerate(mses):
        plt.annotate(f'{txt:.4f}', (depths[i], mses[i]), textcoords="offset points", 
                     xytext=(0,15), ha='center', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    plt.show()