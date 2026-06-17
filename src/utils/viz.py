import re   # For using Regular Expressions (Regex)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_logs(log_file_path: str, title: str = "Training Convergence: MSE Across Seeds"):
    """
    Parses the terminal output log and plots Train vs Validation MSE across all seeds.
    
    Args:
        log_file_path (str): Path to the saved .log file
        title (str): Dynamic title for the plot (Classical vs Quantum context)
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
    colors = sns.color_palette("husl", len(unique_seeds))

    for idx, seed in enumerate(unique_seeds):
        seed_epochs = [e for e, s in zip(epochs, seeds) if s == seed]
        seed_train = [t for t, s in zip(train_losses, seeds) if s == seed]
        seed_val = [v for v, s in zip(val_losses, seeds) if s == seed]

        plt.plot(seed_epochs, seed_train, linestyle="--", alpha=0.7, color=colors[idx], label=f"Seed {seed} (Train)")
        plt.plot(seed_epochs, seed_val, linestyle="-", linewidth=2, color=colors[idx], label=f"Seed {seed} (Val)")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()