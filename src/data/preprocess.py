import os
import glob
import argparse
import logging
import numpy as np
import pandas as pd
import scipy.signal as signal
from tqdm import tqdm
import warnings

# Suppress SciPy Deprecation warnings to prevent terminal I/O bottlenecking during CWT loops
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Phase-4 Update: Hardcode the 6 target sensor columns based on the OpenLAB naming convention
# This replaces the single-sensor CLI argument approach.
TARGET_SENSORS = [
    "G_ACCZ_PE11_CB0750_0", "G_ACCZ_PE12_CB0750_0", "G_ACCZ_PE13_CB0750_0",
    "G_ACCZ_PE21_CB0750_0", "G_ACCZ_PE22_CB0750_0", "G_ACCZ_PE23_CB0750_0"
]

def compute_cwt(signal_window: np.ndarray, fs: int = 500) -> np.ndarray:
    """
    Computes the CWT using a complex Morlet wavelet
    
    Mathematical Justification:
    The dataset is pre-filtered between 0.5 Hz and 100 Hz. We generate 64 logarithmic
    scales to capture this specific bandwidth. We use omega-zero (w) = 6.0 to satisfy the 
    admissibility condition of the Morlet wavelet, balancing time and frequency resolution.
    
    Args:
        signal_window (np.ndarray): 1D array of acceleration data. Shape: (N,)
        fs (int): Sampling frequency in Hz. Default 500 Hz for openLAB data.
        
    Returns:
        np.ndarray: 2D magnitude spectrogram. Shape: (Scales, N) -> (64, 1000) for a 2s window.
    """
    # Define 64 logarithmically spaced frequencies matching the Butterworth filter band
    frequencies = np.geomspace(0.5, 100, num=64)
    w = 6.0 
    
    # Convert frequencies to wavelet scales
    scales = w * fs / (2 * np.pi * frequencies)
    
    # FIX: scipy.signal.cwt returns a single ndarray, not a tuple
    coefficients = signal.cwt(signal_window, signal.morlet2, widths=scales, w=w)
    
    return np.abs(coefficients)

def process_dataset(input_dir: str, output_dir: str, window_size_sec: int = 2, fs: int = 500):
    """
    Segments raw 70-sec acceleration bursts into discrete matrices.
    
    Hardware Constraint: 
    Iterates file-by-file. We do not concatenate DataFrames in memory.
    Saves outputs as highly compressed .npy binaries rather than images.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        logging.error(f"FATAL: No CSV files found in {input_dir}. Check mount path.")
        return

    logging.info(f"Initialized processing pipeline for {len(csv_files)} files. Target nodes: 6-Sensor Fusion.")
    
    samples_per_window = window_size_sec * fs
    global_window_count = 0

    for file_path in tqdm(csv_files, desc="Applying CWT via Morlet Kernel"):
        try:
            # IO Optimization: Load all 6 Z-axis vectors simultaneously. Bypasses timestamp parsing string overhead.
            # We pass the full TARGET_SENSORS list instead of a single sensor_col
            df = pd.read_csv(file_path, usecols=TARGET_SENSORS)
            df = df.dropna()
            
            # Extract numpy arrays for all 6 sensors
            raw_signals = {sensor: df[sensor].values for sensor in TARGET_SENSORS}
            
            # Calculate strict window bounds (drops trailing fractional windows)
            # We'll use the first sensor's length as the master bound (they are synchronized)
            num_windows = len(raw_signals[TARGET_SENSORS[0]]) // samples_per_window
            
            for i in range(num_windows):
                start_idx = i * samples_per_window
                end_idx = start_idx + samples_per_window
                
                # Generate CWT for each sensor and stack them
                multi_channel_spec = []
                for sensor in TARGET_SENSORS:
                    window = raw_signals[sensor][start_idx:end_idx]
                    spectrogram = compute_cwt(window, fs=fs) # Shape: (64, 1000)
                    multi_channel_spec.append(spectrogram)
                
                # Stack into a single tensor of shape (6, 64, 1000)
                # Cast to float32 to cut GPU VRAM usage in half during training (float64 is overkill)
                spectrogram_stack = np.array(multi_channel_spec, dtype=np.float32)
                
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                save_name = f"{base_name}_win{i:03d}.npy"
                save_path = os.path.join(output_dir, save_name)
                
                np.save(save_path, spectrogram_stack)
                global_window_count += 1
                
        except Exception as e:
            logging.warning(f"File skipped due to parsing/CWT failure [{file_path}]: {e}")

    logging.info(f"Pipeline complete. Yielded {global_window_count} multi-channel spectrograms (float32).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # Removed the --sensor argument because we are hardcoding the 6-sensor fusion
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir, args.window)