# OpenLAB Research Bridge: Dataset Specifications

The data utilized in this repository is sourced from the openLAB research bridge located in Bautzen, Germany. This structural health monitoring (SHM) dataset captures the baseline dynamics of an undamaged, large-scale concrete structure under near-real-world environmental conditions between February and October 2024.

The complete dataset is available at [10.25532/OPARA-660](https://doi.org/10.25532/OPARA-660)

This project specifically isolates the **Triggered Acceleration Measurements** to establish a classical and quantum anomaly detection baseline.

## Target Telemetry Node
To perform a controlled ablation study comparing classical and quantum autoencoders, this repository isolates a single sensor node to remove spatial correlation variables:
* **Target Node:** `G_ACCZ_PE11_CB0750_0`
* **Measurement:** Z-axis (vertical) acceleration.
* **Location:** Precast element, span 1, girder 1 (7.5 meters from component origin, bottom edge).
* **Hardware:** PCB 393A03 Piezoelectric Sensor via Gantner Instruments Q.station.

## Raw Data Architecture
The pipeline ingests data from the `01_acceleration_trigger` directory of the OpenLAB dataset.
* **Trigger Condition:** Data is recorded only when vibration activity exceeds 2 × 10⁻⁴ m/s².
* **Burst Duration:** 70 seconds per file.
* **Sampling Rate:** 500 Hz.
* **Volume:** 526 raw CSV files.

## Upstream Signal Correction
Before ingestion into the SPECTRA-CQ repository, the raw data underwent standard signal correction by the OpenLAB team:
* **DC Offset Removal:** The median of each measurement was subtracted to correct piezoelectric hardware drift.
* **Bandpass Filtering:** A 4th-order Butterworth filter was applied with a bandwidth of 0.5 Hz to 100 Hz (serving as an anti-aliasing filter and hardware threshold).

## Phase 1 Limitations & Publication Roadmap
The current repository configuration strictly isolates a single sensor (`PE11`). Subsequent phases of this research will expand this data pipeline to include Multi-Sensor Spatial Fusion (integrating `PE11`, `PE12`, and `PE13`) and Synthetic Data Augmentation to stress-test the empirical resilience of the Variational Quantum Circuit (VQC).

## Acknowledgments
The raw dataset is provided by the openLAB research team (Andreas Jansen, et al.) and is available via OPARA under a CC-BY-SA license.