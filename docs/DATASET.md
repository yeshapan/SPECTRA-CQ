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

## Current Data Scope and Integration Roadmap
The raw OpenLAB dataset contains multi-node telemetry, but the current repository configuration intentionally isolates a single sensor (`PE11`). 

This restriction establishes a controlled baseline to cleanly evaluate the quantum entanglement ablation without the confounding variable of spatial correlation between different sensors.

**Future Data Integration:**
* **Leg 2:** Maintain single-sensor isolation (`PE11`) while implementing Local Cost Functions to break the quantum barren plateau.
* **Leg 3:** Expand pipeline ingestion to execute Multi-Sensor Spatial Fusion (integrating `PE11`, `PE12`,\ and `PE13`) through the optimized quantum architecture.
* **Leg 4:** Introduce Synthetic Data Augmentation to the pipeline to stress-test empirical resilience.

## Acknowledgments
The raw dataset is provided by the openLAB research team (Andreas Jansen, et al.) and is available via OPARA under a CC-BY-SA license.