### Computational Implementation and Data Pipeline
While previous chapters established the mathematical foundations of our hybrid diagnostic system, theoretical models cannot process raw structural data without a robust software engineering pipeline. High-frequency structural monitoring generates massive data volumes. These datasets easily overwhelm standard local computers, so we must design a highly optimized cloud-based approach. This chapter details that sequential implementation plan, outlining exactly how raw structural signals from the openLAB dataset flow through a cloud-based compute environment, undergo mathematical transformation, and enter the neural architectures for anomaly detection.

Critically, this plan has been revised to incorporate stress testing data from a destructive hydraulic load test conducted on the openLAB bridge on May 6, 2025. The original implementation plan was designed around the reference-phase monitoring data alone — consisting exclusively of undamaged baseline signals — which limited the project to a purely semi-supervised anomaly detection framework with no ground-truth damage examples. The subsequent discovery and acquisition of the 2025-05-06 load test dataset fundamentally transforms the project: we now possess controlled, physically labeled structural damage data with progressive severity levels, enabling both supervised classification and rigorous quantitative evaluation of detection performance against real damage signatures.

#### 1. Cloud-Based Infrastructure: Compute and Storage Separation
Training deep neural networks requires immense processing power. Because relying on local hardware introduces severe bottlenecks, we deploy a decoupled cloud architecture utilizing Google Colaboratory (Colab) alongside Google Drive. Google Drive acts exclusively as our high-capacity data warehouse, securely storing the massive raw CSV files, generated image datasets, and final model weights. Google Colab serves as the computational engine. By mounting the storage drive directly into the Colab environment, the compute engine streams data continuously. This prevents the system from attempting to load gigabytes of structural data into fragile local memory, which frequently causes catastrophic crashes.

#### 2. Available Data Sources
The project utilizes the complete openLAB monitoring repository, spanning both the undamaged reference phase and a destructive stress test campaign. The repository contains five distinct data categories, each capturing different structural and environmental phenomena. Understanding the physical nature and format of every dataset is critical to designing a comprehensive data pipeline. All reference-phase data resides in `data/2024-02-01_2024-10-31_ida_ki_export/`, while the stress test data resides in `data/20250506_openLAB_tests/`.

##### Summary Inventory

| Sr No. | Directory / Source | Files | Sampling Rate | Period | Sensor Type | Primary Measurement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `01_acceleration_trigger` | 526 CSVs | 500 Hz | 2024-05-01 → 2024-10-31 | PCB 393A03 piezoelectric | Vertical acceleration [m/s²] |
| 2 | `02_environment` | 9 CSVs | 1/600 Hz (10-min) | 2024-02-01 → 2024-10-31 | Thies Clima / Kipp & Zonen SP Lite2 | Temperature [°C], Humidity [%], Solar radiation [W/m²] |
| 3 | `03_tiltmeter` | 9 CSVs | 1/600 Hz (10-min) | 2024-02-01 → 2024-10-31 | Sisgeo 0S542HD0502 | Tilt [mm/m], Temperature [°C], Humidity [%] |
| 4 | `04_tiltmeter_trigger` | 151 CSVs | 5 Hz | 2024-06-05 → 2024-10-31 | Sisgeo 0S542HD0502 | Longitudinal tilt [mm/m] during vehicle load tests |
| 5 | `20250506_openLAB_tests` | 7 TXTs | ~10 Hz | 2025-05-06 | HBK C6A, Baumer OM30, Althen SG, Type K TC | Force [kN], Displacement [mm], Strain [µm/m], Temperature [°C] |

---

##### 2.1 Triggered Acceleration Data (`01_acceleration_trigger`)
The primary dataset for spectrogram-based anomaly detection. The monitoring system captures vertical acceleration signals at 500 Hz from PCB 393A03 piezoelectric sensors whenever ambient excitation exceeds baseline thresholds (range of sensor G_ACCZ_PE11_CB0750_0 exceeding 2 × 10⁻⁴ m/s² within the considered time window), resulting in standardized 70-second data windows. This directory contains 526 CSV files, each recording six sensor channels across six precast elements (PE11, PE12, PE13, PE21, PE22, PE23). The openLAB researchers have already applied a baseline median subtraction and a fourth-order Butterworth bandpass filter (0.5 Hz – 100 Hz) to these files, so the extracted signals are already clear of low-frequency drift and high-frequency anti-aliasing artifacts, allowing us to proceed directly to time-frequency transformation.

| Column | Data Type | Description |
| :--- | :--- | :--- |
| Timestamp | Datetime | ISO 8601 format (UTC) |
| G_ACCZ_PE11_CB0750_0 | Decimal | Acceleration in m/s² — Span 1, Girder 1 |
| G_ACCZ_PE12_CB0750_0 | Decimal | Acceleration in m/s² — Span 1, Girder 2 |
| G_ACCZ_PE13_CB0750_0 | Decimal | Acceleration in m/s² — Span 1, Girder 3 |
| G_ACCZ_PE21_CB0750_0 | Decimal | Acceleration in m/s² — Span 2, Girder 1 |
| G_ACCZ_PE22_CB0750_0 | Decimal | Acceleration in m/s² — Span 2, Girder 2 |
| G_ACCZ_PE23_CB0750_0 | Decimal | Acceleration in m/s² — Span 2, Girder 3 |

Because this dataset exclusively contains data from the undamaged reference phase, it establishes the ground truth for healthy structural behavior. All 70-second windows are assigned a ground-truth label of **0 (Healthy Baseline)**.

##### 2.2 Environmental Climate Data (`02_environment`)
Nine monthly CSV files capture ambient conditions from a climate station mounted on top of the bridge, sampled at 10-minute intervals (1/600 Hz) from February through October 2024. Three channels are recorded:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| Timestamp | Datetime | ISO 8601 format (UTC) |
| G_HTST_ENVR_EN0000_0 | Decimal | Air temperature [°C] |
| G_HTSH_ENVR_EN0000_0 | Decimal | Relative humidity [%] |
| G_PYRS_ENVR_EN0000_0 | Decimal | Solar radiation [W/m²] |

No preprocessing has been applied. This data serves a critical supporting role: environmental conditions (especially temperature and solar radiation) cause thermal deformation that can be confused with structural damage. By correlating acceleration spectrogram features with concurrent environmental measurements, the model can learn to distinguish genuine structural anomalies from benign environmentally-induced variation — a known challenge in operational SHM.

##### 2.3 Continuous Tiltmeter Data (`03_tiltmeter`)
Nine monthly CSV files contain tilt measurements recorded every 10 minutes (1/600 Hz) from February through October 2024 using Sisgeo 0S542HD0502 inclinometers. Each file records 24 channels: for each of the six precast elements (PE11, PE12, PE13, PE21, PE22, PE23), the sensor captures tilt in the transverse direction (TILY), tilt in the longitudinal direction (TILX), temperature (TILT), and humidity (TILH).

| Column Pattern | Data Type | Description |
| :--- | :--- | :--- |
| G_TILY_PExx_CBxxxx_0 | Decimal | Tilt along transverse bridge direction [mm/m] |
| G_TILX_PExx_CBxxxx_0 | Decimal | Tilt along longitudinal bridge direction [mm/m] |
| G_TILT_PExx_CBxxxx_0 | Decimal | Air temperature at sensor location [°C] |
| G_TILH_PExx_CBxxxx_0 | Decimal | Relative humidity at sensor location [%] |

No preprocessing has been applied. This slow-rate tilt data captures the quasi-static deformation behavior of the bridge under environmental and traffic loading. Changes in the long-term tilt baseline following the stress test can serve as an independent indicator of permanent structural damage (residual deformation).

##### 2.4 Vehicle Load Test Tiltmeter Data (`04_tiltmeter_trigger`)
151 CSV files contain triggered tilt measurements recorded at 5 Hz during controlled vehicle load tests. Each test follows a standardized procedure: a test vehicle starts in bridge span 3 near the abutment, drives to span 1, stops near the abutment, returns to its starting position, at a maximum speed of approximately 4 km/h. Each recording captures a complete test cycle (both crossing directions) in a 90-second window. A threshold trigger initiates the measurement. Only the longitudinal tilt channels (TILX) are recorded, covering all six precast elements.

| Column | Data Type | Description |
| :--- | :--- | :--- |
| Timestamp | Datetime | ISO 8601 format (UTC), millisecond precision |
| G_TILX_PE11_CB1100_0 | Decimal | Longitudinal tilt [mm/m] — Span 1, Girder 1 |
| G_TILX_PE12_CB1100_0 | Decimal | Longitudinal tilt [mm/m] — Span 1, Girder 2 |
| G_TILX_PE13_CB1100_0 | Decimal | Longitudinal tilt [mm/m] — Span 1, Girder 3 |
| G_TILX_PE21_CB0400_0 | Decimal | Longitudinal tilt [mm/m] — Span 2, Girder 1 |
| G_TILX_PE22_CB0400_0 | Decimal | Longitudinal tilt [mm/m] — Span 2, Girder 2 |
| G_TILX_PE23_CB0400_0 | Decimal | Longitudinal tilt [mm/m] — Span 2, Girder 3 |

Preprocessing has been applied: the median of the initial 4 seconds is subtracted from each channel to remove temperature effects, and cross-correlation with a reference crossing (Pearson coefficient > 0.85) filters out falsely triggered measurements. This dataset provides an independent, repeatable structural "fingerprint" under a known load. Future vehicle load tests conducted after the stress test will reveal shifts in this fingerprint caused by permanent stiffness loss.

##### 2.5 Stress Testing: Hydraulic Load Test on PE 2.1 (2025-05-06)
The destructive test dataset resides in `data/20250506_openLAB_tests/` and provides a comprehensive package of static stress testing data, analysis code, visual results, and photographic documentation from a controlled load test conducted on May 6, 2025 (Herbers, Richter, Walker & Marx, TU Dresden; DOI: [10.25532/OPARA-852](https://doi.org/10.25532/OPARA-852)). Two hydraulic jacks applied load centrally into the web of precast element PE 2.1 at position x = 24.3 m (the region of maximum bending moment due to dead load), progressively driving the structure through seven incremental displacement targets. The complete folder contents are documented below.

**Directory Structure:**
```
20250506_openLAB_tests/
├── README_EN.md                                    # English documentation
├── README_DE.md                                    # German documentation
├── Herbers_2024_openLAB - A large-scale demonstrator.pdf   # Reference publication
│
├── Data/                                           # 7 raw measurement files
│   ├── MD_2025_05_06_09_08_25.txt                  # Step 1: u_max = 5 mm
│   ├── MD_2025_05_06_10_43_20.txt                  # Step 2: u_max = 10 mm
│   ├── MD_2025_05_06_12_05_10.txt                  # Step 3: u_max = 20 mm
│   ├── MD_2025_05_06_13_43_17.txt                  # Step 4: u_max = 30 mm
│   ├── MD_2025_05_06_16_07_15.txt                  # Step 5: u_max = 40 mm
│   ├── MD_2025_05_06_17_39_40.txt                  # Step 6: u_max = 50 mm
│   └── MD_2025_05_06_18_30_51.txt                  # Step 7: u_max = 60 mm
│
├── Code/
│   └── create_plots.py                             # Python processing & visualization script
│
├── Figures/                                        # Pre-generated analysis plots
│   ├── u-t (c) Max Herbers.png                     # Max displacement vs. time
│   ├── u-t all (c) Max Herbers.png                 # All displacements vs. time
│   ├── F-t (c) Max Herbers.png                     # Force vs. time
│   ├── F-u (c) Max Herbers.png                     # Force vs. displacement (hysteresis)
│   ├── deformation at 18-00-00.png                 # Deformation profile at peak load
│   └── svg/                                        # Vector versions of all figures
│
└── Photos/                                         # 10 high-resolution JPGs of the test setup
    ├── D65A5641_entwickelt.JPG … D65A5853_entwickelt.JPG   # Ground-level photos
    └── DJI_0083_entwickelt.JPG … DJI_0097_entwickelt.JPG   # Aerial drone photos
```

**2.5.1 Measurement Data (`Data/`)**

Seven Catman-format files capture the full loading protocol:

| Step | File | Target u_max | Approx. Peak Force |
| :--- | :--- | :--- | :--- |
| 1 | MD_2025_05_06_09_08_25.txt | 5 mm | ~75 kN |
| 2 | MD_2025_05_06_10_43_20.txt | 10 mm | ~130 kN |
| 3 | MD_2025_05_06_12_05_10.txt | 20 mm | ~210 kN |
| 4 | MD_2025_05_06_13_43_17.txt | 30 mm | ~270 kN |
| 5 | MD_2025_05_06_16_07_15.txt | 40 mm | ~320 kN |
| 6 | MD_2025_05_06_17_39_40.txt | 50 mm | ~360 kN |
| 7 | MD_2025_05_06_18_30_51.txt | 60 mm | ~395 kN |

The files are in Catman format (tab-separated, cp1252 encoding) with 36 header rows and 17 measurement channels. The instrumentation includes:

| Channel Group | Channels | Quantity | Sensor Type |
| :--- | :--- | :--- | :--- |
| **Force** | 4, 5, 17 | Applied load [kN] | HBK C6A load cells (500 kN max) + calculated total |
| **Displacement** | 10–12, 14–15 | Vertical deflection [mm] at five positions along PE 2.1 (x = 7.79, 15.00, 19.68, 24.30, 30.00 m) | Baumer OM30 laser distance sensors |
| **Strain** | 2 | Local strain [µm/m] near prestressing tendon opening (Damage No. 3 location) | Althen strain gauge (L = 120 mm) |
| **Inductive** | 6 | Horizontal displacement [mm], measurement length 160 mm | WETA 1/10 inductive transducer |
| **Temperature** | 7, 8 | Bridge interior (web PE 2.3, midspan) / ambient (50 cm above pavement) [°C] | Type K thermocouples |

The force-displacement curve exhibits clear nonlinear stiffness degradation — the loading and unloading hysteresis loops widen progressively with each step, providing direct physical evidence of accumulating structural damage. For the first five loading steps (up to u_max = 40 mm), the data was zeroed (tared) before measurement. For steps 6 and 7 no taring was performed; the tare values from the initial steps are stored in lines 28–29 of each file and enable calculation of absolute displacements. During load plateaus, laser beam obstruction in the load application area produced outlier readings in channel LWA_4, which must be filtered (removing points where the length change to the previous measurement exceeds 1 mm and total displacement exceeds 65 mm, followed by a moving average with window size 5).

**2.5.2 Reference Publication**

The accompanying paper `Herbers_2024_openLAB - A large-scale demonstrator.pdf` provides the full scientific context for the openLAB bridge as a large-scale demonstrator for structural health monitoring research. It details the bridge design, monitoring system architecture, and the experimental campaign rationale. This publication serves as the primary methodological reference for understanding the loading protocol and sensor placement decisions.

**2.5.3 Analysis Code (`Code/create_plots.py`)**

A Python script (authored by Max Herbers) is included that performs the complete data processing pipeline for the stress test data:
1. **Data import** — reads all seven Catman files via a custom parser that handles cp1252 encoding, extracts start timestamps from line 13, reads tare values from line 28, and concatenates into a unified DataFrame.
2. **Post-processing** — applies the outlier filter described above to LWA_4 (removing jumps > 1 mm where total displacement > 65 mm) and a moving average smoothing filter (window size 5).
3. **Visualization** — generates five diagnostic plots: displacement at load point vs. time (raw + cleaned), all five displacement sensors vs. time, force vs. time, force vs. displacement (hysteresis loops), and a deformation profile (spline-interpolated deflection shape at a selected timestamp).

This script provides a validated reference implementation for parsing the Catman format and can be adapted directly for our ingestion pipeline (Phase I, Step 2).

**2.5.4 Pre-Generated Figures (`Figures/`)**

Five analysis plots have been pre-generated in both PNG (for quick inspection) and SVG (for publication-quality vector graphics) formats:

| Figure | Description | Engineering Significance |
| :--- | :--- | :--- |
| `u-t` | Max displacement (LWA_4) vs. time, raw and cleaned | Shows the seven loading plateaus and the effect of outlier removal |
| `u-t all` | All five displacement sensors vs. time | Reveals the spatial distribution of deflection along the 30 m span |
| `F-t` | Total applied force vs. time | Confirms the progressive loading protocol, peak ~395 kN |
| `F-u` | Force vs. displacement (hysteresis) | **Key diagnostic plot** — widening hysteresis loops directly visualize stiffness degradation and energy dissipation from accumulating damage |
| `deformation at 18-00-00` | Spline-interpolated deflection profile at peak load | Shows the deformed shape of PE 2.1 at 60 mm displacement, with measurement points at the five sensor locations |

These figures provide immediate visual confirmation of the test quality and the progressive nature of the induced damage. The force-displacement hysteresis plot is especially critical: the increasing loop area and decreasing reload stiffness across steps 1–7 constitute unambiguous physical evidence that structural damage accumulates with each successive loading cycle.

**2.5.5 Photographic Documentation (`Photos/`)**

10 high-resolution JPG photographs document the physical test setup, including both ground-level images (camera prefix D65A) showing the hydraulic jack placement, sensor wiring, and visible structural response, and aerial drone images (camera prefix DJI) providing an overhead view of the bridge during loading. These photographs serve as visual verification of the test configuration described in the README files and may be included in the final thesis documentation.

##### 2.6 Supplementary Materials (`05_supplementary_material`)
The repository includes essential engineering context organized into five subdirectories:
* **01_plans:** Construction drawings of the openLAB bridge (PDF, German annotations).
* **02_bim:** Detailed 3D Building Information Models in IFC format — sub-models for bridge geometry, monitoring system layout, reinforcement, scaffolding, and test vehicle. These enable numerical simulation and sensor position verification.
* **03_sensor_datasheets:** Technical datasheets for all sensors in the monitoring system (PDF).
* **04_sensor_installation:** Photographs documenting the physical sensor installation process (JPG), with captions in `captions.md`.
* **05_website:** Offline copy of the official openLAB project website (HTML), including a video tutorial on navigating the IFC models.

---

#### 3. Phase I: Data Extraction and Ingestion (01_data_extraction.ipynb)
The first implementation phase focuses on secure, efficient ingestion of all data sources into a unified pipeline.

* **1. Parsing the Ambient Vibration Signals**
    The extraction script sequentially iterates through the 526 CSV files from `01_acceleration_trigger`. Using optimized data-handling libraries like Pandas, it parses the ISO 8601 UTC timestamps. It then isolates the vertical acceleration columns for each precast element (e.g., G_ACCZ_PE11_CB0750_0), which represent the structural response measured in m/s². Because the signals are already preprocessed (median-subtracted and Butterworth-filtered), we proceed directly to time-frequency transformation.

* **2. Parsing the Stress Test Data**
    A dedicated Catman parser ingests the seven `.txt` files from the load test. The parser must handle the cp1252 encoding, extract the start timestamp from line 13 (format: `T0 = DD.MM.YYYY HH:MM:SS`), read tare values from line 28, skip the 36-row header, and parse tab-separated decimal data using comma as the decimal separator. The script assigns the standardized column schema (Time_1, DMS_1, Force_N, Force_A, IWA, Temp_Bridge, Temp_Ambient, LWA_1–LWA_5, F_total) and concatenates all seven files into a single time-ordered DataFrame. Outlier filtering is applied to the LWA_4 displacement channel as described above.

* **3. Ingesting Environmental Covariates**
    The nine monthly CSV files from `02_environment` are concatenated into a single time-indexed DataFrame containing air temperature, relative humidity, and solar radiation. Because environmental conditions cause thermally-driven quasi-static deformation that can masquerade as structural damage, these covariates are temporally aligned with each triggered acceleration window. For every 70-second vibration event, the nearest 10-minute environmental sample is attached as metadata, enabling downstream models to condition their anomaly scores on ambient state and reduce environmentally-induced false alarms.

* **4. Ingesting Tiltmeter Data**
    The continuous tiltmeter data (9 monthly CSVs from `03_tiltmeter`, 24 channels at 10-minute intervals) is concatenated and serves two purposes: (a) long-term trend monitoring — tracking whether the quasi-static tilt baseline shifts permanently after the 2025-05-06 stress test, and (b) providing additional slow-rate structural features that complement the high-frequency acceleration spectrograms.

    The 151 triggered vehicle load test files from `04_tiltmeter_trigger` (5 Hz, 90-second windows, 6 longitudinal tilt channels) are ingested separately. Each file captures a complete vehicle crossing cycle and constitutes a repeatable structural "fingerprint" under a known moving load. The preprocessing already applied (4-second median subtraction and Pearson cross-correlation filtering > 0.85) ensures only valid crossings enter the pipeline. These tilt fingerprints form a complementary healthy baseline: if post-damage vehicle crossings are later recorded, shifts in the tilt response curve will provide independent validation of anomaly detection results from the acceleration spectrogram pathway.

* **5. Labeling Strategy**
    With stress test data now available, the project transitions from a purely semi-supervised anomaly detection framework to a framework that includes physically labeled damage states. The ambient vibration windows retain their label of **0 (Healthy Baseline)**. The stress test data introduces progressive damage labels corresponding to the seven displacement levels. The exact labeling granularity (binary healthy/damaged vs. multi-class severity grading) is a design decision that will be evaluated during ablation studies:
    * **Binary:** Label 0 = undamaged baseline; Label 1 = any stress test segment.
    * **Multi-class severity:** Labels 0–7, mapping the healthy baseline and each of the seven displacement stages (5 mm through 60 mm) to distinct severity classes, reflecting the progressive stiffness degradation visible in the force-displacement hysteresis.

#### 4. Phase II: Feature Transformation (02_tfr_generation.ipynb)
Feeding raw numerical arrays directly into a neural network forces the CPU to repeat heavy mathematical conversions during every single training loop. A 70-second window sampled at 500 Hz contains 35,000 discrete data points per sensor. Processing this sequentially creates a massive computational bottleneck. To maximize efficiency, we execute all Time-Frequency Representations (TFRs) strictly as a preprocessing step. The Colab compute engine loads the arrays and applies the Short-Time Fourier Transform (STFT) or the Continuous Wavelet Transform (CWT). The resulting two-dimensional matrices are immediately plotted, converted to grayscale, and saved back into the Google Drive warehouse as standalone PNG image files. During subsequent training, the system simply loads these lightweight visual files rather than constantly recalculating Fourier mathematics.

For the stress test data, an additional signal extraction step is required before TFR generation. Because the raw load test files contain quasi-static force-displacement measurements rather than high-frequency vibration, we extract the dynamic structural response embedded within the data. During each loading step, the structure's natural frequencies shift as stiffness degrades. We segment the displacement and strain time-histories into windows aligned with the loading plateaus and transient loading/unloading phases, then compute TFRs of these segmented signals. The spectrograms generated from later loading steps (higher displacement, greater damage) will exhibit visually distinct frequency content compared to the early, low-displacement steps — providing the CNN with spatially separable damage signatures.

#### 5. Phase III: Classical CNN Training (03_classical_cnn.ipynb)
With the visual dataset established, we initialize the classical machine learning pipeline. This phase utilizes the PyTorch framework to establish baseline reconstruction metrics for our classical ablation studies before quantum mechanics are introduced.

* **1. Data Loading and Batching**
    We construct a custom PyTorch DataLoader, which acts as the data management hub. This mechanism randomly shuffles the spectrogram images. Randomization prevents the network from memorizing sequence patterns, ensuring it learns underlying physical features instead. It then groups the images into computational batches and pushes them onto the GPU for rapid parallel processing. With labeled stress test data now available, the DataLoader supports both the autoencoder reconstruction pathway (trained on healthy-only baseline spectrograms) and a supervised classification pathway (trained on the combined healthy + stress test spectrograms with damage labels).

* **2. The Training Loop**
    Two complementary training paradigms are now available:
    * **Semi-supervised (Autoencoder):** The classical network functions as an autoencoder trained exclusively on healthy baseline spectrograms. It compresses each spectrogram into a low-dimensional latent vector, then reconstructs the original image. The reconstruction error for baseline data defines the healthy boundary. Stress test spectrograms — never seen during training — should produce significantly elevated reconstruction errors, flagging them as anomalous.
    * **Supervised (Classifier):** With ground-truth damage labels from the stress test, a classical classifier head can be trained directly on the labeled dataset. This enables standard cross-entropy loss optimization and allows the network to learn the explicit mapping between spectral damage signatures and severity classes.

    Both paradigms are evaluated in ablation studies to determine which approach yields superior damage detection sensitivity with the available data volume.

#### 6. Phase IV: Hybrid Quantum Execution (04_hqcnn_training.ipynb)
The final phase integrates the Variational Quantum Circuit. To bridge the gap between classical deep learning and quantum simulation, we utilize PennyLane. PennyLane is an open-source library specialized for quantum machine learning; it mathematically wraps the quantum circuit so PyTorch treats it like a standard classical layer. During the hybrid forward pass, the classical CNN extracts the latent vector. PennyLane simulates the Angle Encoding to project this vector into the complex Hilbert space, executes the specified CNOT entanglement patterns, and measures the Pauli-Z expectation value. Crucially, PennyLane calculates the quantum gradients analytically, allowing the PyTorch optimizer to backpropagate the error continuously.

With labeled stress test data, the quantum head is now evaluated under both paradigms. In the autoencoder setting, the quantum circuit maps all healthy baseline signals to a highly specific region of the Hilbert space and identifies stress test spectrograms by their distance from this learned manifold. In the supervised setting, the quantum circuit directly classifies spectrograms into healthy and damaged categories, leveraging the exponentially large Hilbert space to achieve separation that classical dense layers cannot.

#### 7. Evaluation Metrics for SHM Anomaly Detection
While overall accuracy is a standard benchmark in computer science, it fails to evaluate anomaly detection frameworks effectively. To ensure the model successfully defines the healthy baseline and penalizes false alarms, we implement strict distance-based evaluation metrics. The availability of labeled stress test data enables a significantly more rigorous evaluation than was previously possible.

| Metric | Mathematical Focus | Practical Engineering Interpretation |
| :--- | :--- | :--- |
| **Reconstruction Error Threshold** | Mean Squared Error (MSE) bounds | Defines the maximum acceptable deviation for a healthy signal. Any spectrogram exceeding this mathematical threshold is flagged as anomalous. |
| **False Positive Rate (FPR)** | $\frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}$ | Evaluates how often the model mistakenly flags a healthy baseline signal as damaged. Minimizes wasted physical inspection resources. |
| **True Positive Rate / Sensitivity** | $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$ | Evaluates whether the model correctly identifies stress test spectrograms as damaged. Critical because missed damage in civil infrastructure carries catastrophic risk. |
| **Damage Severity Correlation** | Spearman's rank correlation between predicted anomaly score and displacement level | Measures whether the model's anomaly score increases monotonically with physical damage severity (5 mm → 60 mm). A strong positive correlation confirms the model captures genuine structural physics rather than statistical artifacts. |
| **Latent Space Clustering (Silhouette Score)** | Distance between data clusters | Measures how tightly the healthy baseline data is grouped and how clearly it separates from stress test data within the quantum Hilbert space, indicating the model's confidence in distinguishing normal from degraded structural states. |