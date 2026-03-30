## **SPECTRA-CQ: Spectrogram Pattern Evaluation via Classical and Quantum Architectures**

This codebase evaluates Hybrid Quantum Convolutional Neural Networks (HQCNNs) against Classical Convolutional Neural Networks (CNNs) for vibration-based Structural Health Monitoring (SHM) using the full-scale openLAB bridge dataset.

### **Architecture & Infrastructure**
To accommodate the scale of the openLAB dataset and the computational demands of quantum simulation, the project utilizes a decoupled data and compute pipeline: 

* **Version Control & Development**: Authored in local IDE (Google Antigravity) and version controlled via GitHub
* **Execution Engine**: Google Colab Pro (GPU environment).
* **Data Storage**: All massive datasets (`openLAB` raw files, generated images) and trained model weights (`.pt` files) are hosted externally on Google Drive.
* **Frameworks**: PyTorch (Classical CNN) and PennyLane (Hybrid Quantum Neural Networks)

### **Directory Structure:**
```
SPECTRA-CQ/
├── .gitignore                                  
├── README.md                                   
├── requirements.txt                            # installed ONLY in the Colab environment
│
├── data/                                       # NOTE: Empty directories. Mapped to Drive in Colab.
│   ├── raw/                                    # 01_acceleration, 02_environment, etc.
│   ├── processed/                              # spectrogram images
│   └── metadata/                               
│
├── documentation/                              # thesis chapters and math foundations
│   ├── 01_theoretical_justification.md         # 1D vs 2D (BiLSTM vs TFR)
│   ├── 02_model_architectures.md               # CNN & HQCNN mechanics
│   └── 03_implementation_plan.md               # step-by-step dataset mapping
│
├── notebooks/                                  # colab notebooks for execution
│   ├── 01_data_extraction.ipynb 
│   ├── 02_tfr_generation.ipynb  
│   ├── 03_classical_cnn.ipynb   
│   └── 04_hqcnn_training.ipynb  
│
├── src/                                        # core modules imported into Colab notebooks
│   ├── __init__.py
│   ├── config.py                
│   ├── data_loader.py           
│   ├── signal_processing.py     
│   ├── models_classical.py      
│   └── models_quantum.py        
│
└── results/                                    # managed in Drive
    ├── ablation_studies/                       # e.g., /4_qubit_vs_8_qubit/, /STFT_vs_CWT/
    ├── figures/                                # final ROC/AUC curves
    └── checkpoints/                            # saved .pt model weights
```