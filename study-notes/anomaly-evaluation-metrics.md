# Anomaly Evaluation Metrics & Protocol

## Overview
SPECTRA-CQ relies on unsupervised learning (the model is only trained on healthy bridge data).
So, it does not output a binary "Damaged" or "Healthy" prediction.
Instead, the Autoencoders output a reconstructed spectrogram. 

To quantify structural degradation, we must mathematically evaluate the difference between the actual physical vibration and the model's reconstructed prediction.
This document outlines the three core pillars of our evaluation protocol: Mean Squared Error (MSE), AUC-ROC and Target Node Isolation.

---

## 1. The Anomaly Score: Mean Squared Error (MSE)
The fundamental metric for all autoencoder-based anomaly detection is the reconstruction error. We quantify this using the Mean Squared Error (MSE).

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$$

*Where $Y_i$ is the original spectrogram pixel energy, $\hat{Y}_i$ is the reconstructed energy, and $n$ is the total number of pixels in the tensor.*

**The Physical Interpretation:**
1. **Healthy Data (the Baseline):**
    * When a normal vibration burst is fed into the model, the latent bottleneck easily summarizes the familiar frequencies.
    * The reconstruction is highly accurate and results in a low, stable MSE. This establishes the structural "noise floor".
2. **Damaged Data (Acoustic Emission):**
    * When a micro-crack occurs, it introduces a high-frequency energy spike that violates the bridge's normal physical behavior.
    * The bottleneck has never seen this spatial pattern and mathematically fails to compress and decompress it.
    * So, the reconstruction is blurred or entirely drops the high-frequency spike and results in a massive, measurable spike in the MSE.

Therefore, **MSE acts as our direct Anomaly Score**. We compute the MSE for both healthy test data and synthetically damaged test data. The mathematical separation between these two MSE distributions is what dictates the model's detection sensitivity.

---

## 2. Thresholding and AUC-ROC (Area Under Curve of Receiver Operating Characteristic Curve)
If MSE is our anomaly score, how do we decide what MSE value triggers a "Bridge Maintenance Alarm"? 

Setting a static threshold is dangerous. 
* If the threshold is too low, wind or heavy traffic will trigger **False Positives** (False Alarms) $\leftarrow$ may cost the city thousands of dollars in unnecessary physical inspections.
* If the threshold is too high, the system will trigger **False Negatives** (Missed Detections) $\leftarrow$ may allow a micro-crack to propagate into a catastrophic failure.

### The ROC Curve
Instead of picking one threshold, we use the **Receiver Operating Characteristic (ROC) Curve**.
The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) across *every possible threshold*.

* **True Positive Rate (Sensitivity):** Out of all the actual cracks, how many did we catch?
* **False Positive Rate (Fall-out):** Out of all the healthy vibrations, how many did we falsely flag as cracks?

### Area Under the Curve (AUC)
To compare the Classical Autoencoder (CAE) against the Hybrid Quantum Autoencoder (HQAE) objectively, we calculate the Area Under the ROC Curve (AUC).

* **AUC = 1.0:** A perfect model. It detected every micro-crack without a single false alarm.
* **AUC = 0.5:** A worthless model. It is mathematically equivalent to flipping a coin (aka a random guess).

> *NOTE:*
> * AUC = 1.0 is a red flag because it indicates the model may have overfitted and memorized noise patterns
> * We ideally aim for AUC between 0.9 and 0.98

By comparing the AUC scores, we can empirically state which architecture is most sensitive to incipient damage while remaining robust against environmental noise.

---

## 3. Phase 4 Protocol: Target Node Isolation
In Phase 3 (Single Sensor), the AUC is calculated by comparing the MSE of a healthy `PE11` sensor against a degraded `PE11` sensor.

In Phase 4, our architecture processes 6 sensors simultaneously. This introduces a critical evaluation hazard: if we calculate the global MSE across all 6 sensors to detect a localized micro-crack at `PE11`, the massive amount of healthy data from the other 5 sensors will mathematically drown out the tiny anomaly at `PE11`.

To solve this, we implemented **Target Node Isolation** in `evaluator.py`.

**The Isolation Protocol:**
1. The 6-sensor spatial graph is passed through the model.
2. The network utilizes the spatial correlation of all 6 nodes to reconstruct the data.
3. During the loss calculation, **we intentionally discard the reconstruction error of sensors PE12 through PE23**.
4. We compute the MSE strictly on the target node (`PE11`).

**Why is this critical?**
This protocol guarantees a mathematically rigorous, apples-to-apples comparison between Phase 3 and Phase 4. 
* The Phase 3 model reconstructs `PE11` using only `PE11`'s data. 
* The Phase 4 model reconstructs `PE11` using the geometric peer pressure of the entire bridge. 
* By scoring both models *only* on their accuracy at `PE11`, we definitively prove whether spatial multi-sensor fusion improves localized anomaly detection.