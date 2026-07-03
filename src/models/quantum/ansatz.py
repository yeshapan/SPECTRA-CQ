import pennylane as qml
import numpy as np

# Spatial coordinates of the openLAB sensors (X=span length, Y=transverse girder spacing)
# This dictates the physical attenuation of structural waves
SENSOR_COORDS = {
    0: (0.0, 0.0),   # PE11 (Mapped to Q0, Q1)
    1: (0.0, 1.5),   # PE12 (Mapped to Q2, Q3)
    2: (0.0, 3.0),   # PE13 (Mapped to Q4, Q5)
    3: (15.0, 0.0),  # PE21 (Mapped to Q6, Q7)
    4: (15.0, 1.5),  # PE22 (Mapped to Q8, Q9)
    5: (15.0, 3.0)   # PE23 (Mapped to Q10, Q11)
}

def _get_distance(sensor_a, sensor_b):
    """Calculates strict Euclidean distance between two physical sensors."""
    xa, ya = SENSOR_COORDS[sensor_a]
    xb, yb = SENSOR_COORDS[sensor_b]
    return np.sqrt((xa - xb)**2 + (ya - yb)**2)

def build_ansatz(weights, wires, topology="basic"):
    """
    Parameterized Quantum Circuit (VQC Ansatz).
    Dynamically implements different entanglement topologies for ablation studies.
    
    Args:
        weights (Tensor): Trainable angles for the rotation gates. 
        wires (Iterable): The target qubits.
    """
    if topology == "none":
        # Brute-force/Naive approach: No entanglement.
        # Applies independent RX rotations. Proves if quantum correlation is even mathematically necessary.
        layers = weights.shape[0]
        for layer in range(layers):
            for i, wire in enumerate(wires):
                qml.RX(weights[layer, i], wires=wire)
                
    elif topology == "strong":
        # Strongly entangling layers. All-to-all connectivity.
        # Tests if unrestricted global entanglement improves or degrades structural mapping.
        qml.StronglyEntanglingLayers(weights, wires=wires)
        
    elif topology == "bridge_graph_hard":
        # Physics-Informed Hard Cutoff
        # Forbids entanglement between sensors further than 15.0m apart.
        layers = weights.shape[0]
        for layer in range(layers):
            # 1. Feature Binding
            for s in range(6):
                qml.CNOT(wires=[s*2, s*2 + 1])
                
            # 2. Distance-Bounded Spatial Mapping
            weight_idx = 0
            for s1 in range(6):
                for s2 in range(s1 + 1, 6):
                    dist = _get_distance(s1, s2)
                    if dist <= 15.0:
                        # Controlled-Y rotation parameterized by the classical optimizer
                        qml.CRY(weights[layer, weight_idx], wires=[s1*2 + 1, s2*2 + 1])
                    weight_idx += 1
                    
    elif topology == "bridge_graph_soft":
        # Physics-Informed Neural Network (PINN) Pivot
        # The quantum circuit is now allowed full gradient expressivity.
        # The physical dampening (gradient starvation fix) will be handled by the classical PINN Loss Function in src/engine/trainer.py instead of artificially scaling parameters here.
        layers = weights.shape[0]
        for layer in range(layers):
            # 1. Intra-Sensor Feature Binding
            for s in range(6):
                qml.CNOT(wires=[s*2, s*2 + 1])
                
            # 2. Distance-Weighted Spatial Mapping (Now Unshackled)
            weight_idx = 0
            for s1 in range(6):
                for s2 in range(s1 + 1, 6):
                    # We map the pair using the full un-attenuated parameter.
                    # This guarantees the Adam optimizer receives a 100% strength gradient.
                    qml.CRY(weights[layer, weight_idx], wires=[s1*2 + 1, s2*2 + 1])
                    weight_idx += 1
                    
    else:
        raise ValueError(f"Topology '{topology}' is not supported.")